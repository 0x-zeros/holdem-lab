"""Draw analysis and outs calculation for Texas Hold'em."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Sequence

from holdem_lab.cards import Card, Rank, Suit, FULL_DECK
from holdem_lab.evaluator import evaluate_hand, HandType


class DrawType(IntEnum):
    """Types of drawing hands."""

    # Flush draws
    FLUSH_DRAW = auto()        # 4 to a flush (9 outs)
    BACKDOOR_FLUSH = auto()    # 3 to a flush (needs 2 running cards)

    # Straight draws
    OPEN_ENDED = auto()        # 8 outs (e.g., 5-6-7-8)
    GUTSHOT = auto()           # 4 outs (e.g., 5-6-8-9, needs 7)
    DOUBLE_GUTSHOT = auto()    # 8 outs (e.g., 5-7-8-T, needs 6 or 9)
    BACKDOOR_STRAIGHT = auto() # 3 connected cards (needs 2 running)


@dataclass(frozen=True, slots=True)
class FlushDraw:
    """Information about a flush draw."""

    suit: Suit
    cards_held: int           # 3 for backdoor, 4+ for flush draw
    outs: tuple[Card, ...]    # Specific cards that complete the flush
    is_nut: bool              # True if hero has the Ace of this suit

    @property
    def out_count(self) -> int:
        """Number of outs for this draw."""
        return len(self.outs)

    @property
    def draw_type(self) -> DrawType:
        """Type of flush draw."""
        return DrawType.FLUSH_DRAW if self.cards_held >= 4 else DrawType.BACKDOOR_FLUSH


@dataclass(frozen=True, slots=True)
class StraightDraw:
    """Information about a straight draw."""

    draw_type: DrawType       # OPEN_ENDED, GUTSHOT, DOUBLE_GUTSHOT, BACKDOOR
    needed_ranks: tuple[int, ...]  # Ranks that complete the straight
    outs: tuple[Card, ...]    # Specific cards that complete (accounts for dead cards)
    high_card: int            # Highest possible straight if completed
    is_nut: bool              # True if this would make the nut straight (Broadway)

    @property
    def out_count(self) -> int:
        """Number of outs for this draw."""
        return len(self.outs)


@dataclass(frozen=True, slots=True)
class DrawAnalysis:
    """Complete draw analysis for a hand."""

    hole_cards: tuple[Card, ...]
    board: tuple[Card, ...]

    # Made hand status
    has_flush: bool
    has_straight: bool

    # Draw information
    flush_draws: tuple[FlushDraw, ...]
    straight_draws: tuple[StraightDraw, ...]

    # Aggregate outs (no double-counting)
    total_outs: int
    all_outs: tuple[Card, ...]

    @property
    def has_draw(self) -> bool:
        """Check if any draws exist."""
        return bool(self.flush_draws or self.straight_draws)

    @property
    def is_combo_draw(self) -> bool:
        """Check if has both flush and straight draw."""
        return bool(self.flush_draws and self.straight_draws)


def _analyze_flush_draws(
    all_cards: list[Card],
    hole_cards: Sequence[Card],
    dead_cards: frozenset[Card],
    board_size: int,
) -> list[FlushDraw]:
    """Detect flush draws and calculate outs."""
    # Group cards by suit
    by_suit: dict[Suit, list[Card]] = {s: [] for s in Suit}
    for card in all_cards:
        by_suit[card.suit].append(card)

    flush_draws: list[FlushDraw] = []
    known_cards = set(all_cards) | dead_cards

    for suit, cards in by_suit.items():
        count = len(cards)

        # Need at least 3 cards for any flush draw
        if count < 3:
            continue

        # Backdoor flush only valid on flop (3 cards on board)
        if count == 3 and board_size > 3:
            continue

        # 5+ cards means we already have a flush (not a draw)
        if count >= 5:
            continue

        # Calculate outs for this suit
        outs: list[Card] = []
        for rank in Rank:
            card = Card(rank, suit)
            if card not in known_cards:
                outs.append(card)

        # Check if hero has the nut flush draw:
        # 1. Hero holds the Ace of this suit, OR
        # 2. Ace is on the board (part of all_cards but not hole_cards), OR
        # 3. Ace is dead (unavailable to opponents)
        ace_of_suit = Card(Rank.ACE, suit)
        hole_set = set(hole_cards)
        hero_has_ace = ace_of_suit in hole_set
        ace_on_board = ace_of_suit in set(all_cards) and ace_of_suit not in hole_set
        ace_is_dead = ace_of_suit in dead_cards
        is_nut = hero_has_ace or ace_on_board or ace_is_dead

        flush_draws.append(FlushDraw(
            suit=suit,
            cards_held=count,
            outs=tuple(sorted(outs, key=lambda c: -c.rank.value)),
            is_nut=is_nut,
        ))

    return flush_draws


def _build_rank_mask(cards: Sequence[Card]) -> int:
    """
    Build a 14-bit rank bitmask.

    Bit positions:
    - Bit 0: Ace (low, for wheel detection)
    - Bits 1-12: 2 through K
    - Bit 13: Ace (high)
    """
    mask = 0
    for card in cards:
        # Map rank 2-14 to bits 1-13
        mask |= 1 << (card.rank.value - 1)

    # Duplicate Ace at position 0 for wheel detection
    if mask & (1 << 13):  # Ace at bit 13
        mask |= 1  # Also set bit 0

    return mask


def _count_bits(n: int) -> int:
    """Count set bits in an integer."""
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count


def _get_straight_outs(
    needed_ranks: list[int],
    known_cards: set[Card],
) -> list[Card]:
    """Get all cards that complete a straight for the needed ranks."""
    outs: list[Card] = []
    for rank_val in needed_ranks:
        rank = Rank(rank_val)
        for suit in Suit:
            card = Card(rank, suit)
            if card not in known_cards:
                outs.append(card)
    return outs


def _analyze_straight_draws(
    all_cards: list[Card],
    dead_cards: frozenset[Card],
    board_size: int,
) -> list[StraightDraw]:
    """Detect straight draws using bitmask approach."""
    rank_mask = _build_rank_mask(all_cards)
    known_cards = set(all_cards) | dead_cards
    straight_draws: list[StraightDraw] = []

    # Track which 4-card patterns we've found to detect OESD
    # Key: tuple of 4 consecutive rank values, Value: list of needed ranks
    four_card_patterns: dict[tuple[int, ...], list[int]] = {}

    # Check for 4-card straight patterns (OESD, gutshot)
    for high in range(5, 15):  # 5-high (wheel) through A-high
        if high == 5:
            # Wheel window: bits 0-4 (A-2-3-4-5)
            window = rank_mask & 0b11111
        else:
            # Shift to get 5-bit window ending at 'high'
            low_bit = high - 4
            window = (rank_mask >> (low_bit - 1)) & 0b11111

        bit_count = _count_bits(window)

        if bit_count == 5:
            # Already have a straight, skip
            continue

        if bit_count == 4:
            # Find the gap position
            gap_positions = [i for i in range(5) if not (window & (1 << i))]
            if len(gap_positions) != 1:
                continue
            gap = gap_positions[0]

            # Calculate the needed rank
            if high == 5:
                # Wheel: gap positions 0-4 map to A(14), 2, 3, 4, 5
                rank_map = [14, 2, 3, 4, 5]
                needed_rank = rank_map[gap]
                # The 4 held ranks for wheel pattern
                held_ranks = tuple(r for r in [14, 2, 3, 4, 5] if r != needed_rank)
            else:
                needed_rank = (high - 4) + gap
                # The 4 held ranks
                held_ranks = tuple(r for r in range(high - 4, high + 1) if r != needed_rank)

            # Track this pattern
            held_ranks_key = tuple(sorted(held_ranks))
            if held_ranks_key not in four_card_patterns:
                four_card_patterns[held_ranks_key] = []
            four_card_patterns[held_ranks_key].append(needed_rank)

    # Now classify patterns
    for held_ranks, needed_ranks in four_card_patterns.items():
        needed_ranks_list = sorted(set(needed_ranks))

        # Check if this is a wheel pattern (A-2-3-4 or 2-3-4-5)
        is_wheel_pattern = 14 in held_ranks and 2 in held_ranks

        if len(needed_ranks_list) == 2:
            # Two different ranks can complete - this is OESD
            all_outs = _get_straight_outs(needed_ranks_list, known_cards)
            # High card is the max of completed straights
            if is_wheel_pattern:
                # Wheel OESD: A-2-3-4 needs 5 (wheel) or could be part of higher straight
                high_card = max(needed_ranks_list)
            else:
                high_card = max(held_ranks) + 1 if max(needed_ranks_list) > max(held_ranks) else max(held_ranks)
            straight_draws.append(StraightDraw(
                draw_type=DrawType.OPEN_ENDED,
                needed_ranks=tuple(needed_ranks_list),
                outs=tuple(all_outs),
                high_card=high_card,
                is_nut=(high_card == 14),
            ))
        elif len(needed_ranks_list) == 1:
            # Only one rank can complete - gutshot (or single-ended OESD-like)
            needed_rank = needed_ranks_list[0]

            # Calculate high card of completed straight
            if is_wheel_pattern and needed_rank == 5:
                # Wheel completion
                high_card = 5
            else:
                # For non-wheel: the completed straight's high card
                # Get all ranks that would be in the completed straight
                all_straight_ranks = sorted(held_ranks) + [needed_rank]
                all_straight_ranks = sorted(all_straight_ranks)
                # Filter to only keep 5 consecutive
                if 14 in all_straight_ranks and 2 in all_straight_ranks:
                    # This has A and 2, could be wheel
                    if needed_rank <= 5:
                        high_card = 5
                    else:
                        high_card = max(all_straight_ranks)
                else:
                    high_card = max(all_straight_ranks)

            # Determine if gap is internal or at edge
            min_held = min(r for r in held_ranks if r != 14)  # Ignore Ace for min calc in wheel
            max_held = max(held_ranks)
            if is_wheel_pattern:
                # For wheel patterns, gap at position 5 is "edge" (OESD-like)
                draw_type = DrawType.GUTSHOT
            elif needed_rank < min_held or needed_rank > max_held:
                draw_type = DrawType.GUTSHOT
            else:
                draw_type = DrawType.GUTSHOT

            outs = _get_straight_outs(needed_ranks_list, known_cards)
            straight_draws.append(StraightDraw(
                draw_type=draw_type,
                needed_ranks=(needed_rank,),
                outs=tuple(outs),
                high_card=high_card,
                is_nut=(high_card == 14),
            ))

    # Check for double gutshot (4 cards spanning 6 ranks)
    # Only meaningful when more cards are to come (not on river)
    if board_size < 5:
        double_gutshots = _detect_double_gutshots(rank_mask, known_cards)
        straight_draws.extend(double_gutshots)

    # Check for backdoor straights (3 connected cards, only on flop)
    # But only if we don't already have 4+ card straight draw
    if board_size <= 3 and not straight_draws:
        backdoor = _detect_backdoor_straights(rank_mask, known_cards)
        straight_draws.extend(backdoor)

    # Remove duplicate draws (keep the best version of each)
    return _deduplicate_straight_draws(straight_draws)


def _detect_double_gutshots(
    rank_mask: int,
    known_cards: set[Card],
) -> list[StraightDraw]:
    """Detect double gutshot patterns (4 cards spanning 6 ranks, 2 internal gaps)."""
    double_gutshots: list[StraightDraw] = []

    # Check 6-rank windows
    for high in range(7, 15):  # 7-high through A-high
        if high == 7:
            # Special case for wheel-adjacent: A-2-3-4-5-6
            # Bits: 0 (A low), 1 (2), 2 (3), 3 (4), 4 (5), 5 (6)
            # But we can use simplified approach
            low_bit = 1  # Start at 2
            window = (rank_mask >> (low_bit - 1)) & 0b111111
            # Also check if Ace is present for wheel scenarios
            if rank_mask & 1:  # Ace low bit
                window |= 0b100000  # Add Ace at high end of this window
        else:
            low_bit = high - 5
            window = (rank_mask >> (low_bit - 1)) & 0b111111

        bit_count = _count_bits(window)

        if bit_count != 4:
            continue

        # Find gap positions
        gaps = [i for i in range(6) if not (window & (1 << i))]
        if len(gaps) != 2:
            continue

        # Both gaps must be internal (not at positions 0 or 5)
        if 0 in gaps or 5 in gaps:
            continue

        # Calculate needed ranks
        low_rank = high - 5
        needed_ranks = [low_rank + g for g in gaps]

        # Ensure ranks are valid (2-14)
        if any(r < 2 or r > 14 for r in needed_ranks):
            continue

        outs = _get_straight_outs(needed_ranks, known_cards)
        if outs:
            double_gutshots.append(StraightDraw(
                draw_type=DrawType.DOUBLE_GUTSHOT,
                needed_ranks=tuple(needed_ranks),
                outs=tuple(outs),
                high_card=high,
                is_nut=(high == 14),
            ))

    return double_gutshots


def _detect_backdoor_straights(
    rank_mask: int,
    known_cards: set[Card],
) -> list[StraightDraw]:
    """Detect backdoor straight draws (3 connected cards)."""
    backdoor: list[StraightDraw] = []

    # Look for 3 consecutive ranks
    for high in range(4, 15):  # 4-high through A-high
        if high == 4:
            # Could be A-2-3 (wheel start)
            window = rank_mask & 0b111  # bits 0,1,2 = A,2,3
            if _count_bits(window) == 3:
                backdoor.append(StraightDraw(
                    draw_type=DrawType.BACKDOOR_STRAIGHT,
                    needed_ranks=(4, 5),  # Need 4 and 5 for wheel
                    outs=tuple(_get_straight_outs([4, 5], known_cards)),
                    high_card=5,
                    is_nut=False,
                ))
        else:
            low_bit = high - 2
            window = (rank_mask >> (low_bit - 1)) & 0b111
            if _count_bits(window) == 3:
                # 3 consecutive cards found
                # Need the two cards on either end to complete
                needed = [high - 3, high + 1] if high < 14 else [high - 3]
                needed = [r for r in needed if 2 <= r <= 14]
                if needed:
                    outs = _get_straight_outs(needed, known_cards)
                    if outs:
                        backdoor.append(StraightDraw(
                            draw_type=DrawType.BACKDOOR_STRAIGHT,
                            needed_ranks=tuple(needed),
                            outs=tuple(outs),
                            high_card=min(high + 1, 14),
                            is_nut=(high + 1 >= 14),
                        ))

    return backdoor


def _deduplicate_straight_draws(draws: list[StraightDraw]) -> list[StraightDraw]:
    """Remove duplicate or redundant straight draws."""
    if not draws:
        return draws

    # Group by high card and draw type, keep best (most outs)
    best: dict[tuple[int, DrawType], StraightDraw] = {}
    for draw in draws:
        key = (draw.high_card, draw.draw_type)
        if key not in best or draw.out_count > best[key].out_count:
            best[key] = draw

    # Also deduplicate OESD that might be counted from different windows
    # Keep unique by needed_ranks
    unique: dict[tuple[int, ...], StraightDraw] = {}
    for draw in best.values():
        key = draw.needed_ranks
        if key not in unique or draw.out_count > unique[key].out_count:
            unique[key] = draw

    return list(unique.values())


def _check_made_hands(all_cards: list[Card]) -> tuple[bool, bool]:
    """Check if we already have a flush or straight."""
    if len(all_cards) < 5:
        return False, False

    hand_rank = evaluate_hand(all_cards)

    has_flush = hand_rank.hand_type in (
        HandType.FLUSH,
        HandType.STRAIGHT_FLUSH,
        HandType.ROYAL_FLUSH,
    )
    has_straight = hand_rank.hand_type in (
        HandType.STRAIGHT,
        HandType.STRAIGHT_FLUSH,
        HandType.ROYAL_FLUSH,
    )

    return has_flush, has_straight


def analyze_draws(
    hole_cards: Sequence[Card],
    board: Sequence[Card],
    dead_cards: Sequence[Card] = (),
) -> DrawAnalysis:
    """
    Analyze drawing possibilities for a hand.

    Args:
        hole_cards: Hero's 2 hole cards.
        board: Current community cards (0-5 cards).
        dead_cards: Known dead cards (opponent mucked cards, etc.)

    Returns:
        DrawAnalysis with complete draw information.

    Raises:
        ValueError: If invalid card count or duplicate cards.

    Example:
        >>> from holdem_lab import parse_cards
        >>> hole = parse_cards("9h 8h")
        >>> board = parse_cards("7h 6c 2h")
        >>> analysis = analyze_draws(hole, board)
        >>> analysis.flush_draws[0].out_count
        9
    """
    if len(hole_cards) != 2:
        raise ValueError(f"Must provide exactly 2 hole cards, got {len(hole_cards)}")
    if len(board) > 5:
        raise ValueError(f"Board can have at most 5 cards, got {len(board)}")

    # Check for duplicate cards
    all_input = list(hole_cards) + list(board) + list(dead_cards)
    if len(all_input) != len(set(all_input)):
        raise ValueError("Duplicate cards detected")

    all_cards = list(hole_cards) + list(board)
    dead_set = frozenset(dead_cards)
    board_size = len(board)

    # Check for made hands
    has_flush, has_straight = _check_made_hands(all_cards)

    # Analyze flush draws (skip if already have flush)
    flush_draws: list[FlushDraw] = []
    if not has_flush:
        flush_draws = _analyze_flush_draws(all_cards, hole_cards, dead_set, board_size)

    # Analyze straight draws (skip if already have straight)
    straight_draws: list[StraightDraw] = []
    if not has_straight:
        straight_draws = _analyze_straight_draws(all_cards, dead_set, board_size)

    # Calculate total unique outs (no double counting)
    all_outs_set: set[Card] = set()
    for fd in flush_draws:
        all_outs_set.update(fd.outs)
    for sd in straight_draws:
        all_outs_set.update(sd.outs)

    all_outs = tuple(sorted(all_outs_set, key=lambda c: (-c.rank.value, c.suit.value)))

    return DrawAnalysis(
        hole_cards=tuple(hole_cards),
        board=tuple(board),
        has_flush=has_flush,
        has_straight=has_straight,
        flush_draws=tuple(flush_draws),
        straight_draws=tuple(straight_draws),
        total_outs=len(all_outs_set),
        all_outs=all_outs,
    )


def count_flush_outs(
    hole_cards: Sequence[Card],
    board: Sequence[Card],
    dead_cards: Sequence[Card] = (),
) -> int:
    """
    Quick count of flush outs.

    Returns:
        Number of flush outs, or 0 if no flush draw or already have flush.
    """
    analysis = analyze_draws(hole_cards, board, dead_cards)
    if not analysis.flush_draws:
        return 0
    return max(fd.out_count for fd in analysis.flush_draws)


def count_straight_outs(
    hole_cards: Sequence[Card],
    board: Sequence[Card],
    dead_cards: Sequence[Card] = (),
) -> int:
    """
    Quick count of straight outs.

    Returns:
        Number of straight outs, or 0 if no straight draw or already have straight.
    """
    analysis = analyze_draws(hole_cards, board, dead_cards)
    if not analysis.straight_draws:
        return 0
    # Sum outs from non-overlapping draws, avoiding duplicates
    all_outs: set[Card] = set()
    for sd in analysis.straight_draws:
        all_outs.update(sd.outs)
    return len(all_outs)


def get_primary_draw(
    hole_cards: Sequence[Card],
    board: Sequence[Card],
) -> DrawType | None:
    """
    Get the primary (strongest) draw type.

    Returns:
        The strongest DrawType, or None if no draw.
    """
    analysis = analyze_draws(hole_cards, board)

    if not analysis.has_draw:
        return None

    # Priority: Flush draw > OESD/Double Gutshot > Gutshot > Backdoor
    if any(fd.draw_type == DrawType.FLUSH_DRAW for fd in analysis.flush_draws):
        return DrawType.FLUSH_DRAW

    for draw_type in [DrawType.OPEN_ENDED, DrawType.DOUBLE_GUTSHOT, DrawType.GUTSHOT]:
        if any(sd.draw_type == draw_type for sd in analysis.straight_draws):
            return draw_type

    if any(fd.draw_type == DrawType.BACKDOOR_FLUSH for fd in analysis.flush_draws):
        return DrawType.BACKDOOR_FLUSH

    if any(sd.draw_type == DrawType.BACKDOOR_STRAIGHT for sd in analysis.straight_draws):
        return DrawType.BACKDOOR_STRAIGHT

    return None
