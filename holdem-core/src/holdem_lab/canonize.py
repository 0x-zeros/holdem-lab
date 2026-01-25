"""Hand canonization for strategic equivalence analysis."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

from holdem_lab.cards import Card, Rank, Suit


@dataclass(frozen=True, slots=True)
class CanonicalHand:
    """
    Canonical representation of a starting hand.

    Represents the strategic equivalence class of hole cards.
    e.g., AhKh, AsKs, AdKd, AcKc all map to CanonicalHand(ACE, KING, suited=True)

    There are exactly 169 canonical starting hands:
    - 13 pairs (AA, KK, ..., 22)
    - 78 suited hands (AKs, AQs, ..., 32s)
    - 78 offsuit hands (AKo, AQo, ..., 32o)
    """

    high_rank: Rank
    low_rank: Rank
    suited: bool

    def __post_init__(self) -> None:
        """Validate that high_rank >= low_rank."""
        if self.high_rank.value < self.low_rank.value:
            raise ValueError(
                f"high_rank must be >= low_rank, got {self.high_rank} < {self.low_rank}"
            )
        if self.high_rank == self.low_rank and self.suited:
            raise ValueError("Pairs cannot be suited")

    def __str__(self) -> str:
        """Return standard notation like 'AKs', 'AKo', 'AA'."""
        if self.is_pair:
            return f"{self.high_rank}{self.low_rank}"
        suffix = "s" if self.suited else "o"
        return f"{self.high_rank}{self.low_rank}{suffix}"

    def __repr__(self) -> str:
        return f"CanonicalHand({self.high_rank!r}, {self.low_rank!r}, suited={self.suited})"

    @property
    def is_pair(self) -> bool:
        """True if this is a pocket pair."""
        return self.high_rank == self.low_rank

    @property
    def num_combos(self) -> int:
        """
        Number of distinct card combinations for this hand.

        - Pairs: 6 combos (C(4,2) = 6)
        - Suited: 4 combos (one per suit)
        - Offsuit: 12 combos (4 * 3)
        """
        if self.is_pair:
            return 6
        elif self.suited:
            return 4
        else:
            return 12

    @property
    def gap(self) -> int:
        """
        Gap between ranks (0 for pairs, 1 for connectors like AK).

        Useful for categorizing hands (connectors, one-gappers, etc.)
        """
        return self.high_rank.value - self.low_rank.value


def canonize_hole_cards(cards: Sequence[Card]) -> CanonicalHand:
    """
    Canonize a 2-card hole card hand.

    Args:
        cards: Exactly 2 hole cards.

    Returns:
        CanonicalHand representing the equivalence class.

    Raises:
        ValueError: If not exactly 2 cards.

    Example:
        >>> from holdem_lab import parse_cards
        >>> canonize_hole_cards(parse_cards("Ah Kh"))
        CanonicalHand(Rank.ACE, Rank.KING, suited=True)
        >>> str(canonize_hole_cards(parse_cards("7c 2d")))
        '72o'
    """
    if len(cards) != 2:
        raise ValueError(f"Hole cards must be exactly 2 cards, got {len(cards)}")

    c1, c2 = cards

    # Determine high/low rank (higher rank first)
    if c1.rank.value >= c2.rank.value:
        high_rank, low_rank = c1.rank, c2.rank
    else:
        high_rank, low_rank = c2.rank, c1.rank

    # Determine if suited (pairs are never suited)
    suited = c1.suit == c2.suit and high_rank != low_rank

    return CanonicalHand(high_rank, low_rank, suited)


def parse_canonical_hand(s: str) -> CanonicalHand:
    """
    Parse canonical hand notation.

    Accepts formats like:
    - "AKs" - Ace-King suited
    - "AKo" - Ace-King offsuit
    - "AA" - Pocket aces
    - "72o" - Seven-two offsuit

    Args:
        s: Hand notation string.

    Returns:
        CanonicalHand.

    Raises:
        ValueError: If invalid format.

    Example:
        >>> parse_canonical_hand("AKs")
        CanonicalHand(Rank.ACE, Rank.KING, suited=True)
    """
    s = s.strip().upper()
    if len(s) < 2 or len(s) > 3:
        raise ValueError(f"Invalid canonical hand notation: {s!r}")

    # Parse the two rank characters
    rank1 = Rank.from_char(s[0])
    rank2 = Rank.from_char(s[1])

    # Normalize to high/low (standard notation allows either order)
    if rank1.value >= rank2.value:
        high_rank, low_rank = rank1, rank2
    else:
        high_rank, low_rank = rank2, rank1

    # Parse suited/offsuit suffix
    if len(s) == 2:
        # No suffix - must be a pair
        if high_rank != low_rank:
            raise ValueError(
                f"Non-pair hands must have 's' or 'o' suffix, got {s!r}"
            )
        suited = False
    else:
        suffix = s[2].upper()
        if suffix == "S":
            if high_rank == low_rank:
                raise ValueError(f"Pairs cannot be suited: {s!r}")
            suited = True
        elif suffix == "O":
            suited = False
        else:
            raise ValueError(f"Invalid suffix '{suffix}' in hand notation: {s!r}")

    return CanonicalHand(high_rank, low_rank, suited)


def get_all_combos(canonical: CanonicalHand) -> list[tuple[Card, Card]]:
    """
    Get all specific card combinations for a canonical hand.

    Args:
        canonical: The canonical hand.

    Returns:
        List of (card1, card2) tuples, where card1.rank >= card2.rank.
        - Pairs: 6 combos
        - Suited: 4 combos
        - Offsuit: 12 combos

    Example:
        >>> hand = parse_canonical_hand("AKs")
        >>> combos = get_all_combos(hand)
        >>> len(combos)
        4
    """
    combos: list[tuple[Card, Card]] = []

    if canonical.is_pair:
        # All suit combinations for the same rank: C(4,2) = 6
        for s1, s2 in combinations(Suit, 2):
            combos.append((Card(canonical.high_rank, s1), Card(canonical.low_rank, s2)))
    elif canonical.suited:
        # Same suit for both cards: 4 combos
        for suit in Suit:
            combos.append((Card(canonical.high_rank, suit), Card(canonical.low_rank, suit)))
    else:
        # Different suits: 4 * 3 = 12 combos
        for s1 in Suit:
            for s2 in Suit:
                if s1 != s2:
                    combos.append(
                        (Card(canonical.high_rank, s1), Card(canonical.low_rank, s2))
                    )

    return combos


def get_all_canonical_hands() -> list[CanonicalHand]:
    """
    Get all 169 canonical starting hands.

    Returns hands in standard order:
    - Pairs first: AA, KK, QQ, ..., 22
    - Then suited: AKs, AQs, ..., 32s
    - Then offsuit: AKo, AQo, ..., 32o

    Example:
        >>> hands = get_all_canonical_hands()
        >>> len(hands)
        169
        >>> str(hands[0])
        'AA'
    """
    hands: list[CanonicalHand] = []

    # Pairs (13 hands)
    for rank in reversed(Rank):
        hands.append(CanonicalHand(rank, rank, suited=False))

    # Suited hands (78 hands)
    for high in reversed(Rank):
        for low in reversed(Rank):
            if high.value > low.value:
                hands.append(CanonicalHand(high, low, suited=True))

    # Offsuit hands (78 hands)
    for high in reversed(Rank):
        for low in reversed(Rank):
            if high.value > low.value:
                hands.append(CanonicalHand(high, low, suited=False))

    return hands


def get_combos_excluding(
    canonical: CanonicalHand,
    dead_cards: Sequence[Card],
) -> list[tuple[Card, Card]]:
    """
    Get combos for a canonical hand excluding dead cards.

    Args:
        canonical: The canonical hand.
        dead_cards: Cards that are not available (e.g., on board or folded).

    Returns:
        List of available (card1, card2) tuples.

    Example:
        >>> hand = parse_canonical_hand("AA")
        >>> dead = parse_cards("Ah Kc")
        >>> combos = get_combos_excluding(hand, dead)
        >>> len(combos)  # 6 - 3 = 3 (combos using Ah are removed)
        3
    """
    dead_set = set(dead_cards)
    all_combos = get_all_combos(canonical)
    return [(c1, c2) for c1, c2 in all_combos if c1 not in dead_set and c2 not in dead_set]


def are_strategically_equivalent(
    hand1: Sequence[Card],
    hand2: Sequence[Card],
) -> bool:
    """
    Check if two hole card hands are strategically equivalent.

    Two hands are equivalent if they have the same canonical form
    (same ranks and suited/offsuit status).

    Args:
        hand1: First hand (2 cards).
        hand2: Second hand (2 cards).

    Returns:
        True if hands are strategically equivalent.

    Example:
        >>> h1 = parse_cards("Ah Kh")
        >>> h2 = parse_cards("As Ks")
        >>> are_strategically_equivalent(h1, h2)
        True
    """
    return canonize_hole_cards(hand1) == canonize_hole_cards(hand2)
