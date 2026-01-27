"""Service layer wrapping holdem_lab library."""

import time
from typing import Sequence

from holdem_lab import (
    Card,
    Rank,
    Suit,
    parse_cards,
    format_cards,
    CanonicalHand,
    get_all_canonical_hands,
    get_all_combos,
    get_combos_excluding,
    parse_canonical_hand,
    EquityRequest as HoldemEquityRequest,
    PlayerHand,
    calculate_equity,
    analyze_draws,
    DrawType,
    evaluate_hand,
)

from app.schemas.card import (
    CardInfo,
    CanonicalHandInfo,
    ParseCardsResponse,
)
from app.schemas.equity import (
    PlayerHandInput,
    PlayerEquityResult,
    EquityResponse,
)
from app.schemas.draws import (
    FlushDrawInfo,
    StraightDrawInfo,
    DrawsResponse,
)
from app.schemas.evaluate import EvaluateResponse

# Rank order for matrix positioning (A=0, K=1, ..., 2=12)
RANK_ORDER = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]

SUIT_SYMBOLS = {
    "h": "♥",
    "d": "♦",
    "c": "♣",
    "s": "♠",
}


class HoldemService:
    """Service for holdem_lab operations."""

    @staticmethod
    def card_to_info(card: Card) -> CardInfo:
        """Convert holdem_lab Card to CardInfo."""
        rank_char = str(card.rank)
        suit_char = str(card.suit)
        return CardInfo(
            notation=f"{rank_char}{suit_char}",
            rank=rank_char,
            suit=suit_char,
            suit_symbol=SUIT_SYMBOLS.get(suit_char, suit_char),
        )

    @staticmethod
    def parse_cards_safe(input_str: str) -> ParseCardsResponse:
        """Parse cards with error handling."""
        try:
            cards = parse_cards(input_str)
            card_infos = [HoldemService.card_to_info(c) for c in cards]
            return ParseCardsResponse(
                cards=card_infos,
                formatted=format_cards(cards),
                valid=True,
                error=None,
            )
        except ValueError as e:
            return ParseCardsResponse(
                cards=[],
                formatted="",
                valid=False,
                error=str(e),
            )

    @staticmethod
    def get_all_canonical_hands() -> list[CanonicalHandInfo]:
        """Get all 169 canonical hands with matrix positions."""
        hands = get_all_canonical_hands()
        result = []

        for hand in hands:
            high_char = str(hand.high_rank)
            low_char = str(hand.low_rank)

            # Determine matrix position
            high_idx = RANK_ORDER.index(high_char)
            low_idx = RANK_ORDER.index(low_char)

            is_pair = hand.high_rank == hand.low_rank

            if is_pair:
                notation = f"{high_char}{low_char}"
                row, col = high_idx, high_idx
                num_combos = 6
            elif hand.suited:
                notation = f"{high_char}{low_char}s"
                row, col = high_idx, low_idx  # Upper right
                num_combos = 4
            else:
                notation = f"{high_char}{low_char}o"
                row, col = low_idx, high_idx  # Lower left
                num_combos = 12

            result.append(CanonicalHandInfo(
                notation=notation,
                high_rank=high_char,
                low_rank=low_char,
                suited=hand.suited,
                is_pair=is_pair,
                num_combos=num_combos,
                matrix_row=row,
                matrix_col=col,
            ))

        return result

    @staticmethod
    def calculate_equity(
        players: list[PlayerHandInput],
        board: list[str] | None,
        dead_cards: list[str] | None,
        num_simulations: int,
    ) -> EquityResponse:
        """Calculate equity for given players."""
        start_time = time.perf_counter()

        # Parse board and dead cards
        board_cards = parse_cards(" ".join(board)) if board else []
        dead = frozenset(parse_cards(" ".join(dead_cards))) if dead_cards else frozenset()

        # First pass: collect all specific cards from players
        specific_cards: set[Card] = set()
        for p in players:
            if p.cards:
                specific_cards.update(parse_cards(" ".join(p.cards)))

        # Build player hands
        player_hands: list[PlayerHand] = []
        hand_descriptions: list[str] = []
        combo_counts: list[int] = []
        player_is_random: list[bool] = []

        for p in players:
            has_cards = bool(p.cards)
            has_range = bool(p.range)
            is_random = p.random or (not has_cards and not has_range)

            if has_cards:
                # Specific hand
                cards = tuple(parse_cards(" ".join(p.cards)))
                player_hands.append(PlayerHand(cards))
                hand_descriptions.append(format_cards(cards))
                combo_counts.append(1)
                player_is_random.append(False)
            elif has_range:
                # Range - expand to combos
                all_combos: list[tuple[Card, Card]] = []
                for range_str in p.range:
                    canonical = parse_canonical_hand(range_str)
                    combos = get_combos_excluding(canonical, dead | set(board_cards) | specific_cards)
                    all_combos.extend(combos)

                if not all_combos:
                    raise ValueError(f"No valid combos for range: {p.range}")

                # Use first combo for the request, but track all
                player_hands.append(PlayerHand(all_combos[0]))
                hand_descriptions.append(", ".join(p.range))
                combo_counts.append(len(all_combos))
                player_is_random.append(False)
            elif is_random:
                # Random player - sampled each simulation
                player_hands.append(PlayerHand(is_random=True))
                hand_descriptions.append("Random")
                combo_counts.append(1326)  # C(52,2) total possible hands
                player_is_random.append(True)
            else:
                raise ValueError("Each player must have 'cards', 'range', or 'random=true'")

        # Create request and calculate
        request = HoldemEquityRequest(
            players=player_hands,
            board=board_cards,
            num_simulations=num_simulations,
        )
        result = calculate_equity(request)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Build response
        player_results = []
        for i, eq in enumerate(result.players):
            desc = hand_descriptions[i]
            if combo_counts[i] > 1 and not player_is_random[i]:
                desc = f"{desc} ({combo_counts[i]} combos)"

            player_results.append(PlayerEquityResult(
                index=i,
                hand_description=desc,
                equity=eq.equity,
                win_rate=eq.win_rate,
                tie_rate=eq.tie_rate,
                combos=combo_counts[i],
            ))

        return EquityResponse(
            players=player_results,
            total_simulations=result.total_simulations,
            elapsed_ms=elapsed_ms,
        )

    @staticmethod
    def analyze_draws(
        hole_cards: list[str],
        board: list[str],
        dead_cards: list[str] | None,
    ) -> DrawsResponse:
        """Analyze draws for given hole cards and board."""
        hole = parse_cards(" ".join(hole_cards))
        board_parsed = parse_cards(" ".join(board))
        dead = parse_cards(" ".join(dead_cards)) if dead_cards else []

        analysis = analyze_draws(list(hole), list(board_parsed), dead_cards=dead)

        # Convert flush draws
        flush_infos = []
        for fd in analysis.flush_draws:
            suit_char = str(fd.suit)
            flush_infos.append(FlushDrawInfo(
                suit=suit_char,
                suit_symbol=SUIT_SYMBOLS.get(suit_char, suit_char),
                cards_held=fd.cards_held,
                outs=[HoldemService.card_to_info(c).notation for c in fd.outs],
                out_count=fd.out_count,
                is_nut=fd.is_nut,
                draw_type="backdoor_flush" if fd.draw_type == DrawType.BACKDOOR_FLUSH else "flush_draw",
            ))

        # Convert straight draws
        straight_infos = []
        for sd in analysis.straight_draws:
            draw_type_map = {
                DrawType.OPEN_ENDED: "open_ended",
                DrawType.GUTSHOT: "gutshot",
                DrawType.DOUBLE_GUTSHOT: "double_gutshot",
                DrawType.BACKDOOR_STRAIGHT: "backdoor_straight",
            }
            straight_infos.append(StraightDrawInfo(
                draw_type=draw_type_map.get(sd.draw_type, "gutshot"),
                needed_ranks=list(sd.needed_ranks),
                outs=[HoldemService.card_to_info(c).notation for c in sd.outs],
                out_count=len(sd.outs),
                high_card=sd.high_card,
                is_nut=sd.is_nut,
            ))

        return DrawsResponse(
            has_flush=analysis.has_flush,
            has_straight=analysis.has_straight,
            flush_draws=flush_infos,
            straight_draws=straight_infos,
            total_outs=analysis.total_outs,
            all_outs=[HoldemService.card_to_info(c).notation for c in analysis.all_outs],
            is_combo_draw=analysis.is_combo_draw,
        )

    @staticmethod
    def evaluate_hand_cards(cards: list[str]) -> EvaluateResponse:
        """Evaluate 5-7 cards and return the best hand."""
        parsed = parse_cards(" ".join(cards))
        result = evaluate_hand(parsed)
        return EvaluateResponse(
            hand_type=str(result.hand_type),
            description=result.describe(),
            primary_ranks=list(result.primary_ranks),
            kickers=list(result.kickers),
        )
