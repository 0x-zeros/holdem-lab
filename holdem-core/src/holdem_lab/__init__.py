"""holdem-lab: Texas Hold'em poker equity calculator and game engine."""

from holdem_lab.cards import (
    Card,
    Deck,
    Rank,
    Suit,
    format_card,
    format_cards,
    parse_card,
    parse_cards,
)
from holdem_lab.equity import (
    ConvergencePoint,
    EquityRequest,
    EquityResult,
    PlayerEquity,
    PlayerHand,
    calculate_equity,
)
from holdem_lab.evaluator import (
    HandRank,
    HandType,
    evaluate_five,
    evaluate_hand,
    find_winners,
)
from holdem_lab.event_log import (
    Event,
    EventLog,
    EventType,
    HandReplayer,
)
from holdem_lab.game_state import (
    GameState,
    PotResult,
    Street,
)
from holdem_lab.canonize import (
    CanonicalHand,
    are_strategically_equivalent,
    canonize_hole_cards,
    get_all_canonical_hands,
    get_all_combos,
    get_combos_excluding,
    parse_canonical_hand,
)
from holdem_lab.draws import (
    DrawAnalysis,
    DrawType,
    FlushDraw,
    StraightDraw,
    analyze_draws,
    count_flush_outs,
    count_straight_outs,
    get_primary_draw,
)

__all__ = [
    # cards
    "Card",
    "Deck",
    "Rank",
    "Suit",
    "format_card",
    "format_cards",
    "parse_card",
    "parse_cards",
    # evaluator
    "HandRank",
    "HandType",
    "evaluate_five",
    "evaluate_hand",
    "find_winners",
    # equity
    "ConvergencePoint",
    "EquityRequest",
    "EquityResult",
    "PlayerEquity",
    "PlayerHand",
    "calculate_equity",
    # event_log
    "Event",
    "EventLog",
    "EventType",
    "HandReplayer",
    # game_state
    "GameState",
    "PotResult",
    "Street",
    # canonize
    "CanonicalHand",
    "are_strategically_equivalent",
    "canonize_hole_cards",
    "get_all_canonical_hands",
    "get_all_combos",
    "get_combos_excluding",
    "parse_canonical_hand",
    # draws
    "DrawAnalysis",
    "DrawType",
    "FlushDraw",
    "StraightDraw",
    "analyze_draws",
    "count_flush_outs",
    "count_straight_outs",
    "get_primary_draw",
]

__version__ = "0.1.0"
