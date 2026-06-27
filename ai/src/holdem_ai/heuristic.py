"""Deterministic baseline poker policy.

This is intentionally conservative: it gives the game and bot a stable
``decide(state) -> Action`` entry point before CFR/RL training exists.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from holdem_common import Action, ActionType, Card, GameState, Rank, Street, Suit

from holdem_ai.equity import estimate_showdown_equity

_RANK_VALUES = {
    Rank.TWO: 2,
    Rank.THREE: 3,
    Rank.FOUR: 4,
    Rank.FIVE: 5,
    Rank.SIX: 6,
    Rank.SEVEN: 7,
    Rank.EIGHT: 8,
    Rank.NINE: 9,
    Rank.TEN: 10,
    Rank.JACK: 11,
    Rank.QUEEN: 12,
    Rank.KING: 13,
    Rank.ACE: 14,
}


@dataclass(frozen=True, slots=True)
class HeuristicConfig:
    value_raise_threshold: float = 0.78
    protection_bet_threshold: float = 0.64
    semi_bluff_threshold: float = 0.58
    continue_threshold: float = 0.46
    marginal_threshold: float = 0.36
    marginal_call_price_fraction: float = 0.20
    value_bet_pot_fraction: float = 0.65
    strong_value_bet_pot_fraction: float = 0.85
    protection_bet_pot_fraction: float = 0.50
    semi_bluff_pot_fraction: float = 0.45
    draw_call_discount_per_out: float = 0.012
    equity_samples: int = 160
    equity_weight: float = 0.30


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    action: Action
    reason: str
    strength: float
    required_equity: float | None
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _HandAssessment:
    strength: float
    made_hand: str
    draw: str | None
    outs: int
    preflop_strength: float
    active_opponents: int
    showdown_equity: float | None

    @property
    def has_strong_draw(self) -> bool:
        return self.draw in {"flush_draw", "open_ended_straight_draw", "combo_draw"}


class HeuristicPolicy:
    def __init__(self, config: HeuristicConfig | None = None) -> None:
        self.config = config or HeuristicConfig()

    def decide(self, state: GameState) -> Action:
        return self.explain(state).action

    def explain(self, state: GameState) -> PolicyDecision:
        if state.current_seat is None:
            raise ValueError("cannot decide for a terminal state")
        if not state.legal_actions:
            raise ValueError("cannot decide without legal actions")

        legal = _legal_by_type(state.legal_actions)
        player = state.player(state.current_seat)
        assessment = _assess_hand(state, player.hole_cards, state.board, self.config)
        strength = assessment.strength

        if state.to_call > 0:
            if strength >= self.config.value_raise_threshold:
                pressure = _value_bet_or_raise(
                    state,
                    legal,
                    self.config,
                    strength=strength,
                    semi_bluff=False,
                    pot_fraction_override=None,
                )
                if pressure is not None:
                    return _decision(
                        pressure,
                        "value_raise",
                        state,
                        assessment,
                        legal.get(ActionType.CALL),
                    )

            call = legal.get(ActionType.CALL)
            if call is not None and _should_call(state, call, assessment, self.config):
                return _decision(
                    Action(ActionType.CALL, amount=call.amount),
                    "call_price",
                    state,
                    assessment,
                    call,
                )

            fold = legal.get(ActionType.FOLD)
            if fold is not None:
                return _decision(fold, "fold_price", state, assessment, call)
            if call is not None:
                return _decision(
                    Action(ActionType.CALL, amount=call.amount),
                    "call_forced_no_fold",
                    state,
                    assessment,
                    call,
                )
            return _decision(state.legal_actions[0], "fallback", state, assessment, None)

        if strength >= self.config.value_raise_threshold:
            pressure = _value_bet_or_raise(
                state,
                legal,
                self.config,
                strength=strength,
                semi_bluff=False,
                pot_fraction_override=None,
            )
            if pressure is not None:
                return _decision(pressure, "value_bet", state, assessment, None)

        if _should_protection_bet(state, assessment, self.config):
            pressure = _value_bet_or_raise(
                state,
                legal,
                self.config,
                strength=strength,
                semi_bluff=False,
                pot_fraction_override=self.config.protection_bet_pot_fraction,
            )
            if pressure is not None:
                return _decision(pressure, "protection_bet", state, assessment, None)

        if assessment.has_strong_draw and strength >= self.config.semi_bluff_threshold:
            pressure = _value_bet_or_raise(
                state,
                legal,
                self.config,
                strength=strength,
                semi_bluff=True,
                pot_fraction_override=None,
            )
            if pressure is not None:
                return _decision(pressure, "semi_bluff", state, assessment, None)

        check = legal.get(ActionType.CHECK)
        if check is not None:
            return _decision(check, "check", state, assessment, None)

        call = legal.get(ActionType.CALL)
        if call is not None:
            return _decision(
                Action(ActionType.CALL, amount=call.amount),
                "call_no_price",
                state,
                assessment,
                call,
            )

        return _decision(state.legal_actions[0], "fallback", state, assessment, None)


def decide(state: GameState) -> Action:
    return HeuristicPolicy().decide(state)


def explain_decision(state: GameState) -> PolicyDecision:
    return HeuristicPolicy().explain(state)


def estimate_private_strength(
    hole_cards: Iterable[Card],
    board: Iterable[Card] = (),
) -> float:
    return _assess_cards(tuple(hole_cards), tuple(board), active_opponents=1).strength


def _assess_hand(
    state: GameState,
    hole_cards: Iterable[Card],
    board: Iterable[Card],
    config: HeuristicConfig,
) -> _HandAssessment:
    active_opponents = max(
        0,
        sum(1 for player in state.active_players if player.seat != state.current_seat),
    )
    raw = _assess_cards(tuple(hole_cards), tuple(board), active_opponents=active_opponents)
    showdown_equity = _estimate_equity_or_none(state, config)
    position_bonus = _position_bonus(state)
    street_penalty = _multiway_penalty(state.street, active_opponents)
    strength = raw.strength
    if showdown_equity is not None:
        strength = (
            raw.strength * (1.0 - config.equity_weight) + showdown_equity * config.equity_weight
        )
    strength = _clamp(strength + position_bonus - street_penalty)
    return _HandAssessment(
        strength=strength,
        made_hand=raw.made_hand,
        draw=raw.draw,
        outs=raw.outs,
        preflop_strength=raw.preflop_strength,
        active_opponents=active_opponents,
        showdown_equity=showdown_equity,
    )


def _assess_cards(
    hole: tuple[Card, ...],
    community: tuple[Card, ...],
    *,
    active_opponents: int,
) -> _HandAssessment:
    if len(hole) < 2:
        return _HandAssessment(
            strength=0.32,
            made_hand="unknown",
            draw=None,
            outs=0,
            preflop_strength=0.32,
            active_opponents=active_opponents,
            showdown_equity=None,
        )

    preflop_strength = _preflop_strength(hole)
    made_score, made_hand = _made_hand_score(hole, community)
    draw, outs, draw_bonus = _draw_profile(hole, community)
    if community:
        unpaired_ceiling = 0.50 if made_score == 0.0 else 0.58
        base = min(preflop_strength * 0.72, unpaired_ceiling)
        score = max(made_score, base + draw_bonus)
    else:
        score = preflop_strength

    return _HandAssessment(
        strength=_clamp(score),
        made_hand=made_hand,
        draw=draw,
        outs=outs,
        preflop_strength=preflop_strength,
        active_opponents=active_opponents,
        showdown_equity=None,
    )


def _should_call(
    state: GameState,
    call: Action,
    assessment: _HandAssessment,
    config: HeuristicConfig,
) -> bool:
    required_equity = _required_equity(state, call)
    draw_discount = min(0.14, assessment.outs * config.draw_call_discount_per_out)
    adjusted_required_equity = max(0.0, required_equity - draw_discount)
    if (
        assessment.strength >= config.continue_threshold
        and assessment.strength >= adjusted_required_equity
    ):
        return True
    if assessment.has_strong_draw and required_equity <= 0.34:
        return True
    return (
        assessment.strength >= config.marginal_threshold
        and required_equity <= config.marginal_call_price_fraction
    )


def _value_bet_or_raise(
    state: GameState,
    legal: dict[ActionType, Action],
    config: HeuristicConfig,
    *,
    strength: float,
    semi_bluff: bool,
    pot_fraction_override: float | None,
) -> Action | None:
    action = legal.get(ActionType.BET) or legal.get(ActionType.RAISE)
    if action is None:
        all_in = legal.get(ActionType.ALL_IN)
        if all_in is not None and (strength >= 0.88 or semi_bluff):
            return all_in
        return None
    if action.min_amount is None or action.max_amount is None:
        raise ValueError("bet/raise legal action is missing amount bounds")
    if state.current_seat is None:
        return None

    player = state.player(state.current_seat)
    if pot_fraction_override is not None:
        pot_fraction = pot_fraction_override
    elif semi_bluff:
        pot_fraction = config.semi_bluff_pot_fraction
    elif strength >= 0.90:
        pot_fraction = config.strong_value_bet_pot_fraction
    else:
        pot_fraction = config.value_bet_pot_fraction

    target = player.committed + max(
        state.big_blind,
        int(state.pot_total * pot_fraction),
    )
    amount = min(action.max_amount, max(action.min_amount, target))
    if amount >= action.max_amount and action.max_amount > action.min_amount:
        amount = action.min_amount

    return Action(
        action.action_type,
        amount=amount,
        min_amount=action.min_amount,
        max_amount=action.max_amount,
    )


def _should_protection_bet(
    state: GameState,
    assessment: _HandAssessment,
    config: HeuristicConfig,
) -> bool:
    if state.street is Street.PREFLOP or state.street is Street.RIVER:
        return False
    if assessment.strength < config.protection_bet_threshold:
        return False
    return assessment.made_hand in {"pair", "two_pair", "trips"}


def _decision(
    action: Action,
    reason: str,
    state: GameState,
    assessment: _HandAssessment,
    call: Action | None,
) -> PolicyDecision:
    return PolicyDecision(
        action=action,
        reason=reason,
        strength=assessment.strength,
        required_equity=_required_equity(state, call) if call is not None else None,
        metadata={
            "made_hand": assessment.made_hand,
            "draw": assessment.draw,
            "outs": assessment.outs,
            "preflop_strength": assessment.preflop_strength,
            "active_opponents": assessment.active_opponents,
            "showdown_equity": assessment.showdown_equity,
            "pot_total": state.pot_total,
            "to_call": state.to_call,
        },
    )


def _estimate_equity_or_none(
    state: GameState,
    config: HeuristicConfig,
) -> float | None:
    if not state.board or config.equity_samples <= 0:
        return None
    try:
        return estimate_showdown_equity(state, samples=config.equity_samples)
    except ValueError:
        return None


def _preflop_strength(hole: tuple[Card, ...]) -> float:
    first, second = hole[:2]
    first_value = _RANK_VALUES[first.rank]
    second_value = _RANK_VALUES[second.rank]
    high = max(first_value, second_value)
    low = min(first_value, second_value)

    if first.rank == second.rank:
        return _clamp(0.52 + (high / 14.0) * 0.38)

    normalized_high_cards = ((high + low) - 4) / 24.0
    gap = high - low
    suited_bonus = 0.06 if first.suit == second.suit else 0.0
    connected_bonus = max(0.0, (4.0 - float(gap)) * 0.025)
    broadway_bonus = 0.05 * sum(1 for value in (high, low) if value >= 10)
    ace_bonus = 0.06 if high == 14 else 0.0
    return _clamp(
        0.18
        + normalized_high_cards * 0.45
        + suited_bonus
        + connected_bonus
        + broadway_bonus
        + ace_bonus
    )


def _made_hand_score(hole: tuple[Card, ...], board: tuple[Card, ...]) -> tuple[float, str]:
    if not board:
        pair = "pocket_pair" if hole[0].rank == hole[1].rank else "high_card"
        return 0.0, pair

    all_cards = (*hole, *board)
    rank_counts = Counter(card.rank for card in all_cards)
    hole_ranks = {card.rank for card in hole}

    if _has_hole_straight_flush(hole, all_cards):
        return 0.99, "straight_flush"
    if any(count >= 4 and rank in hole_ranks for rank, count in rank_counts.items()):
        return 0.97, "quads"

    trips = {rank for rank, count in rank_counts.items() if count >= 3}
    pairs = {rank for rank, count in rank_counts.items() if count >= 2}
    if trips and len(pairs) >= 2 and (trips | pairs) & hole_ranks:
        return 0.94, "full_house"
    if _has_hole_flush(hole, all_cards):
        return 0.90, "flush"
    if _has_hole_straight(hole, all_cards):
        return 0.88, "straight"
    if any(rank in hole_ranks for rank in trips):
        return 0.82, "trips"

    paired_hole_ranks = {
        rank for rank, count in rank_counts.items() if count >= 2 and rank in hole_ranks
    }
    if len(paired_hole_ranks) >= 2:
        return 0.76, "two_pair"
    if paired_hole_ranks:
        return _pair_score(hole, board, paired_hole_ranks), "pair"

    return 0.0, "high_card"


def _pair_score(
    hole: tuple[Card, ...],
    board: tuple[Card, ...],
    paired_hole_ranks: set[Rank],
) -> float:
    board_values = [_RANK_VALUES[card.rank] for card in board]
    highest_board = max(board_values) if board_values else 0
    best_pair = max(_RANK_VALUES[rank] for rank in paired_hole_ranks)
    if hole[0].rank == hole[1].rank and best_pair > highest_board:
        return 0.72
    if best_pair >= highest_board:
        return 0.66
    return 0.58


def _has_hole_flush(hole: tuple[Card, ...], cards: tuple[Card, ...]) -> bool:
    suit_counts = Counter(card.suit for card in cards)
    hole_suits = {card.suit for card in hole}
    return any(count >= 5 and suit in hole_suits for suit, count in suit_counts.items())


def _has_hole_straight_flush(hole: tuple[Card, ...], cards: tuple[Card, ...]) -> bool:
    for suit in Suit:
        suited = tuple(card for card in cards if card.suit is suit)
        if len(suited) < 5:
            continue
        suited_hole = tuple(card for card in hole if card.suit is suit)
        if suited_hole and _has_hole_straight(suited_hole, suited):
            return True
    return False


def _has_hole_straight(hole: tuple[Card, ...], cards: tuple[Card, ...]) -> bool:
    card_values = _straight_values(cards)
    hole_values = _straight_values(hole)
    return any(hole_values & run for run in _straight_runs(card_values))


def _draw_profile(hole: tuple[Card, ...], board: tuple[Card, ...]) -> tuple[str | None, int, float]:
    if not board or len(board) >= 5:
        return None, 0, 0.0

    flush_outs = _flush_draw_outs(hole, (*hole, *board))
    straight_outs, straight_draw = _straight_draw_outs(hole, (*hole, *board))
    outs = flush_outs + straight_outs
    if flush_outs and straight_outs:
        return "combo_draw", min(15, outs), 0.17
    if flush_outs:
        return "flush_draw", flush_outs, 0.11
    if straight_draw == "open_ended_straight_draw":
        return straight_draw, straight_outs, 0.09
    if straight_draw == "gutshot_straight_draw":
        return straight_draw, straight_outs, 0.04
    return None, 0, 0.0


def _flush_draw_outs(hole: tuple[Card, ...], cards: tuple[Card, ...]) -> int:
    suit_counts: Counter[Suit] = Counter(card.suit for card in cards)
    hole_suits = {card.suit for card in hole}
    if any(count == 4 and suit in hole_suits for suit, count in suit_counts.items()):
        return 9
    return 0


def _straight_draw_outs(hole: tuple[Card, ...], cards: tuple[Card, ...]) -> tuple[int, str | None]:
    values = _straight_values(cards)
    hole_values = _straight_values(hole)
    found_gutshot = False
    for start in range(1, 11):
        run = frozenset(range(start, start + 5))
        present = values & run
        if len(present) != 4 or not hole_values & present:
            continue
        missing = next(value for value in run if value not in present)
        if missing in {start, start + 4} and start > 1:
            return 8, "open_ended_straight_draw"
        found_gutshot = True
    if found_gutshot:
        return 4, "gutshot_straight_draw"
    return 0, None


def _straight_runs(values: set[int]) -> tuple[frozenset[int], ...]:
    return tuple(
        frozenset(range(start, start + 5))
        for start in range(1, 11)
        if set(range(start, start + 5)).issubset(values)
    )


def _straight_values(cards: Iterable[Card]) -> set[int]:
    values = {_RANK_VALUES[card.rank] for card in cards}
    if 14 in values:
        values.add(1)
    return values


def _required_equity(state: GameState, call: Action) -> float:
    price = call.amount or state.to_call
    pot_after_call = state.pot_total + price
    return price / pot_after_call if pot_after_call > 0 else 0.0


def _position_bonus(state: GameState) -> float:
    if state.street is not Street.PREFLOP or state.current_seat is None:
        return 0.0
    if state.current_seat == state.button_seat:
        return 0.025
    return 0.0


def _multiway_penalty(street: Street, active_opponents: int) -> float:
    if street is Street.PREFLOP:
        return min(0.05, max(0, active_opponents - 1) * 0.015)
    return min(0.08, max(0, active_opponents - 1) * 0.02)


def _legal_by_type(actions: tuple[Action, ...]) -> dict[ActionType, Action]:
    return {action.action_type: action for action in actions}


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))
