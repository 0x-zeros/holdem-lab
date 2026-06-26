"""Deterministic baseline poker policy.

This is intentionally conservative: it gives the game and bot a stable
``decide(state) -> Action`` entry point before CFR/RL training exists.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from holdem_common import Action, ActionType, Card, GameState, Rank

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
    continue_threshold: float = 0.46
    marginal_threshold: float = 0.36
    marginal_call_price_fraction: float = 0.20
    value_bet_pot_fraction: float = 0.65


class HeuristicPolicy:
    def __init__(self, config: HeuristicConfig | None = None) -> None:
        self.config = config or HeuristicConfig()

    def decide(self, state: GameState) -> Action:
        if state.current_seat is None:
            raise ValueError("cannot decide for a terminal state")
        if not state.legal_actions:
            raise ValueError("cannot decide without legal actions")

        legal = _legal_by_type(state.legal_actions)
        player = state.player(state.current_seat)
        strength = estimate_private_strength(player.hole_cards, state.board)

        if state.to_call > 0:
            if strength >= self.config.value_raise_threshold:
                pressure = _value_bet_or_raise(state, legal, self.config)
                if pressure is not None:
                    return pressure

            call = legal.get(ActionType.CALL)
            if call is not None and _should_call(state, call, strength, self.config):
                return Action(ActionType.CALL, amount=call.amount)

            fold = legal.get(ActionType.FOLD)
            if fold is not None:
                return fold
            if call is not None:
                return Action(ActionType.CALL, amount=call.amount)
            return state.legal_actions[0]

        if strength >= self.config.value_raise_threshold:
            pressure = _value_bet_or_raise(state, legal, self.config)
            if pressure is not None:
                return pressure

        check = legal.get(ActionType.CHECK)
        if check is not None:
            return check

        call = legal.get(ActionType.CALL)
        if call is not None:
            return Action(ActionType.CALL, amount=call.amount)

        return state.legal_actions[0]


def decide(state: GameState) -> Action:
    return HeuristicPolicy().decide(state)


def estimate_private_strength(
    hole_cards: Iterable[Card],
    board: Iterable[Card] = (),
) -> float:
    hole = tuple(hole_cards)
    community = tuple(board)
    if len(hole) < 2:
        return 0.32

    first, second = hole[:2]
    first_value = _RANK_VALUES[first.rank]
    second_value = _RANK_VALUES[second.rank]
    high = max(first_value, second_value)
    low = min(first_value, second_value)

    if first.rank == second.rank:
        score = 0.52 + (high / 14.0) * 0.38
    else:
        normalized_high_cards = ((high + low) - 4) / 24.0
        gap = high - low
        suited_bonus = 0.06 if first.suit == second.suit else 0.0
        connected_bonus = max(0.0, (4.0 - float(gap)) * 0.025)
        broadway_bonus = 0.05 * sum(1 for value in (high, low) if value >= 10)
        ace_bonus = 0.06 if high == 14 else 0.0
        score = (
            0.18
            + normalized_high_cards * 0.45
            + suited_bonus
            + connected_bonus
            + broadway_bonus
            + ace_bonus
        )

    return _clamp(max(score, _made_hand_strength(hole, community)))


def _should_call(
    state: GameState,
    call: Action,
    strength: float,
    config: HeuristicConfig,
) -> bool:
    price = call.amount or state.to_call
    pot_after_call = state.pot_total + price
    required_equity = price / pot_after_call if pot_after_call > 0 else 0.0
    if strength >= config.continue_threshold and strength >= required_equity:
        return True
    return (
        strength >= config.marginal_threshold
        and required_equity <= config.marginal_call_price_fraction
    )


def _value_bet_or_raise(
    state: GameState,
    legal: dict[ActionType, Action],
    config: HeuristicConfig,
) -> Action | None:
    action = legal.get(ActionType.BET) or legal.get(ActionType.RAISE)
    if action is None:
        return legal.get(ActionType.ALL_IN)
    if action.min_amount is None or action.max_amount is None:
        raise ValueError("bet/raise legal action is missing amount bounds")
    if state.current_seat is None:
        return None

    player = state.player(state.current_seat)
    target = player.committed + max(
        state.big_blind, int(state.pot_total * config.value_bet_pot_fraction)
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


def _made_hand_strength(hole: tuple[Card, ...], board: tuple[Card, ...]) -> float:
    if not board:
        return 0.0

    counts: dict[Rank, int] = {}
    for card in (*hole, *board):
        counts[card.rank] = counts.get(card.rank, 0) + 1

    hole_ranks = {card.rank for card in hole}
    if any(count >= 4 and rank in hole_ranks for rank, count in counts.items()):
        return 0.98
    if any(count >= 3 and rank in hole_ranks for rank, count in counts.items()):
        return 0.86

    paired_hole_ranks = {
        rank for rank, count in counts.items() if count >= 2 and rank in hole_ranks
    }
    if len(paired_hole_ranks) >= 2:
        return 0.78
    if paired_hole_ranks:
        return 0.62

    return 0.0


def _legal_by_type(actions: tuple[Action, ...]) -> dict[ActionType, Action]:
    return {action.action_type: action for action in actions}


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))
