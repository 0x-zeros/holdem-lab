"""RLCard adapter for holdem-lab's canonical GameState.

RLCard is used as an AI/training surface here. PokerKit remains the rules
authority; this module only translates observations and action ids.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from holdem_common import Action, ActionType, Card, GameState
from numpy.typing import NDArray
from rlcard.games.nolimitholdem.round import Action as RLCardAction  # type: ignore[import-untyped]

_CARD_INDEX = {
    f"{suit}{rank}": suit_index * 13 + rank_index
    for suit_index, suit in enumerate(("S", "H", "D", "C"))
    for rank_index, rank in enumerate(
        ("A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K")
    )
}


@dataclass(frozen=True, slots=True)
class RLCardObservation:
    obs: NDArray[np.float64]
    legal_actions: OrderedDict[int, None]
    raw_obs: Mapping[str, object]
    raw_legal_actions: tuple[RLCardAction, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "obs": self.obs,
            "legal_actions": self.legal_actions,
            "raw_obs": self.raw_obs,
            "raw_legal_actions": list(self.raw_legal_actions),
        }


def to_rlcard_observation(state: GameState, seat: int | None = None) -> RLCardObservation:
    acting_seat = state.current_seat if seat is None else seat
    if acting_seat is None:
        raise ValueError("seat is required for terminal states")

    player = state.player(acting_seat)
    obs = np.zeros(54, dtype=np.float64)
    for card in state.board + player.hole_cards:
        obs[_rlcard_card_index(card)] = 1.0
    obs[52] = float(player.committed)
    obs[53] = float(max((p.committed for p in state.players), default=0))

    legal_ids = legal_action_ids(state) if acting_seat == state.current_seat else OrderedDict()
    raw_legal_actions = tuple(RLCardAction(action_id) for action_id in legal_ids)
    raw_obs = {
        "hand": [_rlcard_card_code(card) for card in player.hole_cards],
        "public_cards": [_rlcard_card_code(card) for card in state.board],
        "all_chips": [p.committed for p in state.players],
        "my_chips": player.committed,
        "legal_actions": list(raw_legal_actions),
        "current_player": state.current_seat,
        "pot": state.pot_total,
        "street": state.street.value,
    }

    return RLCardObservation(
        obs=obs,
        legal_actions=legal_ids,
        raw_obs=raw_obs,
        raw_legal_actions=raw_legal_actions,
    )


def legal_action_ids(state: GameState) -> OrderedDict[int, None]:
    actions: OrderedDict[int, None] = OrderedDict()
    action_types = {action.action_type for action in state.legal_actions}

    if ActionType.FOLD in action_types:
        actions[RLCardAction.FOLD.value] = None
    if {ActionType.CHECK, ActionType.CALL} & action_types:
        actions[RLCardAction.CHECK_CALL.value] = None
    bounds = _raise_bounds(state)
    if bounds is not None:
        half_pot = _raise_to_amount(state, fraction=0.5)
        pot = _raise_to_amount(state, fraction=1.0)
        max_amount = bounds[1]
        if half_pot < max_amount:
            actions[RLCardAction.RAISE_HALF_POT.value] = None
        if pot < max_amount:
            actions[RLCardAction.RAISE_POT.value] = None
    if ActionType.ALL_IN in action_types:
        actions[RLCardAction.ALL_IN.value] = None

    return actions


def rlcard_action_to_action(action_id: int | RLCardAction, state: GameState) -> Action:
    rlcard_action = action_id if isinstance(action_id, RLCardAction) else RLCardAction(action_id)

    match rlcard_action:
        case RLCardAction.FOLD:
            return Action(ActionType.FOLD)
        case RLCardAction.CHECK_CALL:
            if state.to_call == 0:
                return Action(ActionType.CHECK)
            return Action(ActionType.CALL, amount=state.to_call)
        case RLCardAction.RAISE_HALF_POT:
            return _raise_action(state, _raise_to_amount(state, fraction=0.5))
        case RLCardAction.RAISE_POT:
            return _raise_action(state, _raise_to_amount(state, fraction=1.0))
        case RLCardAction.ALL_IN:
            bounds = _raise_bounds(state)
            if bounds is None:
                raise ValueError("all-in is not legal in the current state")
            return Action(
                ActionType.ALL_IN, amount=bounds[1], min_amount=bounds[1], max_amount=bounds[1]
            )

    raise ValueError(f"unsupported RLCard action: {rlcard_action}")


def _raise_action(state: GameState, amount: int) -> Action:
    bounds = _raise_bounds(state)
    if bounds is None:
        raise ValueError("raising is not legal in the current state")

    min_amount, max_amount, action_type = bounds
    if amount >= max_amount:
        return Action(
            ActionType.ALL_IN, amount=max_amount, min_amount=max_amount, max_amount=max_amount
        )
    return Action(action_type, amount=amount, min_amount=min_amount, max_amount=max_amount)


def _raise_bounds(state: GameState) -> tuple[int, int, ActionType] | None:
    for action in state.legal_actions:
        if action.action_type in (ActionType.BET, ActionType.RAISE):
            if action.min_amount is None or action.max_amount is None:
                raise ValueError("bet/raise legal action is missing amount bounds")
            return (action.min_amount, action.max_amount, action.action_type)
    return None


def _raise_to_amount(state: GameState, *, fraction: float) -> int:
    bounds = _raise_bounds(state)
    if bounds is None or state.current_seat is None:
        raise ValueError("raising is not legal in the current state")

    min_amount, max_amount, _ = bounds
    committed = state.player(state.current_seat).committed
    increment = max(1, int(state.pot_total * fraction))
    return min(max_amount, max(min_amount, committed + increment))


def _rlcard_card_index(card: Card) -> int:
    return _CARD_INDEX[_rlcard_card_code(card)]


def _rlcard_card_code(card: Card) -> str:
    return f"{card.suit.value.upper()}{card.rank.value}"
