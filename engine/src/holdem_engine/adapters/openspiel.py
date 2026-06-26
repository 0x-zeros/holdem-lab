"""OpenSpiel adapter for holdem-lab's canonical GameState.

PokerKit remains the rules authority. This module exposes an OpenSpiel-shaped
policy surface: current player ids, legal action ids, tensors, and action
decoding. The action ids are holdem-lab's discrete abstraction, not raw
``universal_poker`` move ids.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum

import pyspiel  # type: ignore[import-not-found]
from holdem_common import Action, ActionType, Card, GameState, Street

_CARD_INDEX = {
    f"{suit}{rank}": suit_index * 13 + rank_index
    for suit_index, suit in enumerate(("S", "H", "D", "C"))
    for rank_index, rank in enumerate(
        ("A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K")
    )
}

_STREET_INDEX = {
    Street.PREFLOP: 0,
    Street.FLOP: 1,
    Street.TURN: 2,
    Street.RIVER: 3,
    Street.SHOWDOWN: 4,
}

_PLAYER_FEATURES = 7


class OpenSpielAction(IntEnum):
    FOLD = 0
    CHECK_CALL = 1
    MIN_RAISE = 2
    HALF_POT = 3
    POT = 4
    ALL_IN = 5


@dataclass(frozen=True, slots=True)
class OpenSpielObservation:
    current_player: int
    legal_actions: tuple[int, ...]
    observation_tensor: tuple[float, ...]
    information_state_tensor: tuple[float, ...]
    rewards: tuple[int, ...]
    raw_state: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "current_player": self.current_player,
            "legal_actions": list(self.legal_actions),
            "observation_tensor": list(self.observation_tensor),
            "information_state_tensor": list(self.information_state_tensor),
            "rewards": list(self.rewards),
            "raw_state": self.raw_state,
        }


TERMINAL_PLAYER_ID = int(pyspiel.PlayerId.TERMINAL)


def registered_poker_games() -> tuple[str, ...]:
    return tuple(
        name
        for name in (str(registered_name) for registered_name in pyspiel.registered_names())
        if "poker" in name.lower() or "holdem" in name.lower()
    )


def to_openspiel_observation(state: GameState, seat: int | None = None) -> OpenSpielObservation:
    if seat is not None:
        state.player(seat)

    viewer_seat = state.current_seat if seat is None else seat
    current_player = state.current_seat if state.current_seat is not None else TERMINAL_PLAYER_ID
    legal_actions = (
        openspiel_legal_action_ids(state)
        if state.current_seat is not None and viewer_seat == state.current_seat
        else ()
    )

    return OpenSpielObservation(
        current_player=current_player,
        legal_actions=legal_actions,
        observation_tensor=_state_tensor(state, viewer_seat=None),
        information_state_tensor=_state_tensor(state, viewer_seat=viewer_seat),
        rewards=_rewards(state),
        raw_state={
            "hand_id": state.hand_id,
            "street": state.street.value,
            "current_player": current_player,
            "viewer_seat": viewer_seat,
            "pot": state.pot_total,
            "to_call": state.to_call,
        },
    )


def openspiel_legal_action_ids(state: GameState) -> tuple[int, ...]:
    actions: list[int] = []
    action_types = {action.action_type for action in state.legal_actions}

    if ActionType.FOLD in action_types:
        actions.append(OpenSpielAction.FOLD)
    if {ActionType.CHECK, ActionType.CALL} & action_types:
        actions.append(OpenSpielAction.CHECK_CALL)
    if _raise_bounds(state) is not None:
        actions.extend(
            (
                OpenSpielAction.MIN_RAISE,
                OpenSpielAction.HALF_POT,
                OpenSpielAction.POT,
            ),
        )
    if ActionType.ALL_IN in action_types:
        actions.append(OpenSpielAction.ALL_IN)

    return tuple(int(action) for action in actions)


def openspiel_action_to_action(action_id: int | OpenSpielAction, state: GameState) -> Action:
    openspiel_action = (
        action_id if isinstance(action_id, OpenSpielAction) else OpenSpielAction(action_id)
    )

    match openspiel_action:
        case OpenSpielAction.FOLD:
            return Action(ActionType.FOLD)
        case OpenSpielAction.CHECK_CALL:
            if state.to_call == 0:
                return Action(ActionType.CHECK)
            return Action(ActionType.CALL, amount=state.to_call)
        case OpenSpielAction.MIN_RAISE:
            bounds = _raise_bounds(state)
            if bounds is None:
                raise ValueError("raising is not legal in the current state")
            return _raise_action(state, bounds[0])
        case OpenSpielAction.HALF_POT:
            return _raise_action(state, _raise_to_amount(state, fraction=0.5))
        case OpenSpielAction.POT:
            return _raise_action(state, _raise_to_amount(state, fraction=1.0))
        case OpenSpielAction.ALL_IN:
            bounds = _raise_bounds(state)
            if bounds is None:
                raise ValueError("all-in is not legal in the current state")
            return Action(
                ActionType.ALL_IN, amount=bounds[1], min_amount=bounds[1], max_amount=bounds[1]
            )

    raise ValueError(f"unsupported OpenSpiel action: {openspiel_action}")


def _state_tensor(state: GameState, *, viewer_seat: int | None) -> tuple[float, ...]:
    public_cards = [0.0] * 52
    private_cards = [0.0] * 52
    for card in state.board:
        public_cards[_card_index(card)] = 1.0
    if viewer_seat is not None:
        for card in state.player(viewer_seat).hole_cards:
            private_cards[_card_index(card)] = 1.0

    big_blind = float(max(1, state.big_blind))
    table_features = [
        float(_STREET_INDEX[state.street]),
        float(state.current_seat if state.current_seat is not None else TERMINAL_PLAYER_ID),
        state.pot_total / big_blind,
        state.to_call / big_blind,
        state.min_raise / big_blind,
    ]

    player_features: list[float] = []
    for player in state.players:
        player_features.extend(
            (
                player.stack / big_blind,
                player.committed / big_blind,
                float(player.active),
                float(player.all_in),
                float(player.dealer),
                float(player.small_blind),
                float(player.big_blind),
            ),
        )

    legal_mask = [0.0] * len(OpenSpielAction)
    for action_id in openspiel_legal_action_ids(state):
        legal_mask[action_id] = 1.0

    return tuple(public_cards + private_cards + table_features + player_features + legal_mask)


def _rewards(state: GameState) -> tuple[int, ...]:
    if state.current_seat is not None:
        return tuple(0 for _ in state.players)

    raw_payoffs = state.metadata.get("payoffs")
    if isinstance(raw_payoffs, tuple) and all(isinstance(payoff, int) for payoff in raw_payoffs):
        return raw_payoffs
    return tuple(0 for _ in state.players)


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


def _card_index(card: Card) -> int:
    return _CARD_INDEX[f"{card.suit.value.upper()}{card.rank.value}"]
