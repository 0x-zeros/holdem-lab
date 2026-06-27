"""Deterministic reference opponents for AI evaluation.

These are intentionally simple, well-understood policies that act as an
*absolute* yardstick for the heuristic baseline. Self-play among tweaked
``HeuristicPolicy`` profiles only measures relative strength inside one family;
pitting a policy against a calling station, a rock, a maniac and a uniform
random bot tells us whether it actually exploits known-weak play.

Every reference implements the same duck-typed :class:`Policy` interface used by
``HeuristicPolicy`` (``explain(state) -> PolicyDecision`` and
``decide(state) -> Action``), so the game, evaluation harness and bot can swap
them in without special-casing.
"""

from __future__ import annotations

import hashlib
import random
from typing import Protocol, runtime_checkable

from holdem_common import Action, ActionType, GameState

from holdem_ai.heuristic import PolicyDecision

__all__ = [
    "AggressivePolicy",
    "CallStationPolicy",
    "Policy",
    "RandomPolicy",
    "RockPolicy",
]


@runtime_checkable
class Policy(Protocol):
    """Shared decision interface for every local policy."""

    def explain(self, state: GameState) -> PolicyDecision: ...

    def decide(self, state: GameState) -> Action: ...


class RandomPolicy:
    """Uniformly random over legal action types, with a random legal size.

    Deterministic for a given seed and state, so evaluation stays reproducible.
    """

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    def decide(self, state: GameState) -> Action:
        return self.explain(state).action

    def explain(self, state: GameState) -> PolicyDecision:
        _require_decidable(state)
        rng = random.Random(self._seed ^ _state_fingerprint(state))
        choice = rng.choice(state.legal_actions)
        if (
            choice.action_type in (ActionType.BET, ActionType.RAISE)
            and choice.min_amount is not None
            and choice.max_amount is not None
        ):
            amount = rng.randint(choice.min_amount, choice.max_amount)
            choice = Action(
                choice.action_type,
                amount=amount,
                min_amount=choice.min_amount,
                max_amount=choice.max_amount,
            )
        return _reference_decision(choice, "random", state)


class CallStationPolicy:
    """Never folds, never raises: checks when free, calls when faced with a bet."""

    def decide(self, state: GameState) -> Action:
        return self.explain(state).action

    def explain(self, state: GameState) -> PolicyDecision:
        _require_decidable(state)
        legal = _by_type(state.legal_actions)
        check = legal.get(ActionType.CHECK)
        if check is not None:
            return _reference_decision(check, "call_station_check", state)
        call = legal.get(ActionType.CALL)
        if call is not None:
            return _reference_decision(
                Action(ActionType.CALL, amount=call.amount), "call_station", state
            )
        return _reference_decision(state.legal_actions[0], "call_station_fallback", state)


class RockPolicy:
    """Never invests beyond posted blinds: checks when free, folds to any bet."""

    def decide(self, state: GameState) -> Action:
        return self.explain(state).action

    def explain(self, state: GameState) -> PolicyDecision:
        _require_decidable(state)
        legal = _by_type(state.legal_actions)
        check = legal.get(ActionType.CHECK)
        if check is not None:
            return _reference_decision(check, "rock_check", state)
        fold = legal.get(ActionType.FOLD)
        if fold is not None:
            return _reference_decision(fold, "rock_fold", state)
        call = legal.get(ActionType.CALL)
        if call is not None:
            return _reference_decision(
                Action(ActionType.CALL, amount=call.amount), "rock_call_forced", state
            )
        return _reference_decision(state.legal_actions[0], "rock_fallback", state)


class AggressivePolicy:
    """Hyper-aggressive maniac: bets/raises ~pot, never folds, calls when capped.

    Sizes to roughly the pot (capped at the legal max) instead of always jamming,
    so multi-street action is preserved as a stress test for calldown discipline.
    """

    def __init__(self, pot_fraction: float = 1.0) -> None:
        self._pot_fraction = pot_fraction

    def decide(self, state: GameState) -> Action:
        return self.explain(state).action

    def explain(self, state: GameState) -> PolicyDecision:
        _require_decidable(state)
        legal = _by_type(state.legal_actions)
        aggressive = legal.get(ActionType.RAISE) or legal.get(ActionType.BET)
        if aggressive is not None and aggressive.min_amount is not None:
            amount = _pot_sized_amount(state, aggressive, self._pot_fraction)
            return _reference_decision(
                Action(
                    aggressive.action_type,
                    amount=amount,
                    min_amount=aggressive.min_amount,
                    max_amount=aggressive.max_amount,
                ),
                "maniac_bet",
                state,
            )
        call = legal.get(ActionType.CALL)
        if call is not None:
            return _reference_decision(
                Action(ActionType.CALL, amount=call.amount), "maniac_call", state
            )
        check = legal.get(ActionType.CHECK)
        if check is not None:
            return _reference_decision(check, "maniac_check", state)
        return _reference_decision(state.legal_actions[0], "maniac_fallback", state)


def _pot_sized_amount(state: GameState, action: Action, pot_fraction: float) -> int:
    assert action.min_amount is not None and action.max_amount is not None
    if state.current_seat is None:
        return action.min_amount
    committed = state.player(state.current_seat).committed
    target = committed + max(state.big_blind, int(state.pot_total * pot_fraction))
    return min(action.max_amount, max(action.min_amount, target))


def _reference_decision(action: Action, reason: str, state: GameState) -> PolicyDecision:
    return PolicyDecision(
        action=action,
        reason=reason,
        strength=0.0,
        required_equity=None,
        metadata={"pot_total": state.pot_total, "to_call": state.to_call},
    )


def _require_decidable(state: GameState) -> None:
    if state.current_seat is None:
        raise ValueError("cannot decide for a terminal state")
    if not state.legal_actions:
        raise ValueError("cannot decide without legal actions")


def _by_type(actions: tuple[Action, ...]) -> dict[ActionType, Action]:
    return {action.action_type: action for action in actions}


def _state_fingerprint(state: GameState) -> int:
    assert state.current_seat is not None
    hole = state.player(state.current_seat).hole_cards
    payload = "|".join(
        (
            state.hand_id,
            str(state.current_seat),
            str(state.street),
            str(state.to_call),
            str(state.pot_total),
            ",".join(card.code for card in state.board),
            ",".join(card.code for card in hole),
        )
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
