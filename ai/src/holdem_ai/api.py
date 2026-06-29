"""Public decision API shared by the game and bots."""

from __future__ import annotations

from holdem_common import Action, GameState

from holdem_ai.field import FieldExploitPolicy
from holdem_ai.heuristic import PolicyDecision

_POLICY_BY_SEAT: dict[int, FieldExploitPolicy] = {}
_TERMINAL_POLICY = FieldExploitPolicy()


def decide(state: GameState) -> Action:
    return _policy_for(state).decide(state)


def explain_decision(state: GameState) -> PolicyDecision:
    return _policy_for(state).explain(state)


def reset_decision_policy(seat: int | None = None) -> None:
    """Clear accumulated opponent reads for the public decision API.

    ``decide`` is stateful now: each controlled seat gets its own field-exploit
    model, so bot sessions learn across hands without local multi-AI seats
    polluting each other's reads. Tests and callers that start a fresh session can
    reset all seats, or just one controlled seat.
    """
    if seat is None:
        for policy in _POLICY_BY_SEAT.values():
            policy.reset()
        _POLICY_BY_SEAT.clear()
        _TERMINAL_POLICY.reset()
        return
    _POLICY_BY_SEAT.pop(seat, None)


def _policy_for(state: GameState) -> FieldExploitPolicy:
    if state.current_seat is None:
        return _TERMINAL_POLICY
    policy = _POLICY_BY_SEAT.get(state.current_seat)
    if policy is None:
        policy = FieldExploitPolicy()
        _POLICY_BY_SEAT[state.current_seat] = policy
    return policy
