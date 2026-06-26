"""Public decision API shared by the game and bots."""

from __future__ import annotations

from holdem_common import Action, GameState

from holdem_ai.heuristic import HeuristicPolicy

_DEFAULT_POLICY = HeuristicPolicy()


def decide(state: GameState) -> Action:
    return _DEFAULT_POLICY.decide(state)
