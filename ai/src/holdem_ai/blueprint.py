"""Bridge a solved CFR push/fold blueprint into the shared ``decide()`` API.

This is the S2b payoff: a CFR-solved strategy that the local game and the bot can
use through the same ``explain(state) -> PolicyDecision`` interface as the
heuristic. The push/fold blueprint only covers short-stack preflop all-in spots
(button open-jam, big-blind call-vs-jam), so :class:`PushFoldPolicy` consults it
there and delegates every other decision (deep stacks, postflop, non-jam raises)
to a fallback policy — by default the heuristic.

The bridge is reliable because the abstract game's information state is ours:
``GameState`` hole cards map to the same equity buckets the game was solved over.
"""

from __future__ import annotations

from collections.abc import Mapping

from holdem_common import Action, ActionType, GameState, Street

from holdem_ai.baselines import Policy
from holdem_ai.heuristic import HeuristicPolicy, PolicyDecision
from holdem_ai.preflop import preflop_bucket
from holdem_ai.preflop_game import PushFoldBlueprint, solve_push_fold

__all__ = ["PushFoldPolicy"]

_JAM_THRESHOLD = 0.5


class PushFoldPolicy:
    """Play short-stack preflop via a CFR push/fold blueprint, else fall back."""

    def __init__(
        self,
        *,
        max_jam_bb: int = 12,
        iterations: int = 400,
        fallback: Policy | None = None,
    ) -> None:
        self._max_jam_bb = max_jam_bb
        self._iterations = iterations
        self._fallback: Policy = fallback or HeuristicPolicy()
        self._cache: dict[int, PushFoldBlueprint] = {}

    def decide(self, state: GameState) -> Action:
        return self.explain(state).action

    def explain(self, state: GameState) -> PolicyDecision:
        override = self._maybe_push_fold(state)
        if override is not None:
            return override
        return self._fallback.explain(state)

    def _blueprint(self, stack_bb: int) -> PushFoldBlueprint:
        key = max(2, min(self._max_jam_bb, stack_bb))
        if key not in self._cache:
            self._cache[key] = solve_push_fold(stack=float(key), iterations=self._iterations)
        return self._cache[key]

    def _maybe_push_fold(self, state: GameState) -> PolicyDecision | None:
        if state.street is not Street.PREFLOP or state.current_seat is None:
            return None
        effective_bb = _effective_bb(state)
        if effective_bb is None or effective_bb > self._max_jam_bb:
            return None
        hole = state.player(state.current_seat).hole_cards
        if len(hole) < 2:
            return None

        legal = {action.action_type: action for action in state.legal_actions}
        bucket = preflop_bucket(hole)
        blueprint = self._blueprint(round(effective_bb))

        if _is_button_open(state) and ActionType.ALL_IN in legal:
            if blueprint.sb_jam[bucket] >= _JAM_THRESHOLD:
                return _decision(legal[ActionType.ALL_IN], "pushfold_jam", bucket, effective_bb)
            if ActionType.FOLD in legal:
                return _decision(legal[ActionType.FOLD], "pushfold_fold", bucket, effective_bb)

        if _is_facing_jam(state) and ActionType.CALL in legal:
            if blueprint.bb_call[bucket] >= _JAM_THRESHOLD:
                return _decision(legal[ActionType.CALL], "pushfold_call", bucket, effective_bb)
            if ActionType.FOLD in legal:
                return _decision(legal[ActionType.FOLD], "pushfold_overfold", bucket, effective_bb)

        return None


def _effective_bb(state: GameState) -> float | None:
    active = state.active_players
    if not active or state.big_blind <= 0:
        return None
    effective_chips = min(player.stack + player.committed for player in active)
    return effective_chips / state.big_blind


def _is_button_open(state: GameState) -> bool:
    # Heads-up: the button posts the small blind and acts first preflop.
    if state.current_seat != state.button_seat:
        return False
    committed = [player.committed for player in state.active_players]
    return max(committed, default=0) <= state.big_blind and state.to_call <= state.big_blind


def _is_facing_jam(state: GameState) -> bool:
    if state.current_seat is None:
        return False
    me = state.player(state.current_seat)
    # Calling commits our whole remaining stack to a bet larger than the blind.
    return state.to_call > state.big_blind and state.to_call >= me.stack


def _decision(action: Action, reason: str, bucket: int, effective_bb: float) -> PolicyDecision:
    metadata: Mapping[str, object] = {
        "preflop_bucket": bucket,
        "effective_bb": round(effective_bb, 2),
        "source": "cfr_push_fold",
    }
    return PolicyDecision(
        action=action,
        reason=reason,
        strength=0.0,
        required_equity=None,
        metadata=metadata,
    )
