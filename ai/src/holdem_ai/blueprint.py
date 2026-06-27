"""Bridge a solved CFR push/fold blueprint into the shared ``decide()`` API.

This is the S2b payoff: a CFR-solved strategy that the local game and the bot can
use through the same ``explain(state) -> PolicyDecision`` interface as the
heuristic. The push/fold blueprint only models a **heads-up** short-stack preflop
all-in subgame (button open-jam, big-blind call-vs-jam), so :class:`PushFoldPolicy`
consults it only in genuinely heads-up spots and delegates everything else (more
than two players, deep stacks, postflop, non-jam raises) to a fallback policy —
by default the heuristic.

The bridge is reliable because the abstract game's information state is ours:
``GameState`` hole cards map to the same equity buckets the game was solved over.

By default actions are **sampled from the CFR average policy** with a seed derived
from the information state (reproducible, and preserves the mixed-strategy /
low-exploitability property). ``mode="pure"`` instead takes the >=0.5 argmax — an
explicitly *exploitative deterministic projection*, not the equilibrium policy.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping

from holdem_common import Action, ActionType, GameState, Street

from holdem_ai.baselines import Policy
from holdem_ai.heuristic import HeuristicPolicy, PolicyDecision
from holdem_ai.preflop import preflop_bucket
from holdem_ai.preflop_game import PushFoldBlueprint, solve_push_fold

__all__ = ["PushFoldPolicy"]

_MODES = ("mixed", "pure")


class PushFoldPolicy:
    """Play heads-up short-stack preflop via a CFR push/fold blueprint, else fall back."""

    def __init__(
        self,
        *,
        max_jam_bb: int = 12,
        iterations: int = 400,
        mode: str = "mixed",
        fallback: Policy | None = None,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self._max_jam_bb = max_jam_bb
        self._iterations = iterations
        self._mode = mode
        self._fallback: Policy = fallback or HeuristicPolicy()
        self._cache: dict[int, PushFoldBlueprint] = {}

    def decide(self, state: GameState) -> Action:
        return self.explain(state).action

    def explain(self, state: GameState) -> PolicyDecision:
        override = self._maybe_push_fold(state)
        if override is not None:
            return override
        return self._fallback.explain(state)

    def _blueprint(self, stack_key: int) -> PushFoldBlueprint:
        if stack_key not in self._cache:
            self._cache[stack_key] = solve_push_fold(
                stack=float(stack_key), iterations=self._iterations
            )
        return self._cache[stack_key]

    def _maybe_push_fold(self, state: GameState) -> PolicyDecision | None:
        if state.street is not Street.PREFLOP or state.current_seat is None:
            return None
        # Hard guard: the blueprint is a heads-up SB-vs-BB model. Never apply it
        # to 3+ handed (e.g. 6-max) spots, where the button is not the small blind.
        if len(state.active_players) != 2:
            return None
        effective_bb = _effective_bb(state)
        if effective_bb is None or effective_bb > self._max_jam_bb:
            return None
        hole = state.player(state.current_seat).hole_cards
        if len(hole) < 2:
            return None

        legal = {action.action_type: action for action in state.legal_actions}
        bucket = preflop_bucket(hole)
        stack_key = max(2, min(self._max_jam_bb, round(effective_bb)))
        blueprint = self._blueprint(stack_key)
        context = _Context(bucket, effective_bb, stack_key)

        if _is_button_open(state) and ActionType.ALL_IN in legal:
            jam_prob = blueprint.sb_jam[bucket]
            if self._take_action(state, bucket, jam_prob):
                return _decision(
                    legal[ActionType.ALL_IN], "pushfold_jam", jam_prob, context, self._mode
                )
            if ActionType.FOLD in legal:
                return _decision(
                    legal[ActionType.FOLD], "pushfold_fold", jam_prob, context, self._mode
                )

        if _is_facing_jam(state) and ActionType.CALL in legal:
            call_prob = blueprint.bb_call[bucket]
            if self._take_action(state, bucket, call_prob):
                return _decision(
                    legal[ActionType.CALL], "pushfold_call", call_prob, context, self._mode
                )
            if ActionType.FOLD in legal:
                return _decision(
                    legal[ActionType.FOLD], "pushfold_overfold", call_prob, context, self._mode
                )

        return None

    def _take_action(self, state: GameState, bucket: int, probability: float) -> bool:
        if self._mode == "pure":
            return probability >= 0.5
        return random.Random(_decision_seed(state, bucket)).random() < probability


class _Context:
    __slots__ = ("bucket", "effective_bb", "stack_key")

    def __init__(self, bucket: int, effective_bb: float, stack_key: int) -> None:
        self.bucket = bucket
        self.effective_bb = effective_bb
        self.stack_key = stack_key


def _effective_bb(state: GameState) -> float | None:
    active = state.active_players
    if not active or state.big_blind <= 0:
        return None
    effective_chips = min(player.stack + player.committed for player in active)
    return effective_chips / state.big_blind


def _is_button_open(state: GameState) -> bool:
    # Heads-up: the button posts the small blind and acts first preflop, unraised.
    if state.current_seat != state.button_seat:
        return False
    committed = [player.committed for player in state.active_players]
    return max(committed, default=0) <= state.big_blind and state.to_call <= state.big_blind


def _is_facing_jam(state: GameState) -> bool:
    if state.current_seat is None:
        return False
    me = state.player(state.current_seat)
    villains = [player for player in state.active_players if player.seat != me.seat]
    # A single all-in opponent for more than the blind: facing a jam (even when we
    # cover it). The blueprint already uses the effective (shorter) stack.
    return len(villains) == 1 and villains[0].all_in and state.to_call > state.big_blind


def _decision_seed(state: GameState, bucket: int) -> int:
    payload = "|".join(
        (
            state.hand_id,
            str(state.current_seat),
            str(state.street),
            str(bucket),
            str(state.to_call),
        )
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _decision(
    action: Action, reason: str, probability: float, context: _Context, mode: str
) -> PolicyDecision:
    metadata: Mapping[str, object] = {
        "source": "cfr_push_fold",
        "mode": mode,
        "preflop_bucket": context.bucket,
        "effective_bb": round(context.effective_bb, 2),
        "blueprint_stack_key": context.stack_key,
        "blueprint_action_prob": round(probability, 4),
    }
    return PolicyDecision(
        action=action,
        reason=reason,
        strength=probability,
        required_equity=None,
        metadata=metadata,
    )
