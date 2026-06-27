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

from holdem_common import Action, ActionType, Card, GameState, Street

from holdem_ai.equity import estimate_showdown_equity
from holdem_ai.heuristic import PolicyDecision
from holdem_ai.preflop import preflop_equity

__all__ = [
    "AggressivePolicy",
    "CallStationPolicy",
    "Policy",
    "RandomPolicy",
    "RockPolicy",
    "TAGPolicy",
    "ThreeBetJammerPolicy",
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


# --- Equity-grounded competent references -------------------------------------
#
# Unlike the toy references above (each exploitable in one obvious way), these
# ground every decision in real equity: preflop via ``preflop_equity`` (all-in
# equity vs a random hand) and postflop via ``estimate_showdown_equity`` (Monte
# Carlo vs a random range). They fold air, value-bet strong hands and call down
# with the odds, so beating them is evidence of genuine edge rather than punishing
# a free-money pattern. Both are deterministic given the state.


class TAGPolicy:
    """A tight-aggressive reg: tight preflop, value-driven postflop, folds air.

    Opens / 3bets a tight range, defends with equity and pot odds, value-bets when
    likely ahead and gives up when likely behind — punishing both over-folding (it
    steals relentlessly) and over-calling / over-bluffing (it value-bets and calls
    down with a real edge).
    """

    def __init__(self, *, samples: int = 120) -> None:
        self._samples = samples

    def decide(self, state: GameState) -> Action:
        return self.explain(state).action

    def explain(self, state: GameState) -> PolicyDecision:
        _require_decidable(state)
        legal = _by_type(state.legal_actions)
        if state.street is Street.PREFLOP:
            return self._preflop(state, legal)
        return _equity_postflop(
            state, legal, value_min=0.62, call_margin=0.04, samples=self._samples, prefix="tag"
        )

    def _preflop(self, state: GameState, legal: dict[ActionType, Action]) -> PolicyDecision:
        equity = preflop_equity(_hole(state))
        raise_action = legal.get(ActionType.RAISE)
        if _is_open_spot(state):
            if equity >= 0.57 and raise_action is not None:
                amount = _open_raise_amount(state, raise_action, 2.5)
                return _grounded(_resize(raise_action, amount), "tag_open", state, equity)
            check = legal.get(ActionType.CHECK)
            if check is not None:
                return _grounded(check, "tag_check", state, equity)
            fold = legal.get(ActionType.FOLD)
            if fold is not None:
                return _grounded(fold, "tag_open_fold", state, equity)
        else:
            if equity >= 0.66 and raise_action is not None:
                amount = _pot_sized_amount(state, raise_action, 1.0)
                return _grounded(_resize(raise_action, amount), "tag_3bet", state, equity)
            call = legal.get(ActionType.CALL)
            if call is not None and equity >= 0.55 and _pot_odds(state) <= 0.5:
                return _grounded(call, "tag_call", state, equity)
            fold = legal.get(ActionType.FOLD)
            if fold is not None:
                return _grounded(fold, "tag_fold", state, equity)
            check = legal.get(ActionType.CHECK)
            if check is not None:
                return _grounded(check, "tag_check", state, equity)
        return _grounded(state.legal_actions[0], "tag_fallback", state, equity)


class ThreeBetJammerPolicy:
    """A short-stack 3bet-jammer: polarised all-ins over opens, no flatting.

    At or below ``short_stack_bb`` effective it jams premiums over a raise (and
    open-jams the very top), folding everything else — directly punishing players
    who open too wide and cannot call a jam. Deeper, it 3bets premiums and folds
    the rest. Postflop (rarely reached) it plays fit-or-fold on equity.
    """

    def __init__(
        self,
        *,
        short_stack_bb: float = 25.0,
        jam_min: float = 0.60,
        open_min: float = 0.55,
        samples: int = 120,
    ) -> None:
        self._short_stack_bb = short_stack_bb
        self._jam_min = jam_min
        self._open_min = open_min
        self._samples = samples

    def decide(self, state: GameState) -> Action:
        return self.explain(state).action

    def explain(self, state: GameState) -> PolicyDecision:
        _require_decidable(state)
        legal = _by_type(state.legal_actions)
        if state.street is Street.PREFLOP:
            return self._preflop(state, legal)
        return _equity_postflop(
            state, legal, value_min=0.60, call_margin=0.08, samples=self._samples, prefix="jammer"
        )

    def _preflop(self, state: GameState, legal: dict[ActionType, Action]) -> PolicyDecision:
        equity = preflop_equity(_hole(state))
        short = _effective_bb(state) <= self._short_stack_bb
        all_in = legal.get(ActionType.ALL_IN)
        raise_action = legal.get(ActionType.RAISE)
        if _facing_raise(state):
            if short and equity >= self._jam_min and all_in is not None:
                return _grounded(all_in, "jammer_3bet_jam", state, equity)
            if not short and equity >= 0.66 and raise_action is not None:
                amount = _pot_sized_amount(state, raise_action, 1.0)
                return _grounded(_resize(raise_action, amount), "jammer_3bet", state, equity)
            fold = legal.get(ActionType.FOLD)
            if fold is not None:
                return _grounded(fold, "jammer_fold", state, equity)
        elif _is_open_spot(state):
            if short and equity >= 0.67 and all_in is not None:
                return _grounded(all_in, "jammer_open_jam", state, equity)
            if equity >= self._open_min and raise_action is not None:
                amount = _open_raise_amount(state, raise_action, 2.5)
                return _grounded(_resize(raise_action, amount), "jammer_open", state, equity)
            check = legal.get(ActionType.CHECK)
            if check is not None:
                return _grounded(check, "jammer_check", state, equity)
            fold = legal.get(ActionType.FOLD)
            if fold is not None:
                return _grounded(fold, "jammer_open_fold", state, equity)
        check = legal.get(ActionType.CHECK)
        if check is not None:
            return _grounded(check, "jammer_check", state, equity)
        fold = legal.get(ActionType.FOLD)
        if fold is not None:
            return _grounded(fold, "jammer_fold", state, equity)
        return _grounded(state.legal_actions[0], "jammer_fallback", state, equity)


def _equity_postflop(
    state: GameState,
    legal: dict[ActionType, Action],
    *,
    value_min: float,
    call_margin: float,
    samples: int,
    prefix: str,
) -> PolicyDecision:
    """Shared postflop core: value-bet/raise ahead, call with odds, else fold/check."""
    equity = estimate_showdown_equity(state, samples=samples)
    if state.to_call > 0:
        aggressive = legal.get(ActionType.RAISE) or legal.get(ActionType.BET)
        if equity >= 0.78 and aggressive is not None and aggressive.min_amount is not None:
            amount = _pot_sized_amount(state, aggressive, 0.75)
            return _grounded(_resize(aggressive, amount), f"{prefix}_value_raise", state, equity)
        call = legal.get(ActionType.CALL)
        if call is not None and equity >= _pot_odds(state) + call_margin:
            return _grounded(call, f"{prefix}_call", state, equity)
        fold = legal.get(ActionType.FOLD)
        if fold is not None:
            return _grounded(fold, f"{prefix}_fold", state, equity)
        if call is not None:
            return _grounded(call, f"{prefix}_call_forced", state, equity)
    else:
        aggressive = legal.get(ActionType.BET) or legal.get(ActionType.RAISE)
        if equity >= value_min and aggressive is not None and aggressive.min_amount is not None:
            amount = _pot_sized_amount(state, aggressive, 0.66)
            return _grounded(_resize(aggressive, amount), f"{prefix}_value_bet", state, equity)
        check = legal.get(ActionType.CHECK)
        if check is not None:
            return _grounded(check, f"{prefix}_check", state, equity)
    return _grounded(state.legal_actions[0], f"{prefix}_fallback", state, equity)


def _hole(state: GameState) -> tuple[Card, ...]:
    assert state.current_seat is not None
    return state.player(state.current_seat).hole_cards


def _effective_bb(state: GameState) -> float:
    active = state.active_players
    if not active or state.big_blind <= 0:
        return 0.0
    return min(player.stack + player.committed for player in active) / state.big_blind


def _pot_odds(state: GameState) -> float:
    if state.to_call <= 0:
        return 0.0
    return state.to_call / (state.pot_total + state.to_call)


def _is_open_spot(state: GameState) -> bool:
    committed = [player.committed for player in state.active_players]
    return max(committed, default=0) <= state.big_blind and state.to_call <= state.big_blind


def _facing_raise(state: GameState) -> bool:
    return state.to_call > 0 and any(
        player.committed > state.big_blind for player in state.active_players
    )


def _open_raise_amount(state: GameState, action: Action, to_bb: float) -> int:
    assert action.min_amount is not None and action.max_amount is not None
    target = int(round(to_bb * state.big_blind))
    return min(action.max_amount, max(action.min_amount, target))


def _resize(action: Action, amount: int) -> Action:
    return Action(
        action.action_type,
        amount=amount,
        min_amount=action.min_amount,
        max_amount=action.max_amount,
    )


def _grounded(action: Action, reason: str, state: GameState, equity: float) -> PolicyDecision:
    return PolicyDecision(
        action=action,
        reason=reason,
        strength=float(equity),
        required_equity=None,
        metadata={
            "pot_total": state.pot_total,
            "to_call": state.to_call,
            "equity": round(float(equity), 3),
        },
    )


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
