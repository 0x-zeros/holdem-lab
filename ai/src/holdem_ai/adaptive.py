"""Opponent-adaptive policy: exploit relentless aggression, else play safe.

The S2c reference grid surfaced a *fork*: blind short-stack open-jamming (the full
push/fold blueprint) crushes a maniac at 5-10bb (roughly +40..+67 bb/100) by
denying it the postflop streets where its relentless pot-betting prints, yet it
bleeds against a competent tight-aggressive reg that defends correctly
(roughly -12..-26). No *static* gate wins both — the right line depends on who is
sitting across the table.

:class:`AdaptivePolicy` resolves the fork at the table. It watches how often the
opponent forces it to answer a bet or raise and, once it has gathered enough
evidence that the opponent is a relentless aggressor, switches from the safe
default (the heuristic plus the unexploitable call-vs-jam floor — the ``hybrid``
guardrail) to the exploitative open-jam blueprint (``pushfold``). Against anyone
who is *not* over-aggressive it never leaves the safe default, so it keeps the
heuristic's value and never inherits the open-jam's cost versus competent play.
This is the S4-lite opponent-adaptive layer; the accumulated read is the only new
state, and it is reset between opponents via :meth:`reset`.

The read is reconstructed purely from the states the policy is asked to act on:
``GameState`` carries no action log and ``last_aggressor`` is unpopulated, so the
only channel is the decision snapshots themselves. A decision counts as "facing
aggression" when the focal owes chips (``to_call > 0``) to a single villain who
put in more than the blind. Because ``committed`` is cumulative for the hand,
preflop this means ``villain.committed > big_blind`` (a real raise, not a posted
blind or a limp); postflop both players enter each street matched, so any
``to_call > 0`` is by construction a villain bet. The *frequency* of facing
aggression over a match cleanly fingerprints a maniac (high) against a tag, rock
or calling station (low) — see ``docs/ai-strength.md`` for the measured grid.
"""

from __future__ import annotations

from dataclasses import replace

from holdem_common import Action, GameState, Street

from holdem_ai.baselines import Policy
from holdem_ai.blueprint import PushFoldPolicy
from holdem_ai.heuristic import PolicyDecision

__all__ = ["AdaptivePolicy"]


class AdaptivePolicy:
    """Route between an exploitative and a safe sub-policy by reading aggression.

    ``exploit`` is played only once the opponent is classified a relentless
    aggressor (facing-aggression frequency ``>= maniac_threshold`` after at least
    ``min_observations`` decisions); otherwise — including the whole warm-up before
    enough evidence accrues — ``default`` plays. Both sub-policies default to the
    validated push/fold profiles so the adaptive layer is a *pure router* over
    them: ``pushfold`` (open-jam) when exploiting, ``hybrid`` (heuristic + safe
    call floor) otherwise.
    """

    def __init__(
        self,
        *,
        exploit: Policy | None = None,
        default: Policy | None = None,
        maniac_threshold: float = 0.55,
        min_observations: int = 20,
    ) -> None:
        if not 0.0 < maniac_threshold <= 1.0:
            raise ValueError("maniac_threshold must be in (0, 1]")
        if min_observations < 1:
            raise ValueError("min_observations must be positive")
        # Default to the exact validated profiles: full open-jam to punish a
        # maniac, the defend-only guardrail (heuristic + call floor) otherwise.
        self._exploit: Policy = (
            exploit if exploit is not None else PushFoldPolicy(defend_only=False)
        )
        self._default: Policy = default if default is not None else PushFoldPolicy(defend_only=True)
        self._maniac_threshold = maniac_threshold
        self._min_observations = min_observations
        self._observations = 0
        self._aggressive = 0

    def reset(self) -> None:
        """Forget the accumulated read (call when the opponent changes)."""
        self._observations = 0
        self._aggressive = 0

    @property
    def aggression(self) -> float | None:
        """Facing-aggression frequency, or ``None`` before the warm-up completes."""
        if self._observations < self._min_observations:
            return None
        return self._aggressive / self._observations

    def classified_maniac(self) -> bool:
        rate = self.aggression
        return rate is not None and rate >= self._maniac_threshold

    def decide(self, state: GameState) -> Action:
        return self.explain(state).action

    def explain(self, state: GameState) -> PolicyDecision:
        self._observe(state)
        maniac = self.classified_maniac()
        delegate = self._exploit if maniac else self._default
        decision = delegate.explain(state)
        return _annotate(
            decision,
            maniac=maniac,
            rate=self.aggression,
            observations=self._observations,
            threshold=self._maniac_threshold,
        )

    def _observe(self, state: GameState) -> None:
        if state.current_seat is None:
            return
        self._observations += 1
        if _facing_villain_aggression(state):
            self._aggressive += 1


def _facing_villain_aggression(state: GameState) -> bool:
    """True when the focal must answer a single villain's bet/raise (not a blind)."""
    if state.current_seat is None or state.to_call <= 0:
        return False
    villains = [p for p in state.active_players if p.seat != state.current_seat]
    if len(villains) != 1:  # only well-defined heads-up; ignore the read multiway
        return False
    villain = villains[0]
    if state.street is Street.PREFLOP:
        # committed is cumulative for the hand, so > big_blind isolates a real
        # raise from a posted blind or a limp (both leave committed == big_blind).
        return villain.committed > state.big_blind
    # Postflop both players enter the street matched, so any owed chips is a bet.
    return True


def _annotate(
    decision: PolicyDecision,
    *,
    maniac: bool,
    rate: float | None,
    observations: int,
    threshold: float,
) -> PolicyDecision:
    metadata = dict(decision.metadata)
    metadata["adaptive_route"] = "exploit" if maniac else "default"
    metadata["adaptive_aggression"] = round(rate, 4) if rate is not None else None
    metadata["adaptive_observations"] = observations
    metadata["adaptive_threshold"] = threshold
    return replace(decision, metadata=metadata)
