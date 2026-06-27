"""Weak-field exploitation: adjust the heuristic per the per-seat opponent read.

The S4-lite HU work plugged a short-stack leak versus one maniac. The Poker
Legends target is the opposite end: a ~100bb 6-max table of *weak* players, where
the money comes not from a GTO floor but from exploiting each opponent's specific
leak. :class:`FieldExploitPolicy` is the first slice of that layer.

It wraps the heuristic and, using :class:`~holdem_ai.opponents.OpponentModel`,
switches to an exploitative configuration when a *calling station* (loose-passive
"fish" — high VPIP, ~never raises) is live in the pot. Against a station you:

* **value-bet far thinner** — they call with worse, so betting only premiums
  (the base ``0.78`` threshold) leaves money on the table; drop to ``0.62``;
* **never bluff / semi-bluff** — they do not fold, so fold-equity bets just burn
  chips; lift the semi-bluff threshold out of reach;
* **size value bets bigger** — they call big, so charge the maximum for value.

Measured on the 6-max CRN harness this is worth roughly **+470 bb/100 over the
un-adapted heuristic versus a calling-station field**, with a delta near zero
versus a rock or maniac field (the changes only ever touch value lines that a
station pays off), so gating it behind the station read costs nothing elsewhere.
Nit and maniac exploitation (fold to a nit's bets, call a maniac down lighter)
reuse the same model and are the next slices — see ``docs/ai-strength.md``.
"""

from __future__ import annotations

from dataclasses import replace

from holdem_common import Action, GameState

from holdem_ai.heuristic import HeuristicConfig, HeuristicPolicy, PolicyDecision
from holdem_ai.opponents import OpponentModel, OpponentProfile

__all__ = ["FieldExploitPolicy", "station_config"]


def station_config(base: HeuristicConfig) -> HeuristicConfig:
    """Derive the calling-station exploit config: thin value, no bluffs, big sizing."""
    return replace(
        base,
        value_raise_threshold=0.62,
        protection_bet_threshold=0.56,
        semi_bluff_threshold=0.99,
        value_bet_pot_fraction=0.85,
        strong_value_bet_pot_fraction=1.0,
    )


class FieldExploitPolicy:
    """Heuristic that exploits calling stations, gated by a per-seat opponent read."""

    def __init__(
        self,
        *,
        base_config: HeuristicConfig | None = None,
        model: OpponentModel | None = None,
        station_overrides: HeuristicConfig | None = None,
    ) -> None:
        base = base_config or HeuristicConfig()
        self._model = model or OpponentModel()
        self._base_policy = HeuristicPolicy(base)
        self._station_policy = HeuristicPolicy(station_overrides or station_config(base))

    @property
    def model(self) -> OpponentModel:
        return self._model

    def reset(self) -> None:
        self._model.reset()

    def decide(self, state: GameState) -> Action:
        return self.explain(state).action

    def explain(self, state: GameState) -> PolicyDecision:
        self._model.observe(state)
        station = self._station_live(state)
        policy = self._station_policy if station else self._base_policy
        decision = policy.explain(state)
        return _annotate(decision, station=station, model=self._model, state=state)

    def _station_live(self, state: GameState) -> bool:
        if state.current_seat is None:
            return False
        return any(
            player.seat != state.current_seat
            and self._model.classify(player.seat) is OpponentProfile.STATION
            for player in state.active_players
        )


def _annotate(
    decision: PolicyDecision,
    *,
    station: bool,
    model: OpponentModel,
    state: GameState,
) -> PolicyDecision:
    metadata = dict(decision.metadata)
    metadata["exploit"] = "station" if station else "base"
    if state.current_seat is not None:
        metadata["opponent_profiles"] = {
            player.seat: model.classify(player.seat).value
            for player in state.active_players
            if player.seat != state.current_seat
        }
    return replace(decision, metadata=metadata)
