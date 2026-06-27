"""Per-seat opponent reads for field exploitation (6-max weak fields).

The S4-lite HU work classified a *single* villain by how often it forced us to
answer a bet. Beating a weak 6-max field needs the same idea generalised to a
*table*: a session-scoped, per-seat model of how loose and how aggressive each
opponent is, so the policy can exploit each one specifically — value-bet a calling
station thinner (and never bluff it), fold to a nit's bets, call a maniac down
lighter.

Like the HU read, everything is reconstructed from the snapshots the policy is
asked to act on (no action log, no opponent-turn callbacks). The two robust,
preflop-clean per-seat signals are:

* **VPIP** (voluntarily put in pot): the seat committed more than its blind — it
  is in the hand by choice, not just posting. A station/fish does this constantly;
  a nit almost never.
* **PFR** (preflop raise): the seat is the *unique* high commitment at the table
  and it exceeds a big blind — it put in the last raise. The uniqueness test is
  what separates a raiser from a caller: a pure calling station only ever *matches*
  the standing bet (tying the raiser), so it is never the lone top commit, while a
  maniac that raises is, until someone re-raises it.

Both signals are read **only from preflop snapshots**, because ``committed`` is
cumulative for the hand — postflop a caller's running total balloons past a big
blind and would masquerade as a raise. Each hand is counted once per seat (folding
in the most-complete view gathered across our preflop snapshots in that hand),
which keeps the rates stable. The station/nit/maniac fingerprints are well
separated — see ``docs/ai-strength.md``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum

from holdem_common import GameState, PlayerState, Street

__all__ = ["OpponentModel", "OpponentProfile", "SeatRead"]


class OpponentProfile(Enum):
    UNKNOWN = "unknown"
    STATION = "station"  # loose-passive: high VPIP, ~never raises -> exploit thin value, no bluff
    NIT = "nit"  # tight: low VPIP -> steal more, fold to its bets
    MANIAC = "maniac"  # over-aggressive: high PFR -> call lighter, do not bluff into
    BALANCED = "balanced"  # none of the above (or competent)


@dataclass(frozen=True, slots=True)
class SeatRead:
    seat: int
    hands: int
    vpip: float | None
    pfr: float | None
    profile: OpponentProfile


class OpponentModel:
    """Accumulate per-seat VPIP/PFR across hero decisions and classify each seat."""

    def __init__(
        self,
        *,
        min_hands: int = 12,
        station_vpip: float = 0.40,
        station_pfr_max: float = 0.14,
        nit_vpip_max: float = 0.20,
        maniac_pfr: float = 0.30,
    ) -> None:
        if min_hands < 1:
            raise ValueError("min_hands must be positive")
        self._min_hands = min_hands
        self._station_vpip = station_vpip
        self._station_pfr_max = station_pfr_max
        self._nit_vpip_max = nit_vpip_max
        self._maniac_pfr = maniac_pfr
        self._hands: Counter[int] = Counter()
        self._vpip: Counter[int] = Counter()
        self._pfr: Counter[int] = Counter()
        # Per-seat booleans for the hand currently being observed; flushed into the
        # cumulative counters when a new hand_id arrives (counts each hand once,
        # using the most-complete view we saw of that hand).
        self._cur_hand: str | None = None
        self._cur_vpip: set[int] = set()
        self._cur_pfr: set[int] = set()
        self._cur_seen: set[int] = set()

    def reset(self) -> None:
        self._hands.clear()
        self._vpip.clear()
        self._pfr.clear()
        self._cur_hand = None
        self._cur_vpip.clear()
        self._cur_pfr.clear()
        self._cur_seen.clear()

    def observe(self, state: GameState) -> None:
        if state.current_seat is None:
            return
        if state.hand_id != self._cur_hand:
            self._flush()
            self._cur_hand = state.hand_id
        # Only preflop snapshots feed VPIP/PFR: postflop the cumulative `committed`
        # of a mere caller exceeds a big blind and would look like a raise.
        if state.street is not Street.PREFLOP:
            return
        big_blind = state.big_blind
        max_committed = max(player.committed for player in state.players)
        unique_top = sum(1 for player in state.players if player.committed == max_committed) == 1
        for player in state.players:
            if player.seat == state.current_seat:
                continue
            self._cur_seen.add(player.seat)
            if player.committed > _blind_baseline(player, state):
                self._cur_vpip.add(player.seat)
            # A raise = the lone top commitment above a big blind. A caller only
            # ties the standing bet, so it never trips this.
            if unique_top and player.committed == max_committed and player.committed > big_blind:
                self._cur_pfr.add(player.seat)

    def read(self, seat: int) -> SeatRead:
        hands, vpip_count, pfr_count = self._counts(seat)
        vpip = vpip_count / hands if hands else None
        pfr = pfr_count / hands if hands else None
        return SeatRead(
            seat=seat,
            hands=hands,
            vpip=vpip,
            pfr=pfr,
            profile=self._classify(hands, vpip, pfr),
        )

    def classify(self, seat: int) -> OpponentProfile:
        return self.read(seat).profile

    def _classify(self, hands: int, vpip: float | None, pfr: float | None) -> OpponentProfile:
        if hands < self._min_hands or vpip is None or pfr is None:
            return OpponentProfile.UNKNOWN
        if pfr >= self._maniac_pfr:
            return OpponentProfile.MANIAC
        if vpip >= self._station_vpip and pfr <= self._station_pfr_max:
            return OpponentProfile.STATION
        if vpip <= self._nit_vpip_max:
            return OpponentProfile.NIT
        return OpponentProfile.BALANCED

    def _counts(self, seat: int) -> tuple[int, int, int]:
        # Cumulative counters plus the not-yet-flushed current hand, so a read is
        # accurate even mid-hand and immediately after the final hand observed.
        hands = self._hands[seat]
        vpip = self._vpip[seat]
        pfr = self._pfr[seat]
        if seat in self._cur_seen:
            hands += 1
            vpip += 1 if seat in self._cur_vpip else 0
            pfr += 1 if seat in self._cur_pfr else 0
        return hands, vpip, pfr

    def _flush(self) -> None:
        for seat in self._cur_seen:
            self._hands[seat] += 1
            if seat in self._cur_vpip:
                self._vpip[seat] += 1
            if seat in self._cur_pfr:
                self._pfr[seat] += 1
        self._cur_seen.clear()
        self._cur_vpip.clear()
        self._cur_pfr.clear()


def _blind_baseline(player: PlayerState, state: GameState) -> int:
    if player.big_blind:
        return state.big_blind
    if player.small_blind:
        return state.small_blind
    return 0
