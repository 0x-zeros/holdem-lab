"""Tests for per-seat opponent reads and the calling-station exploit policy."""

from __future__ import annotations

import pytest
from holdem_ai.field import FieldExploitPolicy, nit_config, station_config
from holdem_ai.heuristic import HeuristicConfig, PolicyDecision
from holdem_ai.opponents import OpponentModel, OpponentProfile
from holdem_common import Action, ActionType, GameState, PlayerState, Street

_CHECK = (Action(ActionType.CHECK),)


def _preflop_state(
    *,
    hand_id: str,
    opp_seat: int,
    opp_committed: int,
    hero_committed: int = 2,
    big_blind: int = 2,
    street: Street = Street.PREFLOP,
    seats: int = 6,
    active_seats: frozenset[int] | None = None,
) -> GameState:
    """A 6-max state where one opponent has a chosen commitment; hero acts at 0."""
    players = []
    for seat in range(seats):
        if seat == 0:
            committed = hero_committed
        elif seat == opp_seat:
            committed = opp_committed
        else:
            committed = 0
        players.append(
            PlayerState(
                seat=seat,
                stack=200,
                committed=committed,
                active=active_seats is None or seat in active_seats,
            )
        )
    return GameState(
        hand_id=hand_id,
        street=street,
        players=tuple(players),
        board=(),
        pots=(),
        current_seat=0,
        button_seat=0,
        small_blind=big_blind // 2,
        big_blind=big_blind,
        min_raise=big_blind,
        to_call=0,
        legal_actions=_CHECK,
    )


def _feed(
    model: OpponentModel,
    *,
    opp_seat: int,
    opp_committed: int,
    hands: int,
    hero_committed: int = 2,
    street: Street = Street.PREFLOP,
) -> None:
    for i in range(hands):
        model.observe(
            _preflop_state(
                hand_id=f"h{i}",
                opp_seat=opp_seat,
                opp_committed=opp_committed,
                hero_committed=hero_committed,
                street=street,
            )
        )


# --- OpponentModel classification --------------------------------------------


def test_unknown_before_min_hands() -> None:
    model = OpponentModel(min_hands=12)
    _feed(model, opp_seat=2, opp_committed=2, hands=5)
    assert model.classify(2) is OpponentProfile.UNKNOWN


def test_station_is_high_vpip_no_raise() -> None:
    # Limps every hand (committed == bb): voluntarily in, never the lone top commit.
    model = OpponentModel(min_hands=12)
    _feed(model, opp_seat=2, opp_committed=2, hands=20)
    read = model.read(2)
    assert read.vpip == 1.0
    assert read.pfr == 0.0
    assert read.profile is OpponentProfile.STATION


def test_nit_is_low_vpip() -> None:
    # Folds every hand (commits nothing).
    model = OpponentModel(min_hands=12)
    _feed(model, opp_seat=3, opp_committed=0, hands=20)
    assert model.read(3).vpip == 0.0
    assert model.classify(3) is OpponentProfile.NIT


def test_maniac_is_high_raise() -> None:
    # Raises every hand: the lone top commitment above a big blind.
    model = OpponentModel(min_hands=12)
    _feed(model, opp_seat=4, opp_committed=6, hands=20)
    read = model.read(4)
    assert read.pfr == 1.0
    assert read.profile is OpponentProfile.MANIAC


def test_pure_caller_is_not_counted_as_raise() -> None:
    # Opponent only ever MATCHES the hero's commitment (ties the top, never unique).
    model = OpponentModel(min_hands=12)
    _feed(model, opp_seat=2, opp_committed=6, hero_committed=6, hands=20)
    assert model.read(2).pfr == 0.0  # tie, not a raise
    assert model.read(2).vpip == 1.0
    assert model.classify(2) is OpponentProfile.STATION


def test_postflop_snapshots_are_ignored() -> None:
    # Cumulative committed balloons postflop; it must not look like a preflop raise.
    model = OpponentModel(min_hands=1)
    _feed(model, opp_seat=2, opp_committed=40, hands=10, street=Street.FLOP)
    assert model.read(2).hands == 0  # no preflop evidence gathered
    assert model.classify(2) is OpponentProfile.UNKNOWN


def test_same_hand_counted_once() -> None:
    model = OpponentModel(min_hands=1)
    for _ in range(5):  # five snapshots, one hand id
        model.observe(_preflop_state(hand_id="same", opp_seat=2, opp_committed=2))
    assert model.read(2).hands == 1


def test_reset_clears_reads() -> None:
    model = OpponentModel(min_hands=1)
    _feed(model, opp_seat=2, opp_committed=2, hands=5)
    model.reset()
    assert model.read(2).hands == 0
    assert model.classify(2) is OpponentProfile.UNKNOWN


def test_model_rejects_bad_min_hands() -> None:
    with pytest.raises(ValueError, match="min_hands"):
        OpponentModel(min_hands=0)


# --- station config ----------------------------------------------------------


def test_station_config_is_thinner_value_and_no_bluff() -> None:
    base = HeuristicConfig()
    exploit = station_config(base)
    assert exploit.value_raise_threshold < base.value_raise_threshold
    assert exploit.semi_bluff_threshold > base.semi_bluff_threshold
    assert exploit.strong_value_bet_pot_fraction >= base.strong_value_bet_pot_fraction


def test_nit_config_folds_more() -> None:
    base = HeuristicConfig()
    exploit = nit_config(base)
    assert exploit.continue_threshold > base.continue_threshold
    assert exploit.marginal_threshold > base.marginal_threshold
    assert exploit.marginal_call_price_fraction < base.marginal_call_price_fraction


# --- FieldExploitPolicy routing ----------------------------------------------


class _Spy:
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls = 0

    def explain(self, state: GameState) -> PolicyDecision:
        self.calls += 1
        return PolicyDecision(
            action=state.legal_actions[0],
            reason=self.label,
            strength=0.0,
            required_equity=None,
            metadata={},
        )


def test_exploits_only_when_a_station_is_live() -> None:
    # Pre-train a model so seat 2 is a STATION and seat 3 is a NIT, then route.
    model = OpponentModel(min_hands=4)
    _feed(model, opp_seat=2, opp_committed=2, hands=10)  # station
    _feed(model, opp_seat=3, opp_committed=0, hands=10)  # nit
    assert model.classify(2) is OpponentProfile.STATION
    assert model.classify(3) is OpponentProfile.NIT

    policy = FieldExploitPolicy(model=model)
    base_spy, station_spy = _Spy("base"), _Spy("station")
    policy._base_policy = base_spy  # type: ignore[assignment]
    policy._station_policy = station_spy  # type: ignore[assignment]

    # Hero (0) versus the nit (3) alone (station seat 2 has folded): base config.
    nit_only = _preflop_state(
        hand_id="x", opp_seat=3, opp_committed=0, active_seats=frozenset({0, 3})
    )
    policy.explain(nit_only)
    assert (base_spy.calls, station_spy.calls) == (1, 0)

    # Station (2) live in the pot -> station config.
    station_live = _preflop_state(
        hand_id="y", opp_seat=2, opp_committed=2, active_seats=frozenset({0, 2, 3})
    )
    policy.explain(station_live)
    assert (base_spy.calls, station_spy.calls) == (1, 1)


def _facing_bet_state(
    *, aggressor: int, aggressor_committed: int = 10, seats: int = 4
) -> GameState:
    """Postflop state where hero (0) owes chips to a single bettor `aggressor`."""
    players = tuple(
        PlayerState(
            seat=seat,
            stack=200,
            committed=aggressor_committed if seat == aggressor else 0,
        )
        for seat in range(seats)
    )
    return GameState(
        hand_id="postflop",
        street=Street.FLOP,
        players=players,
        board=(),
        pots=(),
        current_seat=0,
        button_seat=0,
        small_blind=1,
        big_blind=2,
        min_raise=2,
        to_call=aggressor_committed,
        legal_actions=(
            Action(ActionType.FOLD),
            Action(ActionType.CALL, amount=aggressor_committed),
        ),
    )


def test_folds_to_a_nits_bet() -> None:
    model = OpponentModel(min_hands=4)
    _feed(model, opp_seat=2, opp_committed=0, hands=10)  # nit: never voluntarily in
    assert model.classify(2) is OpponentProfile.NIT

    policy = FieldExploitPolicy(model=model)
    base, station, nit = _Spy("base"), _Spy("station"), _Spy("nit")
    policy._base_policy = base  # type: ignore[assignment]
    policy._station_policy = station  # type: ignore[assignment]
    policy._nit_policy = nit  # type: ignore[assignment]

    policy.explain(_facing_bet_state(aggressor=2))
    assert (nit.calls, station.calls, base.calls) == (1, 0, 0)


def test_does_not_fold_to_a_maniacs_bet() -> None:
    # A maniac (high VPIP) is never read as a nit, so we never fold to its bluffs.
    model = OpponentModel(min_hands=4)
    _feed(model, opp_seat=2, opp_committed=6, hands=10)  # raises every hand
    assert model.classify(2) is OpponentProfile.MANIAC

    policy = FieldExploitPolicy(model=model)
    base, nit = _Spy("base"), _Spy("nit")
    policy._base_policy = base  # type: ignore[assignment]
    policy._nit_policy = nit  # type: ignore[assignment]

    policy.explain(_facing_bet_state(aggressor=2))
    assert (nit.calls, base.calls) == (0, 1)


def test_metadata_exposes_exploit_and_profiles() -> None:
    model = OpponentModel(min_hands=4)
    _feed(model, opp_seat=2, opp_committed=2, hands=10)
    policy = FieldExploitPolicy(model=model)
    decision = policy.explain(_preflop_state(hand_id="z", opp_seat=2, opp_committed=2, seats=4))
    assert decision.metadata["exploit"] == "station"
    profiles = decision.metadata["opponent_profiles"]
    assert isinstance(profiles, dict)
    assert profiles[2] == "station"
