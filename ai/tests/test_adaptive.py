"""Tests for the opponent-adaptive routing policy.

Two layers: precise unit tests of the facing-aggression detector and the router
on hand-built states (no game variance), then CRN integration invariants that pin
the two-way win — adaptive must reproduce ``hybrid`` versus a tag while erasing the
heuristic's short-stack leak versus a maniac.
"""

from __future__ import annotations

import pytest
from holdem_ai.adaptive import AdaptivePolicy, _facing_villain_aggression
from holdem_ai.evaluate import evaluate_match, profile_from_name
from holdem_ai.heuristic import PolicyDecision
from holdem_common import Action, ActionType, GameState, PlayerState, Street

_CHECK = (Action(ActionType.CHECK),)


def _state(
    *,
    street: Street,
    to_call: int,
    hero_seat: int = 0,
    hero_committed: int = 0,
    villain_committed: int = 0,
    big_blind: int = 2,
    extra_players: tuple[PlayerState, ...] = (),
) -> GameState:
    committed = {hero_seat: hero_committed, 1 - hero_seat: villain_committed}
    players = (
        PlayerState(seat=0, stack=100, committed=committed[0]),
        PlayerState(seat=1, stack=100, committed=committed[1]),
        *extra_players,
    )
    return GameState(
        hand_id="t",
        street=street,
        players=players,
        board=(),
        pots=(),
        current_seat=hero_seat,
        button_seat=0,
        small_blind=big_blind // 2,
        big_blind=big_blind,
        min_raise=big_blind,
        to_call=to_call,
        legal_actions=_CHECK,
    )


# --- detector ----------------------------------------------------------------


def test_no_chips_owed_is_never_aggression() -> None:
    assert not _facing_villain_aggression(_state(street=Street.FLOP, to_call=0))
    assert not _facing_villain_aggression(_state(street=Street.PREFLOP, to_call=0))


def test_postflop_any_owed_is_aggression() -> None:
    # Postflop both players enter matched, so owing chips means the villain bet.
    assert _facing_villain_aggression(_state(street=Street.FLOP, to_call=6, villain_committed=6))


def test_preflop_blind_is_not_aggression() -> None:
    # Hero is SB facing the unraised big blind: owes chips, but the villain only
    # posted the blind (committed == big_blind), so it is not a raise.
    state = _state(
        street=Street.PREFLOP, to_call=1, hero_committed=1, villain_committed=2, big_blind=2
    )
    assert not _facing_villain_aggression(state)


def test_preflop_raise_is_aggression() -> None:
    state = _state(
        street=Street.PREFLOP, to_call=4, hero_committed=2, villain_committed=6, big_blind=2
    )
    assert _facing_villain_aggression(state)


def test_multiway_is_ignored() -> None:
    # The heads-up attribution is undefined with two villains; do not count it.
    third = (PlayerState(seat=2, stack=100, committed=6),)
    state = _state(street=Street.FLOP, to_call=6, villain_committed=6, extra_players=third)
    assert not _facing_villain_aggression(state)


# --- router ------------------------------------------------------------------


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

    def decide(self, state: GameState) -> Action:
        return self.explain(state).action


def _adaptive_with_spies(**kwargs: object) -> tuple[AdaptivePolicy, _Spy, _Spy]:
    exploit, default = _Spy("exploit"), _Spy("default")
    policy = AdaptivePolicy(exploit=exploit, default=default, **kwargs)  # type: ignore[arg-type]
    return policy, exploit, default


def test_routes_to_default_during_warmup_then_exploits() -> None:
    policy, exploit, default = _adaptive_with_spies(min_observations=5, maniac_threshold=0.55)
    aggressive = _state(street=Street.FLOP, to_call=6, villain_committed=6)
    for _ in range(8):
        policy.explain(aggressive)
    # Calls 1..4 lack the 5-observation warm-up -> default; calls 5..8 classify
    # (rate 1.0 >= 0.55) -> exploit.
    assert default.calls == 4
    assert exploit.calls == 4
    assert policy.classified_maniac()


def test_passive_opponent_never_exploited() -> None:
    policy, exploit, default = _adaptive_with_spies(min_observations=5, maniac_threshold=0.55)
    passive = _state(street=Street.FLOP, to_call=0)
    for _ in range(30):
        policy.explain(passive)
    assert exploit.calls == 0
    assert default.calls == 30
    assert policy.aggression == 0.0
    assert not policy.classified_maniac()


def test_reset_forgets_the_read() -> None:
    policy, exploit, default = _adaptive_with_spies(min_observations=5, maniac_threshold=0.55)
    aggressive = _state(street=Street.FLOP, to_call=6, villain_committed=6)
    for _ in range(8):
        policy.explain(aggressive)
    assert policy.classified_maniac()
    policy.reset()
    assert policy.aggression is None
    assert not policy.classified_maniac()


def test_metadata_exposes_route_and_read() -> None:
    policy, _, _ = _adaptive_with_spies(min_observations=2, maniac_threshold=0.55)
    aggressive = _state(street=Street.FLOP, to_call=6, villain_committed=6)
    policy.explain(aggressive)  # warm-up: below min_observations
    decision = policy.explain(aggressive)  # now classified
    assert decision.metadata["adaptive_route"] == "exploit"
    assert decision.metadata["adaptive_aggression"] == 1.0
    assert decision.metadata["adaptive_observations"] == 2


def test_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError, match="maniac_threshold"):
        AdaptivePolicy(maniac_threshold=0.0)
    with pytest.raises(ValueError, match="maniac_threshold"):
        AdaptivePolicy(maniac_threshold=1.5)
    with pytest.raises(ValueError, match="min_observations"):
        AdaptivePolicy(min_observations=0)


# --- CRN integration: the two-way win ----------------------------------------

_BB = 2
_STACK_8BB = 8 * _BB


def _match(focal: str, opponent: str, *, pairs: int = 80, seed: int = 20260627) -> float:
    report = evaluate_match(
        profile_from_name(focal),
        profile_from_name(opponent),
        pairs=pairs,
        seed=seed,
        starting_stack=_STACK_8BB,
        small_blind=1,
        big_blind=_BB,
        bootstrap=1,
    )
    return report.bb_per_100


def test_adaptive_matches_hybrid_versus_tag_exactly() -> None:
    # Against a tag the aggression rate (~0.21) never crosses 0.55, so adaptive
    # plays its default (the hybrid guardrail) on every decision -> identical play.
    assert _match("adaptive", "tag") == _match("hybrid", "tag")


def test_adaptive_erases_the_maniac_leak() -> None:
    # The heuristic bleeds to a short-stack maniac; adaptive flips to the open-jam
    # exploit and recovers tens of bb/100 over it.
    adaptive = _match("adaptive", "maniac")
    current = _match("current", "maniac")
    assert current < -20.0  # the leak is real and large
    assert adaptive > current + 20.0


def test_adaptive_match_is_reproducible() -> None:
    assert _match("adaptive", "maniac") == _match("adaptive", "maniac")
