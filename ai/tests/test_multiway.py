"""Multiway (3-handed) smoke test: policies must not crash, and the heads-up
push/fold blueprint must never engage at a 3-handed table (the release-blocker
the external review flagged — applying a HU model in a 6-max-style spot)."""

from collections import Counter

from holdem_ai.profiles import PolicyProfile, profile_from_name
from holdem_engine import HoldemConfig, HoldemEnv


def _play_three_handed(
    seat_profiles: dict[int, PolicyProfile], *, hands: int, seed: int, stack: int
) -> tuple[int, Counter[str]]:
    pushfold_acted = 0
    pushfold_blueprint_reasons: Counter[str] = Counter()
    for hand_index in range(hands):
        button = hand_index % 3
        config = HoldemConfig(
            starting_stacks=(stack, stack, stack),
            small_blind=1,
            big_blind=2,
            hand_id=f"mw-{hand_index}",
            button_seat=button,
            small_blind_seat=(button + 1) % 3,
            big_blind_seat=(button + 2) % 3,
        )
        env = HoldemEnv(config)
        state = env.reset(seed=seed + hand_index)
        steps = 0
        while state.current_seat is not None:
            assert steps < 300, "3-handed hand failed to terminate"
            seat = state.current_seat
            profile = seat_profiles[seat]
            decision = profile.policy.explain(env.observe(seat=seat))
            if profile.name == "pushfold":
                pushfold_acted += 1
                if decision.reason.startswith("pushfold"):
                    pushfold_blueprint_reasons[decision.reason] += 1
            state = env.step(decision.action).observation
            steps += 1
        assert sum(env.facade.payoffs.values()) == 0  # every hand is zero-sum
    return pushfold_acted, pushfold_blueprint_reasons


def test_three_handed_never_engages_pushfold_blueprint() -> None:
    seat_profiles = {
        0: profile_from_name("pushfold"),
        1: profile_from_name("current"),
        2: profile_from_name("rock"),
    }
    # 8bb effective: short enough that the blueprint WOULD fire heads-up, so the
    # guard is genuinely exercised rather than skipped for being too deep.
    acted, blueprint_reasons = _play_three_handed(seat_profiles, hands=40, seed=100, stack=16)
    assert acted > 0  # the push/fold policy actually had to act at the 3-handed table
    assert blueprint_reasons == Counter()  # ...but never via the heads-up blueprint


def test_three_handed_mixed_references_complete_without_crashing() -> None:
    seat_profiles = {
        0: profile_from_name("tag"),
        1: profile_from_name("three_bet_jammer"),
        2: profile_from_name("maniac"),
    }
    acted, blueprint_reasons = _play_three_handed(seat_profiles, hands=15, seed=7, stack=80)
    assert blueprint_reasons == Counter()
