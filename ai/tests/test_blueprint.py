from holdem_ai import Policy
from holdem_ai.blueprint import PushFoldPolicy
from holdem_ai.profiles import CFR_PROFILE_NAMES, profile_from_name
from holdem_common import Action, ActionType, Card, GameState, PlayerState, Pot, Street


def cards(*codes: str) -> tuple[Card, ...]:
    return tuple(Card.from_code(code) for code in codes)


def button_open_state(hole: tuple[Card, ...], *, effective: int = 18) -> GameState:
    # Heads-up button/small blind to act, unraised, short effective stack (bb=2).
    return GameState(
        hand_id="pf-test",
        street=Street.PREFLOP,
        players=(
            PlayerState(seat=0, stack=effective - 1, committed=1, hole_cards=hole),
            PlayerState(seat=1, stack=effective - 2, committed=2, hole_cards=()),
        ),
        board=(),
        pots=(Pot(amount=3, eligible_seats=frozenset({0, 1})),),
        current_seat=0,
        button_seat=0,
        small_blind=1,
        big_blind=2,
        min_raise=4,
        to_call=1,
        legal_actions=(
            Action(ActionType.FOLD),
            Action(ActionType.CALL, amount=1),
            Action(ActionType.RAISE, amount=4, min_amount=4, max_amount=effective - 1),
            Action(
                ActionType.ALL_IN,
                amount=effective - 1,
                min_amount=effective - 1,
                max_amount=effective - 1,
            ),
        ),
    )


def test_pushfold_policy_satisfies_protocol() -> None:
    assert isinstance(PushFoldPolicy(), Policy)


def test_short_stack_button_jams_premium_and_folds_trash() -> None:
    policy = PushFoldPolicy()

    jam = policy.explain(button_open_state(cards("As", "Ah"), effective=18))  # 9bb
    assert jam.action.action_type is ActionType.ALL_IN
    assert jam.reason == "pushfold_jam"
    assert jam.metadata["source"] == "cfr_push_fold"

    fold = policy.explain(button_open_state(cards("7c", "2d"), effective=18))
    assert fold.action.action_type is ActionType.FOLD
    assert fold.reason == "pushfold_fold"


def test_deep_stack_falls_back_to_heuristic() -> None:
    policy = PushFoldPolicy(max_jam_bb=12)
    # 100bb effective: push/fold does not apply, so the heuristic open takes over.
    decision = policy.explain(button_open_state(cards("As", "Ah"), effective=200))
    assert not decision.reason.startswith("pushfold")


def test_blueprints_are_cached_per_stack() -> None:
    policy = PushFoldPolicy()
    policy.explain(button_open_state(cards("As", "Ah"), effective=18))  # 9bb -> solves once
    policy.explain(button_open_state(cards("Ks", "Kh"), effective=18))  # 9bb -> cached
    assert set(policy._cache) == {9}


def six_max_button_state(hole: tuple[Card, ...]) -> GameState:
    # Three active players: the button is NOT the small blind, so the heads-up
    # push/fold model must not apply.
    return GameState(
        hand_id="6max",
        street=Street.PREFLOP,
        players=(
            PlayerState(seat=0, stack=17, committed=1, hole_cards=hole),
            PlayerState(seat=1, stack=16, committed=2, hole_cards=()),
            PlayerState(seat=2, stack=18, committed=0, hole_cards=()),
        ),
        board=(),
        pots=(Pot(amount=3, eligible_seats=frozenset({0, 1, 2})),),
        current_seat=0,
        button_seat=0,
        small_blind=1,
        big_blind=2,
        min_raise=4,
        to_call=1,
        legal_actions=(
            Action(ActionType.FOLD),
            Action(ActionType.CALL, amount=1),
            Action(ActionType.ALL_IN, amount=17, min_amount=17, max_amount=17),
        ),
    )


def facing_jam_state(hole: tuple[Card, ...], *, hero_stack: int = 90) -> GameState:
    # Big blind facing an all-in button that we *cover* (hero stack > villain jam).
    villain_jam = 20
    return GameState(
        hand_id="jam",
        street=Street.PREFLOP,
        players=(
            PlayerState(seat=0, stack=0, committed=villain_jam, hole_cards=(), all_in=True),
            PlayerState(seat=1, stack=hero_stack, committed=2, hole_cards=hole),
        ),
        board=(),
        pots=(Pot(amount=villain_jam + 2, eligible_seats=frozenset({0, 1})),),
        current_seat=1,
        button_seat=0,
        small_blind=1,
        big_blind=2,
        min_raise=4,
        to_call=villain_jam - 2,
        legal_actions=(Action(ActionType.FOLD), Action(ActionType.CALL, amount=villain_jam - 2)),
    )


def folded_to_two_survivors_state(hole: tuple[Card, ...]) -> GameState:
    # Three players were dealt; the small blind folded, leaving a short-stack jam
    # between button (all-in) and big blind. It is NOT a clean heads-up table — the
    # folded small blind's dead money shifts the pot odds — so the blueprint, solved
    # for a no-dead-money HU game, must stay out of it.
    return GameState(
        hand_id="reduced",
        street=Street.PREFLOP,
        players=(
            PlayerState(seat=0, stack=0, committed=10, hole_cards=(), all_in=True),
            PlayerState(seat=1, stack=0, committed=1, hole_cards=(), active=False),
            PlayerState(seat=2, stack=8, committed=2, hole_cards=hole),
        ),
        board=(),
        pots=(Pot(amount=13, eligible_seats=frozenset({0, 2})),),
        current_seat=2,
        button_seat=0,
        small_blind=1,
        big_blind=2,
        min_raise=4,
        to_call=8,
        legal_actions=(Action(ActionType.FOLD), Action(ActionType.CALL, amount=8)),
    )


def test_six_max_spot_never_uses_pushfold() -> None:
    decision = PushFoldPolicy().explain(six_max_button_state(cards("As", "Ah")))
    assert not decision.reason.startswith("pushfold")  # hard heads-up guard


def test_reduced_multiway_pot_never_uses_pushfold() -> None:
    # active_players is 2 here, but players (dealt-in) is 3 -> not a heads-up table.
    decision = PushFoldPolicy().explain(folded_to_two_survivors_state(cards("As", "Ah")))
    assert not decision.reason.startswith("pushfold")


def test_big_blind_handles_a_covered_jam() -> None:
    policy = PushFoldPolicy()
    call = policy.explain(facing_jam_state(cards("As", "Ah")))
    assert call.action.action_type is ActionType.CALL
    assert call.reason == "pushfold_call"
    assert call.metadata["effective_bb"] == 10.0  # min(20, 92) / 2, not hero's 46bb

    fold = policy.explain(facing_jam_state(cards("3c", "2d")))
    assert fold.action.action_type is ActionType.FOLD
    assert fold.reason == "pushfold_overfold"


def test_defend_only_skips_opening_but_keeps_the_calling_floor() -> None:
    policy = PushFoldPolicy(defend_only=True)
    # Button open spot: defend_only leaves opening to the fallback heuristic,
    # which preserves value against opponents that defend correctly.
    opening = policy.explain(button_open_state(cards("As", "Ah"), effective=18))
    assert not opening.reason.startswith("pushfold")
    # Facing a jam: the blueprint still supplies the unexploitable calling floor.
    call = policy.explain(facing_jam_state(cards("As", "Ah")))
    assert call.reason == "pushfold_call"


def test_cfr_profiles_resolve_to_policies() -> None:
    assert "hybrid" in CFR_PROFILE_NAMES
    for name in CFR_PROFILE_NAMES:
        profile = profile_from_name(name)
        assert profile.name == name
        assert isinstance(profile.policy, Policy)
    # The hybrid profile is the strict-safe guardrail (defence only).
    hybrid = profile_from_name("hybrid").policy
    assert isinstance(hybrid, PushFoldPolicy)
    assert hybrid._defend_only is True


def test_decision_metadata_carries_probability_and_mode() -> None:
    decision = PushFoldPolicy(mode="mixed").explain(button_open_state(cards("As", "Ah")))
    assert decision.metadata["mode"] == "mixed"
    assert decision.metadata["blueprint_stack_key"] == 9
    assert 0.0 <= float(decision.metadata["blueprint_action_prob"]) <= 1.0  # type: ignore[arg-type]


def test_mixed_mode_is_reproducible_per_state() -> None:
    policy = PushFoldPolicy(mode="mixed")
    state = button_open_state(cards("As", "Ah"))
    assert policy.decide(state) == policy.decide(state)


def test_pure_mode_takes_argmax() -> None:
    decision = PushFoldPolicy(mode="pure").explain(button_open_state(cards("As", "Ah")))
    assert decision.action.action_type is ActionType.ALL_IN
    assert decision.metadata["mode"] == "pure"


def test_invalid_mode_rejected() -> None:
    try:
        PushFoldPolicy(mode="nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid mode")
