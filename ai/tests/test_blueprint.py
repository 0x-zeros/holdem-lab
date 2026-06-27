from holdem_ai import Policy
from holdem_ai.blueprint import PushFoldPolicy
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
