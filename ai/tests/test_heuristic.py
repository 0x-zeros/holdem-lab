from collections.abc import Iterator

import pytest
from holdem_ai import (
    decide,
    estimate_private_strength,
    estimate_showdown_equity,
    evaluate_best_hand,
    explain_decision,
    reset_decision_policy,
)
from holdem_common import Action, ActionType, Card, GameState, PlayerState, Pot, Street


def cards(*codes: str) -> tuple[Card, ...]:
    return tuple(Card.from_code(code) for code in codes)


@pytest.fixture(autouse=True)
def reset_public_decision_policy() -> Iterator[None]:
    reset_decision_policy()
    yield
    reset_decision_policy()


def state_with(
    *,
    hole_cards: tuple[Card, ...],
    legal_actions: tuple[Action, ...],
    to_call: int = 0,
    pot: int = 12,
    committed: int = 0,
    board: tuple[Card, ...] = (),
    current_seat: int | None = 0,
) -> GameState:
    return GameState(
        hand_id="test-hand",
        street=Street.PREFLOP if not board else Street.FLOP,
        players=(
            PlayerState(seat=0, stack=100 - committed, committed=committed, hole_cards=hole_cards),
            PlayerState(seat=1, stack=100, committed=0, hole_cards=()),
        ),
        board=board,
        pots=(Pot(amount=pot, eligible_seats=frozenset({0, 1})),),
        current_seat=current_seat,
        button_seat=0,
        small_blind=1,
        big_blind=2,
        min_raise=4,
        to_call=to_call,
        legal_actions=legal_actions,
    )


def test_estimate_private_strength_ranks_premium_above_trash() -> None:
    assert estimate_private_strength(cards("As", "Ah")) > estimate_private_strength(
        cards("7c", "2d")
    )


def test_decide_checks_weak_hand_when_free() -> None:
    state = state_with(
        hole_cards=cards("7c", "2d"),
        legal_actions=(
            Action(ActionType.CHECK),
            Action(ActionType.BET, amount=6, min_amount=6, max_amount=100),
        ),
    )

    assert decide(state).action_type is ActionType.CHECK


def test_decide_value_bets_premium_hand_when_free() -> None:
    state = state_with(
        hole_cards=cards("As", "Ah"),
        legal_actions=(
            Action(ActionType.CHECK),
            Action(ActionType.BET, amount=6, min_amount=6, max_amount=100),
        ),
    )

    action = decide(state)

    assert action.action_type is ActionType.BET
    assert action.amount >= 6
    assert action.min_amount == 6
    assert action.max_amount == 100


def test_explain_decision_returns_action_reason_and_metadata() -> None:
    state = state_with(
        hole_cards=cards("7c", "2d"),
        legal_actions=(
            Action(ActionType.CHECK),
            Action(ActionType.BET, amount=6, min_amount=6, max_amount=100),
        ),
    )

    decision = explain_decision(state)

    assert decision.action.action_type is ActionType.CHECK
    assert decision.reason == "preflop_check"
    assert decision.required_equity is None
    assert decision.metadata["made_hand"] == "high_card"
    assert decision.metadata["active_opponents"] == 1
    assert decision.metadata["exploit"] == "base"


def test_explain_decision_default_policy_exploits_station_read() -> None:
    reset_decision_policy()
    try:
        decision = None
        for hand_index in range(13):
            decision = explain_decision(
                GameState(
                    hand_id=f"api-station-{hand_index}",
                    street=Street.PREFLOP,
                    players=(
                        PlayerState(
                            seat=0,
                            stack=98,
                            committed=2,
                            hole_cards=cards("7c", "2d"),
                        ),
                        PlayerState(seat=1, stack=98, committed=2),
                    ),
                    board=(),
                    pots=(Pot(amount=4, eligible_seats=frozenset({0, 1})),),
                    current_seat=0,
                    button_seat=0,
                    small_blind=1,
                    big_blind=2,
                    min_raise=4,
                    to_call=0,
                    legal_actions=(Action(ActionType.CHECK),),
                )
            )

        assert decision is not None
        assert decision.metadata["exploit"] == "station"
        assert decision.metadata["opponent_profiles"] == {1: "station"}
    finally:
        reset_decision_policy()


def test_decide_folds_weak_hand_to_bad_price() -> None:
    state = state_with(
        hole_cards=cards("7c", "2d"),
        legal_actions=(
            Action(ActionType.FOLD),
            Action(ActionType.CALL, amount=50),
        ),
        to_call=50,
        pot=10,
    )

    assert decide(state).action_type is ActionType.FOLD


def test_decide_calls_marginal_hand_with_good_price() -> None:
    state = state_with(
        hole_cards=cards("9c", "8c"),
        legal_actions=(
            Action(ActionType.FOLD),
            Action(ActionType.CALL, amount=2),
        ),
        to_call=2,
        pot=60,
    )

    action = decide(state)

    assert action.action_type is ActionType.CALL
    assert action.amount == 2


def test_decide_raises_premium_hand_over_call() -> None:
    state = state_with(
        hole_cards=cards("As", "Ah"),
        legal_actions=(
            Action(ActionType.FOLD),
            Action(ActionType.CALL, amount=2),
            Action(ActionType.RAISE, amount=6, min_amount=6, max_amount=100),
            Action(ActionType.ALL_IN, amount=100, min_amount=100, max_amount=100),
        ),
        to_call=2,
        committed=2,
        pot=6,
    )

    action = decide(state)

    assert action.action_type is ActionType.RAISE
    assert action.amount >= 6


def test_decide_semi_bluff_bets_strong_flush_draw() -> None:
    state = state_with(
        hole_cards=cards("Ah", "Kh"),
        board=cards("Qh", "7h", "2c"),
        legal_actions=(
            Action(ActionType.CHECK),
            Action(ActionType.BET, amount=6, min_amount=6, max_amount=100),
        ),
        pot=40,
    )

    decision = explain_decision(state)

    assert decision.action.action_type is ActionType.BET
    assert decision.action.amount == 18
    assert decision.reason == "semi_bluff"
    assert decision.metadata["draw"] == "flush_draw"
    assert decision.metadata["outs"] == 9


def test_decide_protection_bets_top_pair_when_free() -> None:
    state = state_with(
        hole_cards=cards("Ah", "Qd"),
        board=cards("Qs", "7h", "2c"),
        legal_actions=(
            Action(ActionType.CHECK),
            Action(ActionType.BET, amount=6, min_amount=6, max_amount=100),
        ),
        pot=40,
    )

    decision = explain_decision(state)

    assert decision.action.action_type is ActionType.BET
    assert decision.action.amount == 20
    assert decision.reason == "protection_bet"
    assert decision.metadata["made_hand"] == "pair"


def test_preflop_opens_playable_button_hand_instead_of_limping() -> None:
    # Unraised pot, in position (button == current seat): a playable hand opens
    # to ~2.5x rather than limping.
    state = state_with(
        hole_cards=cards("Kc", "9d"),
        legal_actions=(
            Action(ActionType.FOLD),
            Action(ActionType.CALL, amount=1),
            Action(ActionType.RAISE, amount=4, min_amount=4, max_amount=200),
        ),
        to_call=1,
        pot=3,
        committed=1,
    )

    decision = explain_decision(state)

    assert decision.action.action_type is ActionType.RAISE
    assert decision.action.amount == 5  # 2.5x the big blind of 2
    assert decision.reason == "preflop_open"


def test_preflop_folds_trash_button_when_it_cannot_open() -> None:
    state = state_with(
        hole_cards=cards("7c", "2d"),
        legal_actions=(
            Action(ActionType.FOLD),
            Action(ActionType.CALL, amount=1),
            Action(ActionType.RAISE, amount=4, min_amount=4, max_amount=200),
        ),
        to_call=1,
        pot=3,
        committed=1,
    )

    decision = explain_decision(state)

    assert decision.action.action_type is ActionType.FOLD
    assert decision.reason == "preflop_fold"


def test_preflop_checks_trash_option_in_unraised_pot() -> None:
    # Big-blind option (free to check) with trash: take the free card, never bet.
    state = state_with(
        hole_cards=cards("7c", "2d"),
        legal_actions=(
            Action(ActionType.CHECK),
            Action(ActionType.RAISE, amount=4, min_amount=4, max_amount=200),
        ),
        to_call=0,
        pot=4,
        committed=2,
    )

    decision = explain_decision(state)

    assert decision.action.action_type is ActionType.CHECK
    assert decision.reason == "preflop_check"


def test_decide_value_bets_big_not_min_in_low_spr() -> None:
    # Set of aces in a large pot with a short effective stack: the pot-fraction
    # target exceeds the legal max, so the bot must commit (max), never min-bet.
    state = state_with(
        hole_cards=cards("As", "Ah"),
        board=cards("Ad", "7h", "2c"),
        legal_actions=(
            Action(ActionType.CHECK),
            Action(ActionType.BET, amount=10, min_amount=10, max_amount=40),
        ),
        pot=200,
    )

    decision = explain_decision(state)

    assert decision.action.action_type is ActionType.BET
    assert decision.action.amount == 40  # all-in for value, not the min bet of 10
    assert decision.reason == "value_bet"


def test_decide_calls_strong_draw_with_acceptable_price() -> None:
    state = state_with(
        hole_cards=cards("Ah", "Kh"),
        board=cards("Qh", "7h", "2c"),
        legal_actions=(
            Action(ActionType.FOLD),
            Action(ActionType.CALL, amount=20),
        ),
        to_call=20,
        pot=80,
    )

    decision = explain_decision(state)

    assert decision.action.action_type is ActionType.CALL
    assert decision.reason == "call_price"
    assert decision.required_equity == pytest.approx(0.20)


def test_estimate_private_strength_scores_made_flush_above_top_pair() -> None:
    flush_strength = estimate_private_strength(
        cards("Ah", "Kh"),
        cards("Qh", "7h", "2h", "3c", "4d"),
    )
    top_pair_strength = estimate_private_strength(
        cards("Ah", "Qd"),
        cards("Qs", "7h", "2c", "3d", "4s"),
    )

    assert flush_strength > top_pair_strength


def test_evaluate_best_hand_orders_flush_above_straight() -> None:
    flush = evaluate_best_hand(cards("Ah", "Kh", "Qh", "7h", "2h", "3c", "4d"))
    straight = evaluate_best_hand(cards("As", "Kd", "Qh", "Jc", "Ts", "3c", "4d"))

    assert flush > straight


def test_estimate_showdown_equity_is_deterministic_and_uses_board_context() -> None:
    strong_state = state_with(
        hole_cards=cards("Ah", "Ad"),
        board=cards("As", "7h", "2c"),
        legal_actions=(Action(ActionType.CHECK),),
        pot=40,
    )
    weak_state = state_with(
        hole_cards=cards("8h", "3d"),
        board=cards("As", "7h", "2c"),
        legal_actions=(Action(ActionType.CHECK),),
        pot=40,
    )

    strong_equity = estimate_showdown_equity(strong_state, samples=80)

    assert strong_equity == estimate_showdown_equity(strong_state, samples=80)
    assert strong_equity > estimate_showdown_equity(weak_state, samples=80)


def test_explain_decision_includes_showdown_equity_metadata() -> None:
    state = state_with(
        hole_cards=cards("Ah", "Qd"),
        board=cards("Qs", "7h", "2c"),
        legal_actions=(Action(ActionType.CHECK),),
    )

    decision = explain_decision(state)

    assert isinstance(decision.metadata["showdown_equity"], float)


def test_decide_rejects_terminal_state() -> None:
    state = state_with(
        hole_cards=cards("As", "Ah"),
        legal_actions=(),
        current_seat=None,
    )

    with pytest.raises(ValueError, match="terminal"):
        decide(state)
