import pytest
from holdem_ai import decide, estimate_private_strength, explain_decision
from holdem_common import Action, ActionType, Card, GameState, PlayerState, Pot, Street


def cards(*codes: str) -> tuple[Card, ...]:
    return tuple(Card.from_code(code) for code in codes)


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
    assert decision.reason == "check"
    assert decision.required_equity is None
    assert decision.metadata["made_hand"] == "high_card"
    assert decision.metadata["active_opponents"] == 1


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


def test_decide_rejects_terminal_state() -> None:
    state = state_with(
        hole_cards=cards("As", "Ah"),
        legal_actions=(),
        current_seat=None,
    )

    with pytest.raises(ValueError, match="terminal"):
        decide(state)
