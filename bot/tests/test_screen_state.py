import pytest
from holdem_bot import ScreenKind, ScreenState, evaluate_safety
from holdem_common import Action, ActionType, Card, GameState, PlayerState, Pot, Street


def cards(*codes: str) -> tuple[Card, ...]:
    return tuple(Card.from_code(code) for code in codes)


def make_state(
    *,
    current_seat: int | None = 0,
    legal_actions: tuple[Action, ...] = (Action(ActionType.CHECK),),
) -> GameState:
    return GameState(
        hand_id="safety-test",
        street=Street.FLOP,
        players=(
            PlayerState(seat=0, stack=100, hole_cards=cards("Ah", "Kh")),
            PlayerState(seat=1, stack=100, hole_cards=()),
        ),
        board=cards("2c", "7d", "Ts"),
        pots=(Pot(amount=10, eligible_seats=frozenset({0, 1})),),
        current_seat=current_seat,
        button_seat=0,
        small_blind=1,
        big_blind=2,
        min_raise=4,
        to_call=0,
        legal_actions=legal_actions,
    )


def test_evaluate_safety_allows_actionable_hero_turn() -> None:
    state = make_state()

    decision = evaluate_safety(
        screen=ScreenState.actionable_table(hero_turn=True),
        state=state,
        recognition_confidence=0.95,
        controlled_seat=0,
        min_confidence=0.8,
    )

    assert decision.allowed
    assert decision.reason == "safe_to_act"
    assert decision.state == state
    assert decision.screen.kind is ScreenKind.ACTIONABLE_TABLE


@pytest.mark.parametrize(
    ("screen", "reason"),
    [
        (ScreenState.blocked_overlay(blocking_reason="leave_table_modal"), "blocked_overlay"),
        (ScreenState.non_table_ui(reason="lobby"), "non_table_ui"),
        (ScreenState.unknown_or_transition(confidence=0.9), "unknown_or_transition"),
        (ScreenState.table_observe(reason="showdown"), "table_observe"),
    ],
)
def test_evaluate_safety_blocks_non_actionable_screens(
    screen: ScreenState,
    reason: str,
) -> None:
    decision = evaluate_safety(
        screen=screen,
        state=make_state(),
        recognition_confidence=0.95,
        controlled_seat=0,
        min_confidence=0.8,
    )

    assert not decision.allowed
    assert decision.reason == reason


def test_evaluate_safety_blocks_low_screen_confidence() -> None:
    decision = evaluate_safety(
        screen=ScreenState.actionable_table(confidence=0.5, hero_turn=True),
        state=make_state(),
        recognition_confidence=0.95,
        controlled_seat=0,
        min_confidence=0.8,
    )

    assert not decision.allowed
    assert decision.reason == "screen_low_confidence"


def test_evaluate_safety_blocks_hero_turn_false() -> None:
    decision = evaluate_safety(
        screen=ScreenState.actionable_table(hero_turn=False),
        state=make_state(),
        recognition_confidence=0.95,
        controlled_seat=0,
        min_confidence=0.8,
    )

    assert not decision.allowed
    assert decision.reason == "waiting"


def test_screen_state_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        ScreenState.actionable_table(confidence=1.5)
