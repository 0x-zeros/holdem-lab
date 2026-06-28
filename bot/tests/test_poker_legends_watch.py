from holdem_ai.opponents import OpponentModel
from holdem_bot import RecognitionResult, SafetyDecision, ScreenState, evaluate_safety
from holdem_bot.adapters.poker_legends_host import _watch_record, _watch_summary_lines
from holdem_common import Action, ActionType, Card, GameState, PlayerState, Pot, Street


def _decide(recognition: RecognitionResult, *, seat: int = 0) -> SafetyDecision:
    return evaluate_safety(
        screen=recognition.screen,
        state=recognition.state,
        recognition_confidence=recognition.confidence,
        controlled_seat=seat,
        min_confidence=0.80,
    )


def _actionable_state() -> GameState:
    return GameState(
        hand_id="watch-test",
        street=Street.PREFLOP,
        players=(
            PlayerState(seat=0, stack=100, hole_cards=(Card.from_code("ah"), Card.from_code("kd"))),
            PlayerState(seat=1, stack=100),
        ),
        board=(),
        pots=(Pot(amount=3, eligible_seats=frozenset({0, 1})),),
        current_seat=0,
        button_seat=0,
        small_blind=1,
        big_blind=2,
        min_raise=4,
        to_call=0,
        legal_actions=(Action(ActionType.CHECK),),
    )


def test_watch_summary_lines_surface_state_block_reason() -> None:
    # The HUD's headline value over the old dry-run: a no_game_state frame still tells
    # you *why* state assembly failed (here, the recogniser's state_block_reason).
    recognition = RecognitionResult(
        state=None,
        confidence=0.88,
        metadata={"state_block_reason": "missing_table_metadata"},
        screen=ScreenState.actionable_table(hero_turn=True),
    )
    lines = _watch_summary_lines(
        recognition, _decide(recognition), None, OpponentModel(), controlled_seat=0
    )
    text = "\n".join(lines)

    assert "gate no_game_state" in text
    assert "state_block missing_table_metadata" in text
    assert "state <none>" in text


def test_watch_summary_lines_report_state_and_legal_actions() -> None:
    state = _actionable_state()
    recognition = RecognitionResult(
        state=state, confidence=0.95, screen=ScreenState.actionable_table(hero_turn=True)
    )
    lines = _watch_summary_lines(
        recognition, _decide(recognition), None, OpponentModel(), controlled_seat=0
    )
    text = "\n".join(lines)

    assert "screen actionable_table" in text
    assert "pot 3" in text
    assert "legal check" in text


def test_watch_record_has_diagnostic_shape() -> None:
    recognition = RecognitionResult(
        state=None,
        confidence=0.9,
        metadata={"state_block_reason": "missing_pot"},
        screen=ScreenState.blocked_overlay(blocking_reason="buy_in_prompt"),
    )
    record = _watch_record(
        recognition, _decide(recognition), None, OpponentModel(), controlled_seat=0
    )

    assert set(record) >= {
        "gate",
        "confidence",
        "screen",
        "state",
        "state_block_reason",
        "policy_decision",
        "opponent_reads",
    }
    assert record["gate"] == {"allowed": False, "reason": "blocked_overlay"}
    assert record["state"] is None
    assert record["state_block_reason"] == "missing_pot"
