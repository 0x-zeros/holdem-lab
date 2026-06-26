from holdem_bot import CapturedFrame, ScreenKind
from holdem_bot.adapters import PokerLegendsScreenStateRecognizer


def test_poker_legends_screen_state_recognizer_reads_truth_overlay() -> None:
    recognizer = PokerLegendsScreenStateRecognizer()
    truth = {
        "frame_id": "keyframe_000124",
        "screen": {
            "kind": "actionable_table",
            "confidence": 0.95,
            "reason": "Hero can act.",
            "blocking_reason": None,
            "hero_turn": True,
        },
    }

    result = recognizer.recognize(CapturedFrame(payload=truth, source="truth"))

    assert result.state is None
    assert result.confidence == 0.95
    assert result.screen.kind is ScreenKind.ACTIONABLE_TABLE
    assert result.screen.hero_turn is True
    assert result.metadata["frame_id"] == "keyframe_000124"


def test_poker_legends_screen_state_recognizer_classifies_raw_candidate_overlay() -> None:
    recognizer = PokerLegendsScreenStateRecognizer()
    candidate = {
        "frame_id": "keyframe_000151",
        "table_state": {
            "is_table": True,
            "is_actionable": False,
            "street": "unknown",
            "blocking_reason": "leave_table_modal",
            "summary": "Leave table?",
            "confidence": 0.9,
        },
        "buttons": [],
        "seats": [],
    }

    result = recognizer.recognize(CapturedFrame(payload=candidate, source="candidate"))

    assert result.state is None
    assert result.screen.kind is ScreenKind.BLOCKED_OVERLAY
    assert result.screen.blocking_reason == "leave_table_modal"


def test_poker_legends_screen_state_recognizer_fails_closed_on_unknown_payload() -> None:
    recognizer = PokerLegendsScreenStateRecognizer()

    result = recognizer.recognize(CapturedFrame(payload=object(), source="image"))

    assert result.state is None
    assert result.confidence == 0.0
    assert result.screen.kind is ScreenKind.UNKNOWN_OR_TRANSITION
