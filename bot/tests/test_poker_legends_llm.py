"""LLM-recognizer assembly tests (offline; the LLM call is mocked)."""

from __future__ import annotations

from typing import Any

from holdem_ai.field import FieldExploitPolicy
from holdem_bot.adapters.poker_legends import PokerLegendsTableRecognizer
from holdem_bot.adapters.poker_legends_llm import (
    PokerLegendsLlmRecognizer,
    runtime_annotation_schema,
)
from holdem_bot.capture import CapturedFrame


def _seat(
    name: str, stack: int, committed: int, current: bool, position: str | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "visible": True,
        "stack": stack,
        "committed": committed,
        "current": current,
        "active": True,
        "confidence": 0.95,
        "position": position,
    }


def _button(name: str, label: str, action_type: str) -> dict[str, Any]:
    return {
        "name": name,
        "visible": True,
        "label": label,
        "action_type": action_type,
        "confidence": 0.98,
    }


# What Gemini should return for the real mon2 frame (hero 8S 2D, hero to act preflop).
MON2: dict[str, Any] = {
    "schema_version": 1,
    "frame_id": "mon2",
    "table_state": {
        "is_table": True,
        "is_actionable": True,
        "street": "preflop",
        "blocking_reason": "none",
        "summary": "hero to act preflop",
        "confidence": 0.96,
        "small_blind": 5,
        "big_blind": 10,
    },
    "hero_hole_cards": [
        {"slot": "hero_hole_0", "visible": True, "card": "8S", "confidence": 0.99},
        {"slot": "hero_hole_1", "visible": True, "card": "2D", "confidence": 0.99},
    ],
    "board": [],
    "buttons": [
        _button("call", "Call $10", "call"),
        _button("raise", "Raise", "raise"),
        _button("fold", "Fold", "fold"),
    ],
    "texts": [
        {
            "name": "pot",
            "visible": True,
            "value": "$25",
            "normalized_number": 25,
            "confidence": 0.9,
        },
    ],
    "seats": [
        _seat("hero", 1000, 0, True),
        _seat("sb", 2795, 5, False, "sb"),
        _seat("bb", 990, 10, False, "bb"),
        _seat("skl", 7992, 10, False),
    ],
    "uncertain": [],
}


def test_runtime_schema_adds_blinds_and_position() -> None:
    schema = runtime_annotation_schema()
    props = schema["properties"]
    assert isinstance(props, dict)
    table_state = props["table_state"]
    assert isinstance(table_state, dict)
    assert "small_blind" in table_state["properties"]
    assert "big_blind" in table_state["required"]
    seat_items = props["seats"]["items"]
    assert "position" in seat_items["properties"]
    assert "position" in seat_items["required"]


def test_llm_annotation_assembles_state_and_ai_folds_trash() -> None:
    recognizer = PokerLegendsTableRecognizer.for_llm(controlled_seat=0)
    result = recognizer.recognize_from_llm_annotation(MON2, image="mon2.png", frame_id="mon2")

    state = result.state
    assert state is not None
    assert state.street.value == "preflop"
    assert state.to_call == 10
    assert state.pots[0].amount == 25
    legal = {action.action_type.value for action in state.legal_actions}
    assert legal == {"call", "raise", "fold"}
    assert [card.rank.value for card in state.players[0].hole_cards] == ["8", "2"]
    assert len(state.players) == 4

    decision = FieldExploitPolicy().explain(state)
    # 82o is the worst starting hand; the field-exploit policy should fold it preflop.
    assert decision.action.action_type.value == "fold"


def test_recognizer_uses_injected_reader(tmp_path: Any) -> None:
    image = tmp_path / "mon2.png"
    image.write_bytes(b"not-a-real-png")
    recognizer = PokerLegendsLlmRecognizer(
        reader=lambda _path: MON2,
        recognizer=PokerLegendsTableRecognizer.for_llm(controlled_seat=0),
    )
    frame = CapturedFrame(payload=str(image), source="test", metadata={})

    result = recognizer.recognize(frame)
    assert result.state is not None
    assert result.screen.kind.value == "actionable_table"
    assert result.screen.hero_turn is True


def test_blocking_overlay_fails_closed() -> None:
    blocked = {
        **MON2,
        "table_state": {
            **MON2["table_state"],
            "is_actionable": False,
            "blocking_reason": "buy_in_modal",
        },
    }
    recognizer = PokerLegendsTableRecognizer.for_llm(controlled_seat=0)
    result = recognizer.recognize_from_llm_annotation(blocked)
    assert result.state is None
    assert result.screen.kind.value == "blocked_overlay"
