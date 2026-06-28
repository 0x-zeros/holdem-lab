"""LLM-recognizer assembly tests (offline; the LLM call is mocked)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from holdem_ai.field import FieldExploitPolicy
from holdem_bot.adapters.poker_legends import PokerLegendsTableRecognizer
from holdem_bot.adapters.poker_legends_llm import (
    PokerLegendsLlmRecognizer,
    _downscale_image,
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
    "game_region": None,
    "uncertain": [],
}


def test_runtime_schema_adds_blinds_position_and_region() -> None:
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
    assert "game_region" in props
    required = schema["required"]
    assert isinstance(required, list)
    assert "game_region" in required


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


def test_recognizer_uses_injected_reader_and_exposes_submitted_image(tmp_path: Path) -> None:
    image = tmp_path / "mon2.png"
    cv2.imwrite(str(image), np.zeros((90, 160, 3), dtype=np.uint8))
    recognizer = PokerLegendsLlmRecognizer(
        reader=lambda _image: MON2,
        recognizer=PokerLegendsTableRecognizer.for_llm(controlled_seat=0),
        submitted_path=tmp_path / "submitted.png",
    )
    frame = CapturedFrame(payload=str(image), source="test", metadata={})

    result = recognizer.recognize(frame)
    assert result.state is not None
    assert result.screen.kind.value == "actionable_table"
    assert result.screen.hero_turn is True
    assert result.metadata["submitted_image"] == str(tmp_path / "submitted.png")
    assert (tmp_path / "submitted.png").exists()


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


def test_locator_region_then_crops(tmp_path: Path) -> None:
    frame_file = tmp_path / "f.png"
    cv2.imwrite(str(frame_file), np.zeros((100, 200, 3), dtype=np.uint8))
    shapes: list[tuple[int, int]] = []

    def reader(image: Any) -> dict[str, Any]:
        shapes.append(tuple(image.shape[:2]))
        return MON2

    def locator(_image: Any) -> tuple[float, float, float, float]:
        return (0.25, 0.2, 0.5, 0.6)

    recognizer = PokerLegendsLlmRecognizer(
        reader=reader,
        locator=locator,
        recognizer=PokerLegendsTableRecognizer.for_llm(controlled_seat=0),
        max_edge=0,
        margin=0.0,
        submitted_path=tmp_path / "submitted.png",
        crop=True,
    )
    frame = CapturedFrame(payload=str(frame_file), source="t", metadata={})
    first = recognizer.recognize(frame)
    recognizer.recognize(frame)

    # locate runs on the first frame, so both reads see the cropped box (y 20..80, x 50..150)
    assert shapes == [(60, 100), (60, 100)]
    assert first.metadata["game_region_fraction"] == [0.25, 0.2, 0.5, 0.6]


def test_downscale_caps_longest_edge() -> None:
    out = _downscale_image(np.zeros((1000, 2000, 3), dtype=np.uint8), 1280)
    assert max(out.shape[:2]) == 1280


def test_downscale_leaves_small_image() -> None:
    out = _downscale_image(np.zeros((600, 800, 3), dtype=np.uint8), 1280)
    assert out.shape[:2] == (600, 800)


def test_downscale_zero_disabled() -> None:
    out = _downscale_image(np.zeros((1000, 2000, 3), dtype=np.uint8), 0)
    assert out.shape[:2] == (1000, 2000)


def test_crop_disabled_sends_full_frame(tmp_path: Path) -> None:
    frame_file = tmp_path / "f.png"
    cv2.imwrite(str(frame_file), np.zeros((100, 200, 3), dtype=np.uint8))
    shapes: list[tuple[int, int]] = []
    annotation = {**MON2, "game_region": {"x": 50, "y": 20, "width": 100, "height": 60}}

    def reader(image: Any) -> dict[str, Any]:
        shapes.append(tuple(image.shape[:2]))
        return annotation

    recognizer = PokerLegendsLlmRecognizer(
        reader=reader,
        recognizer=PokerLegendsTableRecognizer.for_llm(controlled_seat=0),
        max_edge=0,
        submitted_path=tmp_path / "s.png",
        crop=False,
    )
    frame = CapturedFrame(payload=str(frame_file), source="t", metadata={})
    result = recognizer.recognize(frame)
    recognizer.recognize(frame)

    assert shapes == [(100, 200), (100, 200)]  # never cropped
    assert result.metadata["game_region_fraction"] is None
