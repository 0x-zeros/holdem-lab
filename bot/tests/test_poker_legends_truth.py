import json
from pathlib import Path
from typing import Any, cast

from holdem_bot import ScreenKind
from holdem_bot.vision import (
    build_poker_legends_truth_overlay,
    merge_poker_legends_truth,
    screen_state_from_poker_legends_annotation,
)


def test_merge_normalizes_prefixed_candidate_and_applies_hero_override() -> None:
    candidate = {
        "frame_id": "keyframe_000124",
        "schema_version": 1,
        "table_state": {
            "is_table": True,
            "is_actionable": True,
            "street": "flop",
            "blocking_reason": "none",
            "summary": "Hero can act.",
            "confidence": 0.95,
        },
        "hero_hole_cards": [
            {"slot": "cards.hero_hole_0", "visible": True, "card": "AH", "confidence": 1.0},
            {"slot": "cards.hero_hole_1", "visible": True, "card": "9H", "confidence": 1.0},
        ],
        "board": [
            {"slot": "board.board_0", "visible": True, "card": "8D", "confidence": 1.0},
            {"slot": "board.board_1", "visible": True, "card": "QH", "confidence": 1.0},
            {"slot": "board.board_2", "visible": True, "card": "JH", "confidence": 1.0},
        ],
        "buttons": [
            {
                "name": "buttons.primary_left",
                "visible": True,
                "label": "Check",
                "action_type": "check",
                "confidence": 1.0,
            }
        ],
        "texts": [
            {
                "name": "texts.hero_stack",
                "visible": True,
                "value": "$1,112+10",
                "normalized_number": 1122,
                "confidence": 0.95,
            }
        ],
        "seats": [{"name": "Grace", "visible": True, "stack": 115, "committed": None}],
        "uncertain": [],
    }
    decision = {
        "decision": "corrected_context",
        "overrides": {
            "hero_seat": "bottom",
            "is_actionable": True,
            "seat_overrides": {"hero": {"current": True, "position": "bottom"}},
        },
    }

    truth = cast(dict[str, Any], merge_poker_legends_truth(candidate, decision=decision))

    assert truth["screen"]["kind"] == ScreenKind.ACTIONABLE_TABLE.value
    assert truth["screen"]["hero_turn"] is True
    assert truth["hero"] == {"seat": "bottom", "current": True}
    assert truth["board"][0]["slot"] == "board_0"
    assert truth["hero_hole_cards"][0]["slot"] == "hero_hole_0"
    assert truth["buttons"][0]["name"] == "primary_left"
    hero_seat = [seat for seat in truth["seats"] if seat["name"] == "hero"][0]
    assert hero_seat["stack"] == 1122
    assert truth["validation"]["status"] == "ok"


def test_merge_applies_blocking_overlay_and_ignored_fields() -> None:
    candidate = {
        "frame_id": "keyframe_000051",
        "schema_version": 1,
        "table_state": {
            "is_table": True,
            "is_actionable": False,
            "street": "flop",
            "blocking_reason": "challenge_overlay",
            "summary": "Daily challenges overlay.",
            "confidence": 0.95,
        },
        "hero_hole_cards": [
            {"slot": "hero_hole_0", "visible": True, "card": "AS", "confidence": 0.5}
        ],
        "board": [],
        "buttons": [],
        "texts": [
            {
                "name": "pot",
                "visible": True,
                "value": "$100",
                "normalized_number": 100,
                "confidence": 0.5,
            }
        ],
        "seats": [],
        "uncertain": [],
    }
    decision = {
        "decision": "confirmed_with_context",
        "overrides": {
            "blocking_reason": "challenge_overlay",
            "ignore_fields": ["pot", "hero_hole_cards"],
            "is_actionable": False,
        },
    }

    truth = cast(dict[str, Any], merge_poker_legends_truth(candidate, decision=decision))

    assert truth["screen"]["kind"] == ScreenKind.BLOCKED_OVERLAY.value
    assert truth["screen"]["blocking_reason"] == "challenge_overlay"
    assert truth["hero_hole_cards"] == []
    assert truth["texts"] == []
    assert truth["ignored_fields"] == ["pot", "hero_hole_cards"]


def test_merge_applies_board_card_override() -> None:
    candidate = {
        "frame_id": "keyframe_000145",
        "table_state": {
            "is_table": True,
            "is_actionable": True,
            "street": "unknown",
            "blocking_reason": "none",
            "summary": "Hero can act.",
            "confidence": 0.95,
        },
        "hero_hole_cards": [
            {"slot": "hero_hole_0", "visible": True, "card": "AC", "confidence": 1.0}
        ],
        "board": [{"slot": "board_2", "visible": True, "card": "9C", "confidence": 0.5}],
        "buttons": [
            {
                "name": "primary_left",
                "visible": True,
                "label": "Check",
                "action_type": "check",
                "confidence": 1.0,
            }
        ],
        "texts": [],
        "seats": [{"name": "hero", "visible": True, "current": True}],
        "uncertain": [],
    }
    decision = {"decision": "confirmed", "overrides": {"board.board_2": "9S"}}

    truth = cast(dict[str, Any], merge_poker_legends_truth(candidate, decision=decision))

    assert truth["board"][0]["card"] == "9S"


def test_screen_state_from_raw_candidate_classifies_table_observe() -> None:
    candidate = {
        "table_state": {
            "is_table": True,
            "is_actionable": False,
            "street": "showdown",
            "blocking_reason": "none",
            "summary": "Showdown.",
            "confidence": 1.0,
        },
        "buttons": [],
        "seats": [],
    }

    screen = screen_state_from_poker_legends_annotation(candidate)

    assert screen.kind is ScreenKind.TABLE_OBSERVE
    assert screen.hero_turn is None


def test_build_poker_legends_truth_overlay_writes_reviewable_outputs(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    candidate_path = candidate_dir / "keyframe_000151.json"
    candidate_path.write_text(
        json.dumps(
            {
                "frame_id": "keyframe_000151",
                "table_state": {
                    "is_table": True,
                    "is_actionable": False,
                    "street": "unknown",
                    "blocking_reason": "leave_table_modal",
                    "summary": "Leave table?",
                    "confidence": 0.9,
                },
                "hero_hole_cards": [],
                "board": [],
                "buttons": [],
                "texts": [],
                "seats": [],
                "uncertain": [],
            }
        ),
        encoding="utf-8",
    )
    decisions_path = tmp_path / "human_review_decisions.json"
    decisions_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "decisions": [
                    {
                        "frame_id": "keyframe_000151",
                        "decision": "confirmed",
                        "overrides": {
                            "blocking_reason": "leave_table_modal",
                            "is_actionable": False,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = build_poker_legends_truth_overlay(
        [candidate_path],
        review_decisions_path=decisions_path,
        output_dir=tmp_path / "truth",
    )

    assert summary["frames"] == 1
    assert summary["human_reviewed_frames"] == 1
    assert (tmp_path / "truth" / "truth_overlays" / "keyframe_000151.json").exists()
    assert (tmp_path / "truth" / "truth_overlay_report.md").exists()
