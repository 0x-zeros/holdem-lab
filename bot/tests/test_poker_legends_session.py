import json
from pathlib import Path
from typing import Any, cast

from holdem_bot.vision import (
    PokerLegendsFrameObservation,
    PokerLegendsSessionTracker,
    build_poker_legends_session_timeline,
)


def test_session_tracker_segments_hands_and_blocked_events() -> None:
    observations = [
        PokerLegendsFrameObservation(
            frame_id="frame_001",
            timestamp_seconds=1.0,
            screen_kind="actionable_table",
            street="preflop",
            blocking_reason=None,
            hero_turn=True,
            hero_cards=("AS", "JS"),
            action_types=("call", "raise", "fold"),
        ),
        PokerLegendsFrameObservation(
            frame_id="frame_002",
            timestamp_seconds=2.0,
            screen_kind="actionable_table",
            street="flop",
            blocking_reason=None,
            hero_turn=True,
            hero_cards=("AS", "JS"),
            board_cards=("7H", "TS", "6H"),
            action_types=("check", "raise", "fold"),
        ),
        PokerLegendsFrameObservation(
            frame_id="frame_003",
            timestamp_seconds=3.0,
            screen_kind="blocked_overlay",
            street="showdown",
            blocking_reason="buy_in_modal",
            hero_turn=False,
            hero_cards=("AS", "JS"),
            board_cards=("7H", "TS", "6H", "QD", "7C"),
        ),
        PokerLegendsFrameObservation(
            frame_id="frame_004",
            timestamp_seconds=4.0,
            screen_kind="actionable_table",
            street="preflop",
            blocking_reason=None,
            hero_turn=True,
            hero_cards=("QH", "6S"),
            action_types=("check", "raise", "fold"),
        ),
    ]

    timeline = PokerLegendsSessionTracker().track(observations)

    assert len(timeline.hands) == 2
    assert timeline.hands[0].hero_cards == ("AS", "JS")
    assert timeline.hands[0].final_board_cards == ("7H", "TS", "6H", "QD", "7C")
    assert timeline.hands[0].streets == ("preflop", "flop", "showdown")
    assert timeline.hands[1].hero_cards == ("QH", "6S")
    assert {event.event_type for event in timeline.events} >= {
        "hand_started",
        "hand_ended",
        "blocked_started",
        "blocked_ended",
        "street_changed",
        "board_changed",
    }


def test_session_tracker_ignores_blocked_showdown_card_changes_for_boundaries() -> None:
    observations = [
        PokerLegendsFrameObservation(
            frame_id="frame_001",
            timestamp_seconds=1.0,
            screen_kind="actionable_table",
            street="flop",
            blocking_reason=None,
            hero_turn=True,
            hero_cards=("AS", "JS"),
            board_cards=("7H", "TS", "6H"),
        ),
        PokerLegendsFrameObservation(
            frame_id="frame_002",
            timestamp_seconds=2.0,
            screen_kind="blocked_overlay",
            street="showdown",
            blocking_reason="other_overlay",
            hero_turn=False,
            hero_cards=("5H", "JC"),
            board_cards=("7C", "8C", "9D", "TC", "JH"),
        ),
        PokerLegendsFrameObservation(
            frame_id="frame_003",
            timestamp_seconds=3.0,
            screen_kind="actionable_table",
            street="preflop",
            blocking_reason=None,
            hero_turn=True,
            hero_cards=("QH", "6S"),
        ),
    ]

    timeline = PokerLegendsSessionTracker().track(observations)

    assert len(timeline.hands) == 2
    assert timeline.hands[0].start_frame_id == "frame_001"
    assert timeline.hands[0].end_frame_id == "frame_002"
    assert timeline.hands[1].start_frame_id == "frame_003"


def test_build_session_timeline_writes_outputs(tmp_path: Path) -> None:
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    selected_manifest = tmp_path / "selected_manifest.json"
    frames: list[tuple[str, float, dict[str, Any]]] = [
        (
            "frame_001",
            1.0,
            {
                "screen": {"kind": "actionable_table", "hero_turn": True},
                "table_state": {"street": "preflop"},
                "hero_hole_cards": [
                    {"slot": "hero_hole_0", "visible": True, "card": "AS"},
                    {"slot": "hero_hole_1", "visible": True, "card": "JS"},
                ],
                "board": [],
                "buttons": [{"visible": True, "action_type": "call"}],
            },
        ),
        (
            "frame_002",
            2.0,
            {
                "screen": {"kind": "blocked_overlay", "blocking_reason": "buy_in_modal"},
                "table_state": {"street": "showdown"},
                "hero_hole_cards": [
                    {"slot": "hero_hole_0", "visible": True, "card": "AS"},
                    {"slot": "hero_hole_1", "visible": True, "card": "JS"},
                ],
                "board": [{"slot": "board_0", "visible": True, "card": "7H"}],
                "buttons": [],
            },
        ),
    ]
    for frame_id, _, truth in frames:
        payload = {"frame_id": frame_id, **truth}
        (truth_dir / f"{frame_id}.json").write_text(json.dumps(payload), encoding="utf-8")
    selected_manifest.write_text(
        json.dumps(
            {
                "frames": [
                    {"frame_id": frame_id, "timestamp_seconds": timestamp}
                    for frame_id, timestamp, _ in frames
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = cast(
        dict[str, Any],
        build_poker_legends_session_timeline(
            sorted(truth_dir.glob("*.json")),
            output_dir=tmp_path / "timeline",
            selection_manifest_path=selected_manifest,
        ),
    )

    assert summary["frames"] == 2
    assert summary["hands"] == 1
    assert (tmp_path / "timeline" / "session_timeline.json").exists()
    assert (tmp_path / "timeline" / "session_timeline.md").exists()
