import json
from pathlib import Path

import pytest
from holdem_bot.adapters import (
    MacOSScreenCapture,
    PokerLegendsCaptureMetadata,
    PokerLegendsDryRunAutomator,
    PokerLegendsImageCapture,
    PokerLegendsLayoutClickPlanner,
)
from holdem_bot.adapters.poker_legends_host import plan_click_main
from holdem_common import Action, ActionType, Card, GameState, PlayerState, Pot, Street


def test_macos_screen_capture_builds_screencapture_command(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    capture = MacOSScreenCapture(
        output_dir=tmp_path,
        window_id=42,
        runner=lambda command: commands.append(list(command)),
        now=lambda: 123.456,
        system_name=lambda: "Darwin",
    )

    frame = capture.capture()

    assert commands == [["screencapture", "-x", "-l", "42", str(tmp_path / "frame_123456.png")]]
    assert frame.payload == tmp_path / "frame_123456.png"
    assert frame.metadata["poker_legends_image_path"] == str(tmp_path / "frame_123456.png")
    assert frame.metadata["coordinate_space"] == "image"


def test_macos_screen_capture_rejects_non_macos(tmp_path: Path) -> None:
    capture = MacOSScreenCapture(
        output_dir=tmp_path,
        runner=lambda _command: None,
        system_name=lambda: "Linux",
    )

    with pytest.raises(RuntimeError, match="macOS"):
        capture.capture()


def test_poker_legends_image_capture_adds_recognizer_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    layout_path = tmp_path / "layout.json"
    annotation_path = tmp_path / "annotation.json"
    capture = PokerLegendsImageCapture(
        image_path=image_path,
        layout_annotation_path=layout_path,
        annotation_path=annotation_path,
    )

    frame = capture.capture()

    assert frame.payload == image_path
    assert frame.source == "poker_legends_image"
    assert frame.metadata["poker_legends_image_path"] == str(image_path)
    assert frame.metadata["poker_legends_layout_annotation_path"] == str(layout_path)
    assert frame.metadata["poker_legends_annotation_path"] == str(annotation_path)
    assert frame.metadata["coordinate_space"] == "image"


def test_poker_legends_capture_metadata_wraps_macos_capture(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    macos_capture = MacOSScreenCapture(
        output_dir=tmp_path,
        runner=lambda command: commands.append(list(command)),
        now=lambda: 1.0,
        system_name=lambda: "Darwin",
    )
    capture = PokerLegendsCaptureMetadata(
        capture=macos_capture,
        layout_annotation_path=tmp_path / "layout.json",
        annotation_path=tmp_path / "annotation.json",
    )

    frame = capture.capture()

    assert commands == [["screencapture", "-x", str(tmp_path / "frame_1000.png")]]
    assert frame.payload == tmp_path / "frame_1000.png"
    assert frame.metadata["poker_legends_layout_annotation_path"] == str(tmp_path / "layout.json")
    assert frame.metadata["poker_legends_annotation_path"] == str(tmp_path / "annotation.json")


def test_poker_legends_layout_click_planner_maps_primary_buttons() -> None:
    planner = PokerLegendsLayoutClickPlanner(layout_annotation())

    assert planner.plan(Action(ActionType.CHECK)).to_dict() == {
        "action_type": "check",
        "amount": 0,
        "command": "primary_left",
        "coordinate_space": "image",
        "x": 1165,
        "y": 838,
    }
    assert planner.plan(Action(ActionType.RAISE, amount=40)).command == "primary_middle"
    assert planner.plan(Action(ActionType.FOLD)).command == "primary_right"


def test_poker_legends_dry_run_automator_writes_jsonl(tmp_path: Path) -> None:
    planner = PokerLegendsLayoutClickPlanner(layout_annotation())
    log_path = tmp_path / "dry_run.jsonl"
    automator = PokerLegendsDryRunAutomator(
        planner=planner,
        log_path=log_path,
        now=lambda: 123.0,
    )

    automator.perform(Action(ActionType.CALL, amount=25), state())

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["mode"] == "dry_run"
    assert record["executed"] is False
    assert record["action"] == {
        "amount": 25,
        "max_amount": None,
        "min_amount": None,
        "type": "call",
    }
    assert record["click_plan"]["command"] == "primary_left"
    assert record["click_plan"]["x"] == 1165
    assert automator.records == [record]


def test_poker_legends_click_planner_rejects_missing_button_region() -> None:
    planner = PokerLegendsLayoutClickPlanner({"regions": {"buttons": []}})

    with pytest.raises(KeyError, match="primary_left"):
        planner.plan(Action(ActionType.CHECK))


def test_plan_click_main_writes_dry_run_record(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps(layout_annotation()), encoding="utf-8")
    log_path = tmp_path / "click_plan.jsonl"

    plan_click_main(
        [
            "--layout-annotation",
            str(layout_path),
            "--action",
            "raise",
            "--amount",
            "40",
            "--out-jsonl",
            str(log_path),
        ]
    )

    stdout_record = json.loads(capsys.readouterr().out)
    file_record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert stdout_record == file_record
    assert file_record["mode"] == "dry_run_click_plan"
    assert file_record["executed"] is False
    assert file_record["action"]["type"] == "raise"
    assert file_record["click_plan"]["command"] == "primary_middle"
    assert file_record["click_plan"]["x"] == 1323


def layout_annotation() -> dict[str, object]:
    return {
        "regions": {
            "buttons": [
                {
                    "name": "primary_left",
                    "rect": {"x": 1110, "y": 776, "width": 110, "height": 124},
                },
                {
                    "name": "primary_middle",
                    "rect": {"x": 1268, "y": 776, "width": 110, "height": 124},
                },
                {
                    "name": "primary_right",
                    "rect": {"x": 1425, "y": 776, "width": 110, "height": 124},
                },
            ]
        }
    }


def state() -> GameState:
    return GameState(
        hand_id="dry-run-hand",
        street=Street.FLOP,
        players=(
            PlayerState(
                seat=0,
                stack=100,
                hole_cards=(Card.from_code("As"), Card.from_code("Kd")),
            ),
            PlayerState(seat=1, stack=100),
        ),
        board=(Card.from_code("2c"), Card.from_code("7d"), Card.from_code("Ts")),
        pots=(Pot(amount=50, eligible_seats=frozenset({0, 1})),),
        current_seat=0,
        button_seat=0,
        small_blind=5,
        big_blind=10,
        min_raise=20,
        to_call=25,
        legal_actions=(Action(ActionType.CALL, amount=25), Action(ActionType.FOLD)),
    )
