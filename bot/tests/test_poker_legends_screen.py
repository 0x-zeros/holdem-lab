import json
from pathlib import Path

import cv2
import numpy as np
from holdem_bot import ScreenKind
from holdem_bot.vision import (
    detect_poker_legends_screen_state_from_image,
    evaluate_poker_legends_screen_state,
    poker_legends_layout_regions,
)


def test_detect_poker_legends_screen_actionable_from_button_cluster() -> None:
    image = table_image()
    regions = poker_legends_layout_regions(1600, 982)
    draw_primary_action_buttons(image, regions)

    detection = detect_poker_legends_screen_state_from_image(image, regions=regions)

    assert detection.screen.kind is ScreenKind.ACTIONABLE_TABLE
    assert detection.screen.hero_turn is True
    assert detection.active_primary_buttons == 3


def test_detect_poker_legends_screen_blocks_overlay_before_buttons() -> None:
    image = table_image()
    regions = poker_legends_layout_regions(1600, 982)
    draw_primary_action_buttons(image, regions)
    fill_region(image, regions, "overlays", "left_panel", 30)

    detection = detect_poker_legends_screen_state_from_image(image, regions=regions)

    assert detection.screen.kind is ScreenKind.BLOCKED_OVERLAY
    assert "left_panel_low_mean" in detection.overlay_signals


def test_evaluate_poker_legends_screen_state_writes_report(tmp_path: Path) -> None:
    regions = poker_legends_layout_regions(1600, 982)
    image = table_image()
    draw_primary_action_buttons(image, regions)
    image_path = tmp_path / "frame.png"
    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    (annotation_dir / "frame.json").write_text(
        json.dumps({"image": "frame.png", "regions": regions}),
        encoding="utf-8",
    )
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    truth_path = truth_dir / "frame.json"
    truth_path.write_text(
        json.dumps(
            {
                "frame_id": "frame",
                "screen": {
                    "kind": "actionable_table",
                    "confidence": 0.95,
                    "reason": "synthetic",
                    "blocking_reason": None,
                    "hero_turn": True,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = evaluate_poker_legends_screen_state(
        [truth_path],
        annotation_dir=annotation_dir,
        image_root=tmp_path,
        output_dir=tmp_path / "screen_eval",
    )

    assert summary["accuracy"] == 1.0
    assert (tmp_path / "screen_eval" / "screen_state_report.md").exists()


def table_image() -> np.ndarray:
    return np.full((982, 1600, 3), 80, dtype=np.uint8)


def draw_primary_action_buttons(
    image: np.ndarray,
    regions: dict[str, list[dict[str, object]]],
) -> None:
    for name in ("primary_left", "primary_middle", "primary_right"):
        rect = region_rect(regions, "buttons", name)
        roi = image[rect["y"] : rect["y"] + rect["height"], rect["x"] : rect["x"] + rect["width"]]
        roi[:, ::2] = 20
        roi[:, 1::2] = 220


def fill_region(
    image: np.ndarray,
    regions: dict[str, list[dict[str, object]]],
    group: str,
    name: str,
    value: int,
) -> None:
    rect = region_rect(regions, group, name)
    image[
        rect["y"] : rect["y"] + rect["height"],
        rect["x"] : rect["x"] + rect["width"],
    ] = value


def region_rect(
    regions: dict[str, list[dict[str, object]]],
    group: str,
    name: str,
) -> dict[str, int]:
    for region in regions[group]:
        if region["name"] == name:
            return region["rect"]  # type: ignore[return-value]
    raise AssertionError(f"missing region: {group}.{name}")
