import json
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from holdem_bot.vision import apply_poker_legends_layout
from holdem_bot.vision.poker_legends_layout import main as layout_main


def write_draft_annotation(path: Path, *, image: str, width: int, height: int) -> None:
    draft = {
        "schema_version": 1,
        "source": "poker_legends_video",
        "image": image,
        "video": "sample.mov",
        "width": width,
        "height": height,
        "source_width": width,
        "source_height": height,
        "frame_index": 0,
        "timestamp_seconds": 0.0,
        "todo": ["hero_hole_cards", "board", "buttons"],
        "metadata": {"existing": True},
        "regions": {"board": [], "buttons": [], "seats": [], "texts": []},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(draft), encoding="utf-8")


def test_apply_poker_legends_layout_scales_regions(tmp_path: Path) -> None:
    annotation_path = tmp_path / "draft.json"
    output_path = tmp_path / "updated.json"
    write_draft_annotation(annotation_path, image="frames/frame.png", width=800, height=491)

    updated = apply_poker_legends_layout(annotation_path, output_path)

    assert updated["metadata"] == {
        "existing": True,
        "layout": {
            "base_height": 982,
            "base_width": 1600,
            "name": "poker_legends_1600w_v1",
            "status": "roi_applied_values_pending",
        },
    }
    regions = cast(dict[str, list[dict[str, object]]], updated["regions"])
    assert set(regions) == {"board", "buttons", "cards", "overlays", "seats", "texts"}
    assert regions["board"][0]["rect"] == {"height": 56, "width": 42, "x": 254, "y": 216}
    assert regions["cards"][0]["name"] == "hero_hole_0"
    assert any(region["name"] == "hero_current_bet" for region in regions["texts"])


def test_apply_poker_legends_layout_writes_overlay(tmp_path: Path) -> None:
    image_root = tmp_path / "selection"
    frames_dir = image_root / "frames"
    annotations_dir = image_root / "annotations"
    overlay_dir = image_root / "layout_overlays"
    image_path = frames_dir / "frame.png"
    annotation_path = annotations_dir / "frame.json"
    frames_dir.mkdir(parents=True)
    blank = np.zeros((491, 800, 3), dtype=np.uint8)
    if not cv2.imwrite(str(image_path), blank):
        raise RuntimeError("could not write blank test image")
    write_draft_annotation(annotation_path, image="frames/frame.png", width=800, height=491)

    apply_poker_legends_layout(
        annotation_path,
        image_root=image_root,
        overlay_dir=overlay_dir,
    )

    assert (overlay_dir / "frame_layout.png").exists()


def test_poker_legends_layout_cli_writes_report(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotations" / "frame.json"
    report_path = tmp_path / "layout_report.md"
    write_draft_annotation(annotation_path, image="frames/frame.png", width=800, height=491)

    layout_main([str(annotation_path), "--report", str(report_path)])

    assert report_path.exists()
    assert "poker_legends_1600w_v1" in report_path.read_text(encoding="utf-8")
