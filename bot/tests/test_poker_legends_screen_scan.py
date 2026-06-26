import json
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from holdem_bot.vision import (
    build_poker_legends_screen_scan,
    poker_legends_layout_regions,
    scan_poker_legends_ingest,
)
from holdem_bot.vision.video import (
    ExtractedFrame,
    VideoIngestManifest,
    VideoMetadata,
)


def test_screen_scan_uses_draft_annotations_and_selects_candidates(tmp_path: Path) -> None:
    manifest_path = write_synthetic_ingest(tmp_path)

    scan = scan_poker_legends_ingest(
        manifest_path,
        max_llm_candidates=12,
        spaced_sample_seconds=1.0,
    )

    assert scan.screen_kind_counts == {
        "table_observe": 2,
        "actionable_table": 2,
        "blocked_overlay": 1,
    }
    assert [run.screen_kind for run in scan.runs] == [
        "table_observe",
        "actionable_table",
        "blocked_overlay",
        "table_observe",
        "actionable_table",
    ]
    assert any(candidate.category == "blocked_overlay" for candidate in scan.llm_candidates)
    assert any(
        candidate.category == "actionable_edge_2_buttons" for candidate in scan.llm_candidates
    )


def test_build_screen_scan_writes_outputs_and_selection(tmp_path: Path) -> None:
    manifest_path = write_synthetic_ingest(tmp_path)

    summary = cast(
        dict[str, Any],
        build_poker_legends_screen_scan(
            manifest_path,
            output_dir=tmp_path / "scan",
            max_llm_candidates=10,
            spaced_sample_seconds=1.0,
            selection_output_dir=tmp_path / "selection",
        ),
    )

    assert summary["frames"] == 5
    assert summary["runs"] == 5
    assert summary["llm_candidates"] == 5
    assert (tmp_path / "scan" / "screen_state_scan.json").exists()
    assert (tmp_path / "scan" / "screen_state_scan.md").exists()
    assert (tmp_path / "selection" / "selected_manifest.json").exists()


def write_synthetic_ingest(root: Path) -> Path:
    frames_dir = root / "frames"
    annotations_dir = root / "annotations"
    frames_dir.mkdir()
    annotations_dir.mkdir()
    metadata = VideoMetadata(
        source="synthetic.mov",
        fps=1.0,
        frame_count=5,
        width=1600,
        height=982,
        duration_seconds=5.0,
    )
    frames: list[ExtractedFrame] = []
    specs = (
        ("observe", 0),
        ("three_buttons", 1),
        ("buy_in", 2),
        ("observe", 3),
        ("two_buttons", 4),
    )
    for index, (kind, timestamp) in enumerate(specs):
        frame_id = f"keyframe_{index:06d}"
        image_rel = f"frames/{frame_id}.png"
        annotation_rel = f"annotations/{frame_id}.json"
        write_synthetic_frame(frames_dir / f"{frame_id}.png", kind)
        (annotations_dir / f"{frame_id}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source": "poker_legends_video",
                    "image": image_rel,
                    "width": 1600,
                    "height": 982,
                    "frame_index": index,
                    "timestamp_seconds": float(timestamp),
                    "regions": {
                        "board": [],
                        "buttons": [],
                        "seats": [],
                        "texts": [],
                    },
                }
            ),
            encoding="utf-8",
        )
        frames.append(
            ExtractedFrame(
                image=image_rel,
                annotation=annotation_rel,
                frame_index=index,
                timestamp_seconds=float(timestamp),
                mean_abs_diff=None if index == 0 else 1.0,
                reason="first" if index == 0 else "changed",
            )
        )
    manifest = VideoIngestManifest(
        schema_version=1,
        metadata=metadata,
        sample_fps=1.0,
        diff_threshold=0.0,
        max_gap_seconds=5.0,
        resize_width=1600,
        contact_sheet=None,
        contact_sheets=(),
        process_report="process_report.md",
        frames=tuple(frames),
    )
    manifest_path = root / "manifest.json"
    manifest.write_json(manifest_path)
    return manifest_path


def write_synthetic_frame(path: Path, kind: str) -> None:
    image = np.full((982, 1600, 3), 80, dtype=np.uint8)
    regions = poker_legends_layout_regions(1600, 982)
    if kind in {"three_buttons", "two_buttons"}:
        names: tuple[str, ...] = ("primary_left", "primary_middle", "primary_right")
        if kind == "two_buttons":
            names = ("primary_middle", "primary_right")
        for name in names:
            rect = region_rect(regions, "buttons", name)
            roi = image[
                rect["y"] : rect["y"] + rect["height"],
                rect["x"] : rect["x"] + rect["width"],
            ]
            roi[:, ::2] = 20
            roi[:, 1::2] = 220
    if kind == "buy_in":
        rect = region_rect(regions, "overlays", "bottom_buy_in_prompt")
        image[
            rect["y"] : rect["y"] + rect["height"],
            rect["x"] : rect["x"] + rect["width"],
        ] = np.array([180, 0, 220], dtype=np.uint8)
    cv2.imwrite(str(path), image)


def region_rect(
    regions: dict[str, list[dict[str, object]]],
    group: str,
    name: str,
) -> dict[str, int]:
    for region in regions[group]:
        if region["name"] == name:
            return region["rect"]  # type: ignore[return-value]
    raise AssertionError(f"missing region: {group}.{name}")
