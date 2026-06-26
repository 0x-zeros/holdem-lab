import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from holdem_bot.vision import (
    PokerLegendsCardReviewSource,
    PokerLegendsExcludedTruth,
    build_and_evaluate_poker_legends_card_part_templates,
    build_and_evaluate_poker_legends_card_templates,
    poker_legends_layout_regions,
    select_poker_legends_card_review_candidates,
)


def test_card_review_candidates_prioritize_unseen_target_card(tmp_path: Path) -> None:
    template_root = tmp_path / "templates"
    truth_paths = []
    for card in ("AS", "AH", "KS"):
        truth_paths.append(write_template_frame(template_root, card=card))
    card_summary = cast(
        dict[str, Any],
        build_and_evaluate_poker_legends_card_templates(
            truth_paths,
            annotation_dir=template_root / "annotations",
            image_root=template_root / "images",
            output_dir=tmp_path / "card_templates",
        ),
    )
    part_summary = cast(
        dict[str, Any],
        build_and_evaluate_poker_legends_card_part_templates(
            truth_paths,
            annotation_dir=template_root / "annotations",
            image_root=template_root / "images",
            output_dir=tmp_path / "part_templates",
            rank_max_distance=0.02,
            suit_max_distance=0.02,
        ),
    )
    assert card_summary["unique_card_codes"] == ["AH", "AS", "KS"]
    assert part_summary["rank_labels"] == ["A", "K"]
    assert part_summary["suit_labels"] == ["H", "S"]

    source_root = tmp_path / "source"
    write_source_ingest(source_root, cards=("AS", "KH"))
    exclude_dir = tmp_path / "exclude_truth"
    exclude_dir.mkdir()
    (exclude_dir / "frame_001.json").write_text(
        json.dumps({"frame_id": "frame_001"}) + "\n",
        encoding="utf-8",
    )

    summary = select_poker_legends_card_review_candidates(
        [
            PokerLegendsCardReviewSource(
                name="session-test",
                manifest=source_root / "manifest.json",
            )
        ],
        output_dir=tmp_path / "selection",
        card_manifest=tmp_path / "card_templates" / "card_template_manifest.json",
        card_part_manifest=tmp_path / "part_templates" / "card_part_template_manifest.json",
        exclude_truths=[PokerLegendsExcludedTruth("session-test", exclude_dir)],
        target_cards=["KH"],
        max_candidates=5,
    )

    selected_frames = summary["selected_frames"]
    assert isinstance(selected_frames, int)
    assert selected_frames >= 1
    manifest = cast(
        dict[str, Any],
        json.loads(
            (tmp_path / "selection" / "card_review_candidate_manifest.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    frames = cast(list[Mapping[str, object]], manifest["frames"])
    frame_ids = [str(frame["frame_id"]) for frame in frames]
    assert "session-test__frame_002" in frame_ids
    assert "session-test__frame_001" not in frame_ids

    selected_annotation = json.loads(
        (tmp_path / "selection" / "annotations" / "session-test__frame_002.json").read_text(
            encoding="utf-8"
        )
    )
    assert selected_annotation["image"] == "frames/session-test__frame_002.png"
    assert selected_annotation["regions"]["cards"][0]["name"] == "hero_hole_0"


def write_template_frame(root: Path, *, card: str) -> Path:
    image_root = root / "images"
    annotation_dir = root / "annotations"
    truth_dir = root / "truth"
    image_root.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    frame_id = f"template_{card}"
    image_path = image_root / f"{frame_id}.png"
    annotation_path = annotation_dir / f"{frame_id}.json"
    write_card_frame(image_path, card=card)
    annotation_path.write_text(
        json.dumps(
            {
                "image": image_path.name,
                "width": 1600,
                "height": 982,
                "regions": poker_legends_layout_regions(1600, 982),
            }
        ),
        encoding="utf-8",
    )
    truth_path = truth_dir / f"{frame_id}.json"
    truth_path.write_text(
        json.dumps(
            {
                "frame_id": frame_id,
                "screen": {"kind": "table_observe"},
                "hero_hole_cards": [
                    {
                        "slot": "hero_hole_0",
                        "visible": True,
                        "card": card,
                        "confidence": 1.0,
                    }
                ],
                "board": [],
                "ignored_fields": [],
            }
        ),
        encoding="utf-8",
    )
    return truth_path


def write_source_ingest(root: Path, *, cards: tuple[str, ...]) -> None:
    frames_dir = root / "frames"
    annotations_dir = root / "annotations"
    frames_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)
    frames = []
    for index, card in enumerate(cards, start=1):
        frame_id = f"frame_{index:03d}"
        image_path = frames_dir / f"{frame_id}.png"
        annotation_path = annotations_dir / f"{frame_id}.json"
        write_card_frame(image_path, card=card)
        annotation_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "image": f"frames/{frame_id}.png",
                    "width": 1600,
                    "height": 982,
                    "regions": {"board": [], "buttons": [], "cards": [], "texts": []},
                }
            ),
            encoding="utf-8",
        )
        frames.append(
            {
                "image": f"frames/{frame_id}.png",
                "annotation": f"annotations/{frame_id}.json",
                "frame_index": index,
                "timestamp_seconds": float(index * 10),
                "mean_abs_diff": None,
                "reason": "synthetic",
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metadata": {
                    "source": "synthetic.mov",
                    "fps": 30.0,
                    "frame_count": len(frames),
                    "width": 1600,
                    "height": 982,
                    "duration_seconds": 20.0,
                },
                "sample_fps": 1.0,
                "diff_threshold": 0.0,
                "max_gap_seconds": 5.0,
                "resize_width": 1600,
                "contact_sheet": None,
                "contact_sheets": [],
                "process_report": "process_report.md",
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )


def write_card_frame(path: Path, *, card: str) -> None:
    image = np.full((982, 1600, 3), 80, dtype=np.uint8)
    regions = poker_legends_layout_regions(1600, 982)
    rect = cast(Mapping[str, object], regions["cards"][0]["rect"])
    x = int_value(rect["x"])
    y = int_value(rect["y"])
    width = int_value(rect["width"])
    height = int_value(rect["height"])
    image[y : y + height, x : x + width] = 245
    color = (0, 0, 180) if card[1] == "H" else (0, 0, 0)
    cv2.putText(
        image,
        card[0],
        (x + 12, y + 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        color,
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        card[1],
        (x + 15, y + 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        color,
        4,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), image)


def int_value(value: object) -> int:
    if type(value) is int:
        return value
    raise TypeError(f"expected int: {value!r}")
