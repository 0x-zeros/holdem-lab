import json
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from holdem_bot.vision import (
    PokerLegendsCardTemplateManifest,
    PokerLegendsCardTemplateRecognizer,
    build_and_evaluate_poker_legends_card_templates,
    build_poker_legends_card_template_library,
)


def test_build_and_evaluate_card_templates_on_synthetic_frames(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "annotations"
    image_root = tmp_path / "images"
    annotation_dir.mkdir()
    image_root.mkdir()
    truth_paths = []
    for frame_id in ("frame_001", "frame_002"):
        write_synthetic_card_frame(
            image_root / f"{frame_id}.png",
            annotation_dir / f"{frame_id}.json",
            card_text="AS",
        )
        truth_path = tmp_path / f"{frame_id}_truth.json"
        truth_path.write_text(
            json.dumps(
                {
                    "frame_id": frame_id,
                    "screen": {"kind": "actionable_table"},
                    "hero_hole_cards": [
                        {
                            "slot": "hero_hole_0",
                            "visible": True,
                            "card": "AS",
                            "confidence": 1.0,
                        }
                    ],
                    "board": [],
                    "ignored_fields": [],
                }
            ),
            encoding="utf-8",
        )
        truth_paths.append(truth_path)

    summary = cast(
        dict[str, Any],
        build_and_evaluate_poker_legends_card_templates(
            truth_paths,
            annotation_dir=annotation_dir,
            image_root=image_root,
            output_dir=tmp_path / "cards",
        ),
    )

    assert summary["templates"] == 2
    assert summary["unique_cards"] == 1
    assert summary["self_eval"]["visible_accuracy"] == 1.0
    assert summary["leave_frame_eval"]["visible_accuracy"] == 1.0
    assert (tmp_path / "cards" / "card_template_manifest.json").exists()
    assert (tmp_path / "cards" / "card_template_report.md").exists()

    recognizer = PokerLegendsCardTemplateRecognizer.from_manifest(
        tmp_path / "cards" / "card_template_manifest.json"
    )
    annotation = json.loads((annotation_dir / "frame_001.json").read_text(encoding="utf-8"))
    predictions = recognizer.recognize(
        image_root / "frame_001.png",
        annotation,
        frame_id="frame_001",
    )
    assert predictions[0].card == "AS"
    assert predictions[0].visible


def test_card_template_library_skips_blocked_overlay_by_default(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "annotations"
    image_root = tmp_path / "images"
    annotation_dir.mkdir()
    image_root.mkdir()
    write_synthetic_card_frame(
        image_root / "frame_001.png",
        annotation_dir / "frame_001.json",
        card_text="AS",
    )
    truth_path = tmp_path / "frame_001_truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "frame_id": "frame_001",
                "screen": {"kind": "blocked_overlay"},
                "hero_hole_cards": [
                    {
                        "slot": "hero_hole_0",
                        "visible": True,
                        "card": "AS",
                        "confidence": 1.0,
                    }
                ],
                "board": [],
                "ignored_fields": [],
            }
        ),
        encoding="utf-8",
    )

    manifest = build_poker_legends_card_template_library(
        [truth_path],
        annotation_dir=annotation_dir,
        image_root=image_root,
        output_dir=tmp_path / "cards",
    )

    assert manifest.templates == ()
    reloaded = PokerLegendsCardTemplateManifest.read_json(
        tmp_path / "cards" / "card_template_manifest.json"
    )
    assert reloaded.templates == ()


def write_synthetic_card_frame(
    image_path: Path,
    annotation_path: Path,
    *,
    card_text: str,
) -> None:
    image = np.full((160, 160, 3), 32, dtype=np.uint8)
    image[20:132, 30:114] = 245
    cv2.putText(
        image,
        card_text,
        (42, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(image_path), image)
    annotation_path.write_text(
        json.dumps(
            {
                "image": image_path.name,
                "regions": {
                    "cards": [
                        {
                            "name": "hero_hole_0",
                            "kind": "card",
                            "rect": {"x": 30, "y": 20, "width": 84, "height": 112},
                        }
                    ],
                    "board": [],
                },
            }
        ),
        encoding="utf-8",
    )
