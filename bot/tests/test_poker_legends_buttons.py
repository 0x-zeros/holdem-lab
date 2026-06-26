import json
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from holdem_bot.vision import (
    PokerLegendsButtonRecognizer,
    PokerLegendsButtonTemplateManifest,
    build_and_evaluate_poker_legends_button_templates,
    build_poker_legends_button_template_library,
)


def test_build_and_evaluate_button_templates_on_synthetic_frames(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "annotations"
    image_root = tmp_path / "images"
    annotation_dir.mkdir()
    image_root.mkdir()
    truth_paths = []
    for index, action_type in enumerate(("check", "check", "call", "call"), start=1):
        frame_id = f"frame_{index:03d}"
        write_synthetic_button_frame(
            image_root / f"{frame_id}.png",
            annotation_dir / f"{frame_id}.json",
            left_label=action_type.title(),
        )
        truth_path = tmp_path / f"{frame_id}_truth.json"
        truth_path.write_text(
            json.dumps(
                {
                    "frame_id": frame_id,
                    "screen": {"kind": "actionable_table"},
                    "buttons": [
                        {
                            "name": "primary_left",
                            "visible": True,
                            "action_type": action_type,
                            "label": action_type.title(),
                        },
                        {
                            "name": "primary_middle",
                            "visible": True,
                            "action_type": "raise",
                            "label": "Raise",
                        },
                        {
                            "name": "primary_right",
                            "visible": True,
                            "action_type": "fold",
                            "label": "Fold",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        truth_paths.append(truth_path)

    summary = cast(
        dict[str, Any],
        build_and_evaluate_poker_legends_button_templates(
            truth_paths,
            annotation_dir=annotation_dir,
            image_root=image_root,
            output_dir=tmp_path / "buttons",
        ),
    )

    assert summary["templates"] == 4
    assert summary["action_counts"] == {"call": 2, "check": 2}
    assert summary["self_eval"]["accuracy"] == 1.0
    assert summary["leave_frame_eval"]["accuracy"] == 1.0

    recognizer = PokerLegendsButtonRecognizer.from_manifest(
        tmp_path / "buttons" / "button_template_manifest.json"
    )
    annotation = json.loads((annotation_dir / "frame_003.json").read_text(encoding="utf-8"))
    predictions = recognizer.recognize(
        image_root / "frame_003.png",
        annotation,
        frame_id="frame_003",
    )
    observed = {prediction.slot: prediction.action_type for prediction in predictions}
    assert observed == {
        "primary_left": "call",
        "primary_middle": "raise",
        "primary_right": "fold",
    }


def test_button_template_library_skips_non_left_buttons(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "annotations"
    image_root = tmp_path / "images"
    annotation_dir.mkdir()
    image_root.mkdir()
    write_synthetic_button_frame(
        image_root / "frame_001.png",
        annotation_dir / "frame_001.json",
        left_label="Check",
    )
    truth_path = tmp_path / "frame_001_truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "frame_id": "frame_001",
                "screen": {"kind": "blocked_overlay"},
                "buttons": [
                    {
                        "name": "primary_left",
                        "visible": True,
                        "action_type": "check",
                        "label": "Check",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = build_poker_legends_button_template_library(
        [truth_path],
        annotation_dir=annotation_dir,
        image_root=image_root,
        output_dir=tmp_path / "buttons",
    )

    assert manifest.templates == ()
    reloaded = PokerLegendsButtonTemplateManifest.read_json(
        tmp_path / "buttons" / "button_template_manifest.json"
    )
    assert reloaded.templates == ()


def write_synthetic_button_frame(
    image_path: Path,
    annotation_path: Path,
    *,
    left_label: str,
) -> None:
    image = np.full((220, 420, 3), 32, dtype=np.uint8)
    rects = {
        "primary_left": {"x": 20, "y": 50, "width": 110, "height": 124},
        "primary_middle": {"x": 155, "y": 50, "width": 110, "height": 124},
        "primary_right": {"x": 290, "y": 50, "width": 110, "height": 124},
    }
    for name, rect in rects.items():
        roi = image[rect["y"] : rect["y"] + rect["height"], rect["x"] : rect["x"] + rect["width"]]
        roi[:, ::2] = 20
        roi[:, 1::2] = 220
        label = left_label if name == "primary_left" else name.split("_")[1].title()
        cv2.putText(
            roi,
            label,
            (4, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(image_path), image)
    annotation_path.write_text(
        json.dumps(
            {
                "image": image_path.name,
                "regions": {
                    "buttons": [
                        {"name": name, "kind": "button", "rect": rect}
                        for name, rect in rects.items()
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
