import json
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from holdem_bot.vision import (
    PokerLegendsTemplateSource,
    build_and_evaluate_poker_legends_multi_templates,
)


def test_multi_templates_namespaces_overlapping_frame_ids(tmp_path: Path) -> None:
    source_a = tmp_path / "source_a"
    source_b = tmp_path / "source_b"
    write_source_frame(source_a, card="AS", left_action="check")
    write_source_frame(source_b, card="KH", left_action="call")

    summary = cast(
        dict[str, Any],
        build_and_evaluate_poker_legends_multi_templates(
            (
                PokerLegendsTemplateSource(
                    name="session-a",
                    truth_path=source_a / "truth",
                    annotation_dir=source_a / "annotations",
                    image_root=source_a / "images",
                ),
                PokerLegendsTemplateSource(
                    name="session-b",
                    truth_path=source_b / "truth",
                    annotation_dir=source_b / "annotations",
                    image_root=source_b / "images",
                ),
            ),
            output_dir=tmp_path / "multi",
        ),
    )

    dataset = cast(dict[str, Any], summary["dataset"])
    card_parts = cast(dict[str, Any], summary["card_parts"])
    buttons = cast(dict[str, Any], summary["buttons"])

    assert dataset["frame_count"] == 2
    assert card_parts["templates"] == 4
    assert card_parts["unique_card_codes"] == ["AS", "KH"]
    assert buttons["templates"] == 2
    assert buttons["action_counts"] == {"call": 1, "check": 1}
    assert (
        tmp_path / "multi" / "merged_dataset" / "truth_overlays" / "session-a__frame_001.json"
    ).exists()
    assert (
        tmp_path / "multi" / "merged_dataset" / "truth_overlays" / "session-b__frame_001.json"
    ).exists()
    assert (tmp_path / "multi" / "multi_template_report.md").exists()


def write_source_frame(root: Path, *, card: str, left_action: str) -> None:
    annotation_dir = root / "annotations"
    image_root = root / "images"
    truth_dir = root / "truth"
    annotation_dir.mkdir(parents=True)
    image_root.mkdir(parents=True)
    truth_dir.mkdir(parents=True)

    image_path = image_root / "frame_001.png"
    image = np.full((300, 520, 3), 42, dtype=np.uint8)
    image[30:166, 35:137] = 245
    card_color = (0, 0, 180) if card[1] == "H" else (0, 0, 0)
    cv2.putText(
        image,
        card[0],
        (47, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        card_color,
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        card[1],
        (50, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        card_color,
        4,
        cv2.LINE_AA,
    )
    button_rects = {
        "primary_left": {"x": 150, "y": 80, "width": 110, "height": 124},
        "primary_middle": {"x": 275, "y": 80, "width": 110, "height": 124},
        "primary_right": {"x": 400, "y": 80, "width": 110, "height": 124},
    }
    for name, rect in button_rects.items():
        roi = image[rect["y"] : rect["y"] + rect["height"], rect["x"] : rect["x"] + rect["width"]]
        roi[:, ::2] = 20
        roi[:, 1::2] = 220
        label = left_action.title() if name == "primary_left" else name.split("_")[1].title()
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

    (annotation_dir / "frame_001.json").write_text(
        json.dumps(
            {
                "image": image_path.name,
                "regions": {
                    "cards": [
                        {
                            "name": "hero_hole_0",
                            "kind": "card",
                            "rect": {"x": 35, "y": 30, "width": 102, "height": 136},
                        }
                    ],
                    "board": [],
                    "buttons": [
                        {"name": name, "kind": "button", "rect": rect}
                        for name, rect in button_rects.items()
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    (truth_dir / "frame_001.json").write_text(
        json.dumps(
            {
                "frame_id": "frame_001",
                "screen": {"kind": "actionable_table"},
                "hero_hole_cards": [
                    {
                        "slot": "hero_hole_0",
                        "visible": True,
                        "card": card,
                        "confidence": 1.0,
                    }
                ],
                "board": [],
                "buttons": [
                    {
                        "name": "primary_left",
                        "visible": True,
                        "action_type": left_action,
                        "label": left_action.title(),
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
                "ignored_fields": [],
            }
        ),
        encoding="utf-8",
    )
