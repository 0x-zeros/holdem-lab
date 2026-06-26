import json
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from holdem_bot.vision import (
    PokerLegendsCardClassifier,
    PokerLegendsCardClassifierManifest,
    build_and_evaluate_poker_legends_card_classifier,
    build_poker_legends_card_classifier,
)


def test_card_classifier_generalizes_to_unseen_card_combinations(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "annotations"
    image_root = tmp_path / "images"
    annotation_dir.mkdir()
    image_root.mkdir()
    truth_paths = []
    for card in ("AS", "AH", "KS", "KH"):
        frame_id = f"frame_{card}"
        write_synthetic_classifier_card_frame(
            image_root / f"{frame_id}.png",
            annotation_dir / f"{frame_id}.json",
            card=card,
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
        truth_paths.append(truth_path)

    summary = cast(
        dict[str, Any],
        build_and_evaluate_poker_legends_card_classifier(
            truth_paths,
            annotation_dir=annotation_dir,
            image_root=image_root,
            output_dir=tmp_path / "classifier",
            visibility_max_distance=0.30,
            rank_max_distance=0.30,
            suit_max_distance=0.30,
        ),
    )

    assert summary["rank_labels"] == ["A", "K"]
    assert summary["suit_labels"] == ["H", "S"]
    assert summary["self_eval"]["visible_accuracy"] == 1.0
    assert summary["self_eval"]["hidden_false_positive_rate"] == 0.0
    assert summary["leave_card_eval"]["visible_accuracy"] == 1.0
    assert summary["leave_card_eval"]["rank_accuracy"] == 1.0
    assert summary["leave_card_eval"]["suit_accuracy"] == 1.0
    assert summary["leave_card_eval"]["hidden_false_positive_rate"] == 0.0
    assert (tmp_path / "classifier" / "card_classifier_manifest.json").exists()
    assert (tmp_path / "classifier" / "card_classifier_report.md").exists()

    classifier = PokerLegendsCardClassifier.from_manifest(
        tmp_path / "classifier" / "card_classifier_manifest.json"
    )
    annotation = json.loads((annotation_dir / "frame_AH.json").read_text(encoding="utf-8"))
    predictions = classifier.recognize(
        image_root / "frame_AH.png",
        annotation,
        frame_id="frame_AH",
        exclude_card="AH",
    )
    assert predictions[0].card == "AH"
    assert predictions[0].visible
    assert predictions[1].card is None
    assert not predictions[1].visible


def test_card_classifier_skips_blocked_overlay_by_default(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "annotations"
    image_root = tmp_path / "images"
    annotation_dir.mkdir()
    image_root.mkdir()
    write_synthetic_classifier_card_frame(
        image_root / "frame_001.png",
        annotation_dir / "frame_001.json",
        card="AS",
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

    manifest = build_poker_legends_card_classifier(
        [truth_path],
        annotation_dir=annotation_dir,
        image_root=image_root,
        output_dir=tmp_path / "classifier",
    )

    assert manifest.samples == ()
    reloaded = PokerLegendsCardClassifierManifest.read_json(
        tmp_path / "classifier" / "card_classifier_manifest.json"
    )
    assert reloaded.samples == ()


def write_synthetic_classifier_card_frame(
    image_path: Path,
    annotation_path: Path,
    *,
    card: str,
) -> None:
    image = np.full((180, 220, 3), 32, dtype=np.uint8)
    image[20:156, 30:132] = 245
    image[20:156, 148:202] = 58
    color = (0, 0, 180) if card[1] == "H" else (0, 0, 0)
    cv2.putText(
        image,
        card[0],
        (42, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        color,
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        card[1],
        (45, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        color,
        4,
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
                            "rect": {"x": 30, "y": 20, "width": 102, "height": 136},
                        }
                    ],
                    "board": [
                        {
                            "name": "board_0",
                            "kind": "card",
                            "rect": {"x": 148, "y": 20, "width": 54, "height": 136},
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
