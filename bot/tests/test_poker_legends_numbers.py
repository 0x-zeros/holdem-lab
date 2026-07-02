import json
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from holdem_bot.vision import (
    PokerLegendsNumberRecognizer,
    build_poker_legends_number_ocr_report,
    parse_poker_legends_chip_amount,
    parse_poker_legends_chip_components,
    parse_poker_legends_chip_numbers,
    poker_legends_numbers,
)


def test_poker_legends_number_recognizer_parses_core_numeric_rois(tmp_path: Path) -> None:
    image_path = tmp_path / "numbers.png"
    annotation_path = tmp_path / "numbers.json"
    write_number_fixture(image_path, annotation_path)
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))

    predictions = {
        prediction.name: prediction
        for prediction in PokerLegendsNumberRecognizer().recognize(image_path, annotation)
    }

    assert predictions["pot"].normalized_number == 1350
    assert predictions["hero_stack"].normalized_number == 1000
    assert predictions["hero_stack"].base_number == 290
    assert predictions["hero_stack"].overlay_number == 710
    assert predictions["hero_stack"].total_number == 1000
    assert predictions["primary_left"].normalized_number == 25


def test_poker_legends_number_recognizer_adds_missing_canonical_text_roi(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "frame.png"
    image = np.full((982, 1600, 3), 24, dtype=np.uint8)
    cv2.putText(
        image,
        "$10",
        (716, 552),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(image_path), image)
    annotation = {"image": image_path.name, "regions": {"texts": [], "buttons": []}}

    predictions = PokerLegendsNumberRecognizer().recognize(
        image_path,
        annotation,
        text_names=("hero_current_bet",),
        button_names=(),
    )

    assert predictions[0].name == "hero_current_bet"
    assert predictions[0].normalized_number == 10
    assert predictions[0].confidence >= 0.70


def test_poker_legends_number_ocr_report_compares_truth(tmp_path: Path) -> None:
    image_path = tmp_path / "numbers.png"
    annotation_path = tmp_path / "numbers.json"
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    write_number_fixture(image_path, annotation_path)
    (truth_dir / "numbers.json").write_text(
        json.dumps(
            {
                "frame_id": "numbers",
                "texts": [
                    {"name": "pot", "visible": True, "normalized_number": 1350},
                    {"name": "hero_stack", "visible": True, "normalized_number": 1000},
                ],
                "buttons": [
                    {"name": "primary_left", "visible": True, "label": "Call $25"},
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = cast(
        dict[str, Any],
        build_poker_legends_number_ocr_report(
            [annotation_path],
            image_root=tmp_path,
            truth_dir=truth_dir,
            output_dir=tmp_path / "out",
        ),
    )

    assert summary["compared"] == 3
    assert summary["correct"] == 3
    assert summary["accuracy"] == 1.0
    assert (tmp_path / "out" / "number_ocr_report.md").exists()


def test_poker_legends_chip_parser_normalizes_split_ocr_text() -> None:
    assert parse_poker_legends_chip_amount("$1.2\n\n3K") == 1230
    assert parse_poker_legends_chip_amount("$31.0") == 310
    assert parse_poker_legends_chip_amount("$1,250") == 1250
    assert parse_poker_legends_chip_amount("$990+10") == 1000
    assert parse_poker_legends_chip_amount(",\n\n$\n\n3\n\n00\n\n.") == 300
    assert parse_poker_legends_chip_numbers("2\n\n,\n\n$\n\n5\n\n00") == (500,)
    assert parse_poker_legends_chip_amount("1,052+50\n\nM") == 1102
    assert parse_poker_legends_chip_amount("$1.25K") == 1250
    assert parse_poker_legends_chip_components("$334+10") == {
        "base_number": 334,
        "overlay_number": 10,
        "total_number": 344,
    }
    assert parse_poker_legends_chip_components("$1,250") == {
        "base_number": 1250,
        "overlay_number": None,
        "total_number": 1250,
    }


def test_poker_legends_number_confidence_marks_fragmented_ocr_low() -> None:
    assert poker_legends_numbers._confidence("$1,250", (1250,)) == 0.90
    assert poker_legends_numbers._confidence("$990+10", (990, 10)) == 0.82
    assert poker_legends_numbers._confidence("$1,005 +10", (1005, 10)) == 0.82
    assert poker_legends_numbers._confidence("$1,146+80", (1146, 80)) == 0.82
    assert poker_legends_numbers._confidence("4\n\n6\n\n0", (460,)) == 0.65
    assert poker_legends_numbers._confidence("$890+110 4", (890, 1104)) == 0.65
    assert poker_legends_numbers._confidence("$840+1 +160", (840, 1, 160)) == 0.65
    assert poker_legends_numbers._confidence("M7\n\n999052\n\n5", (79990525,)) == 0.65
    assert poker_legends_numbers._confidence("1$890+10", (890, 10)) == 0.65
    assert poker_legends_numbers._confidence("930,", (930,)) == 0.65
    assert poker_legends_numbers._confidence("6845+ 50", (6845, 50)) == 0.65
    assert poker_legends_numbers._confidence("99+995", (99, 995)) == 0.65


def write_number_fixture(image_path: Path, annotation_path: Path) -> None:
    image = np.full((260, 520, 3), 24, dtype=np.uint8)
    regions = {
        "texts": [
            {"name": "pot", "kind": "text", "rect": {"x": 24, "y": 24, "width": 170, "height": 54}},
            {
                "name": "hero_stack",
                "kind": "text",
                "rect": {"x": 24, "y": 96, "width": 210, "height": 54},
            },
        ],
        "buttons": [
            {
                "name": "primary_left",
                "kind": "button",
                "rect": {"x": 260, "y": 88, "width": 210, "height": 78},
            }
        ],
    }
    cv2.putText(
        image,
        "$1.35K",
        (34, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "$290+710",
        (34, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    image[88:166, 260:470] = (24, 72, 190)
    cv2.putText(
        image,
        "Call $25",
        (278, 138),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(image_path), image)
    annotation_path.write_text(
        json.dumps(
            {
                "frame_id": "numbers",
                "image": image_path.name,
                "regions": regions,
            }
        ),
        encoding="utf-8",
    )
