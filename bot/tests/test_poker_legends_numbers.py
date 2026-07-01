import json
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from holdem_bot.vision import (
    PokerLegendsNumberRecognizer,
    build_poker_legends_number_ocr_report,
    parse_poker_legends_chip_amount,
    parse_poker_legends_chip_numbers,
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
    assert predictions["primary_left"].normalized_number == 25


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
