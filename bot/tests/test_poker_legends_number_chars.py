import json
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from holdem_bot.vision import (
    build_and_evaluate_poker_legends_number_char_recognizers,
    segment_number_characters,
)


def test_segment_number_characters_on_synthetic_stack_crop() -> None:
    crop = synthetic_number_crop("$123+45")

    boxes = segment_number_characters(crop)

    assert len(boxes) == 7
    assert [box.width > 0 and box.height > 0 for box in boxes] == [True] * 7


def test_number_char_recognizer_report_compares_template_mlp_and_tesseract(
    tmp_path: Path,
) -> None:
    crop_dir = tmp_path / "crops"
    crop_dir.mkdir()
    rows: list[dict[str, object]] = []
    for index in range(4):
        frame_id = f"frame_{index:03d}"
        crop_path = crop_dir / f"{frame_id}.png"
        cv2.imwrite(
            str(crop_path),
            cv2.cvtColor(synthetic_number_crop("$123+45"), cv2.COLOR_RGB2BGR),
        )
        rows.append(
            {
                "frame_id": frame_id,
                "group": "texts",
                "name": "hero_stack",
                "role": "hero_stack",
                "crop_variant": "default",
                "crop_path": str(crop_path.relative_to(tmp_path)),
                "truth_canonical_text": "$123+45",
                "truth_normalized_number": 123,
            }
        )
    manifest_path = tmp_path / "number_crop_dataset_manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "rows": rows}),
        encoding="utf-8",
    )

    summary = build_and_evaluate_poker_legends_number_char_recognizers(
        manifest_path,
        output_dir=tmp_path / "chars",
        test_frame_modulo=2,
    )

    assert summary["rows"] == 4
    assert summary["test_rows"] == 2
    assert summary["glyph_samples"] == 28
    assert cast(dict[str, Any], summary["segmentation"])["counts"] == {"match": 4}
    assert cast(dict[str, Any], summary["template"])["evaluated"] == 2
    assert cast(dict[str, Any], summary["opencv_mlp"])["evaluated"] == 2
    assert cast(dict[str, Any], summary["cnn"])["status"] == "not_run"
    assert (tmp_path / "chars" / "number_char_recognizer_summary.json").exists()
    assert (tmp_path / "chars" / "number_char_recognizer_report.md").exists()
    assert (tmp_path / "chars" / "number_char_recognizer_review.html").exists()


def synthetic_number_crop(text: str) -> np.ndarray:
    crop = np.full((54, 220, 3), (24, 60, 110), dtype=np.uint8)
    cv2.putText(
        crop,
        text,
        (6, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return crop
