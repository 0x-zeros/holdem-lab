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
            cv2.cvtColor(synthetic_number_crop("$123"), cv2.COLOR_RGB2BGR),
        )
        rows.append(
            {
                "frame_id": frame_id,
                "group": "texts",
                "name": "hero_stack",
                "role": "hero_stack",
                "crop_variant": "default",
                "crop_path": str(crop_path.relative_to(tmp_path)),
                "truth_canonical_text": "$123",
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
        cnn_epochs=2,
        crnn_epochs=1,
        transformer_epochs=1,
        enable_ctc=True,
    )

    assert summary["rows"] == 4
    assert summary["test_rows"] == 2
    assert summary["glyph_samples"] == 32
    targets = cast(dict[str, Any], summary["targets"])
    base = cast(dict[str, Any], targets["base"])
    display = cast(dict[str, Any], targets["display"])
    overlay = cast(dict[str, Any], targets["overlay"])
    assert cast(dict[str, Any], base["segmentation"])["counts"] == {"match": 4}
    assert cast(dict[str, Any], display["segmentation"])["counts"] == {"match": 4}
    assert overlay["rows"] == 0
    assert cast(dict[str, Any], base["template"])["evaluated"] == 2
    assert cast(dict[str, Any], base["opencv_mlp"])["evaluated"] == 2
    assert cast(dict[str, Any], base["cnn"])["status"] == "trained"
    assert cast(dict[str, Any], base["cnn"])["evaluated"] == 2
    assert cast(dict[str, Any], base["crnn_ctc"])["status"] == "trained"
    assert cast(dict[str, Any], base["crnn_ctc"])["time_step_budget"] == {
        "checked": 4,
        "failed": 0,
        "failed_examples": [],
        "input_timesteps": 40,
        "max_effective_input_timesteps": 24,
        "max_required_timesteps": 4,
        "max_target_length": 4,
        "min_effective_input_timesteps": 24,
        "min_observed_ratio": 6.0,
        "min_ratio": 2.0,
    }
    assert cast(dict[str, Any], base["transformer_ctc"])["status"] == "trained"
    assert cast(dict[str, Any], base["template_cnn"])["evaluated"] == 2
    assert (tmp_path / "chars" / "number_char_recognizer_summary.json").exists()
    assert (tmp_path / "chars" / "number_char_recognizer_report.md").exists()
    assert (tmp_path / "chars" / "number_char_recognizer_review.html").exists()


def test_number_char_recognizer_reports_hard_negative_false_accepts(
    tmp_path: Path,
) -> None:
    crop_dir = tmp_path / "crops"
    crop_dir.mkdir()
    rows: list[dict[str, object]] = []
    for index, text in enumerate(["$100", "$200", "$300"]):
        frame_id = f"train_{index:03d}"
        crop_path = crop_dir / f"{frame_id}.png"
        cv2.imwrite(
            str(crop_path),
            cv2.cvtColor(synthetic_number_crop(text), cv2.COLOR_RGB2BGR),
        )
        rows.append(
            {
                "frame_id": frame_id,
                "group": "texts",
                "name": "hero_stack",
                "role": "hero_stack",
                "crop_variant": "default",
                "crop_path": str(crop_path.relative_to(tmp_path)),
                "truth_canonical_text": text,
                "truth_normalized_number": int(text[1:]),
                "split": "train",
            }
        )
    test_crop_path = crop_dir / "test_000.png"
    cv2.imwrite(
        str(test_crop_path),
        cv2.cvtColor(synthetic_number_crop("$100"), cv2.COLOR_RGB2BGR),
    )
    rows.append(
        {
            "frame_id": "test_000",
            "group": "texts",
            "name": "hero_stack",
            "role": "hero_stack",
            "crop_variant": "default",
            "crop_path": str(test_crop_path.relative_to(tmp_path)),
            "truth_canonical_text": "$100",
            "truth_normalized_number": 100,
            "split": "test",
        }
    )
    negative_crop = np.full((54, 220, 3), (24, 60, 110), dtype=np.uint8)
    negative_crop_path = crop_dir / "test_negative.png"
    cv2.imwrite(str(negative_crop_path), cv2.cvtColor(negative_crop, cv2.COLOR_RGB2BGR))
    rows.append(
        {
            "frame_id": "test_negative",
            "group": "texts",
            "name": "hero_stack",
            "role": "hero_stack",
            "crop_variant": "default",
            "crop_path": str(negative_crop_path.relative_to(tmp_path)),
            "truth_canonical_text": None,
            "truth_visible": False,
            "clean_status": "no_visible_number",
            "split": "test",
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
        targets=("base",),
        cnn_epochs=2,
        enable_ctc=False,
        enable_tesseract=False,
    )

    assert summary["hard_negative_rows"] == 1
    assert summary["hard_negative_test_rows"] == 1
    base = cast(dict[str, Any], cast(dict[str, Any], summary["targets"])["base"])
    assert base["hard_negative_test_rows"] == 1
    hard_negatives = cast(dict[str, Any], base["hard_negatives"])
    for method in ("template", "cnn", "template_cnn"):
        assert cast(dict[str, Any], hard_negatives[method])["false_accepts"] == 0


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
