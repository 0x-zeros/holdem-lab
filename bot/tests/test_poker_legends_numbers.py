import json
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from holdem_bot.vision import (
    PokerLegendsComponentNumberShadowRecognizer,
    PokerLegendsNumberRecognizer,
    build_poker_legends_number_crop_dataset,
    build_poker_legends_number_ocr_report,
    evaluate_poker_legends_number_crop_dataset,
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
    assert predictions["hero_stack"].normalized_number == 290
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
                    {"name": "hero_stack", "visible": True, "normalized_number": 290},
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


def test_poker_legends_number_crop_dataset_exports_variants_and_labels(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "numbers.png"
    annotation_path = tmp_path / "numbers.json"
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    write_number_fixture(image_path, annotation_path)
    (truth_dir / "numbers.json").write_text(
        json.dumps(
            {
                "frame_id": "numbers",
                "screen": {"kind": "actionable_table"},
                "texts": [
                    {
                        "name": "hero_stack",
                        "visible": True,
                        "value": "$290+710",
                        "normalized_number": 1000,
                    },
                ],
                "seats": [
                    {
                        "name": "hero",
                        "visible": True,
                        "stack": 290,
                        "committed": 25,
                    }
                ],
                "buttons": [
                    {"name": "primary_left", "visible": True, "label": "Call $25"},
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = build_poker_legends_number_crop_dataset(
        [annotation_path],
        image_root=tmp_path,
        truth_dir=truth_dir,
        output_dir=tmp_path / "out",
        text_names=("hero_stack", "hero_current_bet"),
        button_names=("primary_left",),
    )

    assert summary["frames"] == 1
    assert summary["crops"] == 5
    assert summary["labeled_crops"] == 5
    assert summary["field_counts"] == {
        "buttons:primary_left": 1,
        "texts:hero_current_bet": 1,
        "texts:hero_stack": 3,
    }
    rows = cast(list[dict[str, Any]], summary["rows"])
    hero_rows = [row for row in rows if row["name"] == "hero_stack"]
    assert {row["crop_variant"] for row in hero_rows} == {
        "default",
        "hero_stack_no_pad",
        "hero_stack_trim_right_16",
    }
    assert {row["truth_canonical_text"] for row in hero_rows} == {"$290+710"}
    assert {row["truth_normalized_number"] for row in hero_rows} == {290}
    assert {tuple(row["truth_chip_numbers"]) for row in hero_rows} == {(290, 710)}
    current_bet_row = next(row for row in rows if row["name"] == "hero_current_bet")
    assert current_bet_row["truth_canonical_text"] == "$25"
    assert current_bet_row["truth_chip_numbers"] == [25]
    assert all((tmp_path / "out" / str(row["crop_path"])).exists() for row in rows)
    assert (tmp_path / "out" / "number_crop_dataset_manifest.json").exists()
    assert (tmp_path / "out" / "number_crop_dataset_report.md").exists()


def test_poker_legends_number_crop_dataset_defaults_skip_unstable_numeric_sources(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "numbers.png"
    annotation_path = tmp_path / "numbers.json"
    write_number_fixture(image_path, annotation_path)

    summary = build_poker_legends_number_crop_dataset(
        [annotation_path],
        image_root=tmp_path,
        output_dir=tmp_path / "out",
    )

    field_counts = cast(dict[str, int], summary["field_counts"])
    assert "buttons:primary_left" not in field_counts
    assert "texts:hero_current_bet" not in field_counts
    assert field_counts["texts:hero_stack"] == 3
    assert field_counts["texts:pot"] == 1
    assert field_counts["texts:right_top_stack"] == 2


def test_right_top_stack_dataset_does_not_export_left_trim_variant() -> None:
    specs = poker_legends_numbers._dataset_crop_specs_for_region(
        poker_legends_numbers.ScreenRect(972, 300, 160, 58),
        group="texts",
        name="right_top_stack",
    )

    assert {spec.variant for spec in specs} == {"default", "right_top_stack_no_pad"}


def test_poker_legends_number_crop_ocr_evaluator_reports_variant_stats(
    tmp_path: Path,
) -> None:
    crop_dir = tmp_path / "crops"
    crop_dir.mkdir()
    crop_path = crop_dir / "pot.png"
    crop = np.full((90, 220, 3), 24, dtype=np.uint8)
    cv2.putText(
        crop,
        "$100",
        (12, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        (245, 245, 245),
        3,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(crop_path), crop)
    manifest_path = tmp_path / "number_crop_dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rows": [
                    {
                        "frame_id": "frame_001",
                        "group": "texts",
                        "name": "pot",
                        "role": "pot",
                        "crop_variant": "default",
                        "crop_path": "crops/pot.png",
                        "roi_rect": [0, 0, 220, 90],
                        "screen_kind": "actionable_table",
                        "truth_normalized_number": 100,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = evaluate_poker_legends_number_crop_dataset(
        manifest_path,
        output_dir=tmp_path / "eval",
    )

    overall = cast(dict[str, Any], summary["overall"])
    assert overall["labeled"] == 1
    assert overall["correct"] == 1
    assert overall["accepted_correct"] == 1
    assert overall["accepted_wrong"] == 0
    assert summary["review_queue"] == []
    by_field_variant = cast(dict[str, dict[str, Any]], summary["by_field_variant"])
    assert by_field_variant["texts:pot:default"]["correct"] == 1
    assert (tmp_path / "eval" / "number_crop_ocr_summary.json").exists()
    assert (tmp_path / "eval" / "number_crop_ocr_report.md").exists()
    assert (tmp_path / "eval" / "number_crop_ocr_review_queue.json").exists()
    assert (tmp_path / "eval" / "number_crop_ocr_review_queue.md").exists()


def test_poker_legends_number_crop_dataset_does_not_label_ambiguous_hero_stack(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "numbers.png"
    annotation_path = tmp_path / "numbers.json"
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    write_number_fixture(image_path, annotation_path)
    (truth_dir / "numbers.json").write_text(
        json.dumps(
            {
                "frame_id": "numbers",
                "screen": {"kind": "table_observe"},
                "seats": [
                    {
                        "name": "hero",
                        "visible": True,
                        "stack": 1000,
                        "committed": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = build_poker_legends_number_crop_dataset(
        [annotation_path],
        image_root=tmp_path,
        truth_dir=truth_dir,
        output_dir=tmp_path / "out",
        text_names=("hero_stack",),
        button_names=(),
    )

    assert summary["labeled_crops"] == 0
    rows = cast(list[dict[str, Any]], summary["rows"])
    assert {row["truth_normalized_number"] for row in rows} == {None}


def test_poker_legends_number_crop_dataset_requires_reviewed_stack_overlay_label(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "numbers.png"
    annotation_path = tmp_path / "numbers.json"
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    write_number_fixture(image_path, annotation_path)
    (truth_dir / "numbers.json").write_text(
        json.dumps(
            {
                "frame_id": "numbers",
                "review": {"status": "candidate_unreviewed"},
                "texts": [
                    {
                        "name": "hero_stack",
                        "visible": True,
                        "value": "1229",
                        "normalized_number": 1229,
                    },
                ],
                "seats": [
                    {
                        "name": "hero",
                        "visible": True,
                        "stack": 1229,
                        "committed": 30,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = build_poker_legends_number_crop_dataset(
        [annotation_path],
        image_root=tmp_path,
        truth_dir=truth_dir,
        output_dir=tmp_path / "out",
        text_names=("hero_stack",),
        button_names=(),
    )

    assert summary["labeled_crops"] == 0
    rows = cast(list[dict[str, Any]], summary["rows"])
    assert {row["truth_value"] for row in rows} == {None}
    assert {row["truth_normalized_number"] for row in rows} == {None}


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


def test_component_number_shadow_recognizer_loads_template_cnn_summary(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "number_char_recognizer_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evaluation_rows": [
                    {
                        "frame_id": "frame_001",
                        "field": "texts.hero_stack",
                        "crop_variant": "default",
                        "targets": {
                            "base": {
                                "expected": "$290",
                                "is_positive": True,
                                "segmentation_status": "match",
                                "template_cnn": {
                                    "text": "$290",
                                    "confidence": 0.91,
                                    "accepted": True,
                                    "reason": "accepted",
                                },
                            },
                            "overlay": {
                                "expected": "+710",
                                "is_positive": True,
                                "segmentation_status": "match",
                                "template_cnn": {
                                    "text": "+710",
                                    "confidence": 0.97,
                                    "accepted": True,
                                    "reason": "accepted",
                                },
                            },
                            "display": {
                                "expected": "$290+710",
                                "is_positive": True,
                                "segmentation_status": "match",
                                "template_cnn": {
                                    "text": "$290+710",
                                    "confidence": 0.88,
                                    "accepted": True,
                                    "reason": "accepted",
                                },
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    recognizer = PokerLegendsComponentNumberShadowRecognizer.from_summary(summary_path)
    predictions = recognizer.recognize("frame_001", text_names=("hero_stack",))

    assert len(predictions) == 1
    prediction = predictions[0]
    assert prediction.raw == "$290+710"
    assert prediction.normalized_number == 290
    assert prediction.base_number == 290
    assert prediction.overlay_number == 710
    assert prediction.total_number == 1000
    assert prediction.accepted is True
    assert prediction.confidence == 0.91


def test_component_number_shadow_consensus_uses_template_base_when_display_agrees(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "number_char_recognizer_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evaluation_rows": [
                    {
                        "frame_id": "frame_001",
                        "field": "texts.hero_stack",
                        "crop_variant": "default",
                        "targets": {
                            "base": {
                                "expected": "$399",
                                "is_positive": True,
                                "segmentation_status": "match",
                                "template_cnn": {
                                    "text": None,
                                    "confidence": 0.18,
                                    "accepted": False,
                                    "reason": "disagreement",
                                },
                                "template": {
                                    "text": "$399",
                                    "confidence": 0.18,
                                    "accepted": True,
                                    "reason": "accepted",
                                },
                            },
                            "overlay": {
                                "expected": "+5",
                                "is_positive": True,
                                "segmentation_status": "match",
                                "template_cnn": {
                                    "text": "+5",
                                    "confidence": 0.99,
                                    "accepted": True,
                                    "reason": "accepted",
                                },
                            },
                            "display": {
                                "expected": "$399+5",
                                "is_positive": True,
                                "segmentation_status": "match",
                                "template_cnn": {
                                    "text": "$399+5",
                                    "confidence": 0.28,
                                    "accepted": True,
                                    "reason": "accepted",
                                },
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    recognizer = PokerLegendsComponentNumberShadowRecognizer.from_summary(summary_path)
    prediction = recognizer.recognize("frame_001", text_names=("hero_stack",))[0]

    assert prediction.method == "component_consensus"
    assert prediction.raw == "$399+5"
    assert prediction.base_number == 399
    assert prediction.overlay_number == 5
    assert prediction.total_number == 404
    assert prediction.accepted is True
    assert prediction.reason == "accepted_component_consensus"
    components = dict(prediction.components or {})
    assert dict(components["base"])["selected_method"] == "template"
    assert dict(components["base"])["requires_display_agreement"] is True


def test_component_number_shadow_consensus_rejects_template_base_without_display_agreement(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "number_char_recognizer_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evaluation_rows": [
                    {
                        "frame_id": "frame_001",
                        "field": "texts.hero_stack",
                        "crop_variant": "default",
                        "targets": {
                            "base": {
                                "expected": "$399",
                                "is_positive": True,
                                "segmentation_status": "match",
                                "template_cnn": {
                                    "text": None,
                                    "confidence": 0.18,
                                    "accepted": False,
                                    "reason": "disagreement",
                                },
                                "template": {
                                    "text": "$399",
                                    "confidence": 0.18,
                                    "accepted": True,
                                    "reason": "accepted",
                                },
                            },
                            "overlay": {
                                "expected": "+5",
                                "is_positive": True,
                                "segmentation_status": "match",
                                "template_cnn": {
                                    "text": "+5",
                                    "confidence": 0.99,
                                    "accepted": True,
                                    "reason": "accepted",
                                },
                            },
                            "display": {
                                "expected": "$390+5",
                                "is_positive": True,
                                "segmentation_status": "match",
                                "template_cnn": {
                                    "text": "$390+5",
                                    "confidence": 0.28,
                                    "accepted": True,
                                    "reason": "accepted",
                                },
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    recognizer = PokerLegendsComponentNumberShadowRecognizer.from_summary(summary_path)
    prediction = recognizer.recognize("frame_001", text_names=("hero_stack",))[0]

    assert prediction.raw == "$390+5"
    assert prediction.accepted is False
    assert prediction.reason == "component_fallback_display_mismatch"


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
    assert poker_legends_numbers._confidence("5900+100m", (5900, 100_000_000)) == 0.65
    assert poker_legends_numbers._confidence("6780+1559", (6780, 1559)) == 0.65


def test_hero_current_bet_confidence_rejects_tiny_false_positives() -> None:
    assert (
        poker_legends_numbers._field_confidence(
            group="texts",
            name="hero_current_bet",
            raw="1",
            numbers=(1,),
        )
        == 0.65
    )
    assert (
        poker_legends_numbers._field_confidence(
            group="texts",
            name="hero_current_bet",
            raw=".7",
            numbers=(7,),
        )
        == 0.65
    )
    assert (
        poker_legends_numbers._field_confidence(
            group="texts",
            name="hero_current_bet",
            raw="$5",
            numbers=(5,),
        )
        == 0.90
    )


def test_stack_variant_acceptance_requires_currency_for_overlay() -> None:
    assert not poker_legends_numbers._is_safe_stack_variant(
        number_prediction("560+100", 560, 0.82, "right_top_stack_no_pad")
    )
    assert poker_legends_numbers._is_safe_stack_variant(
        number_prediction("$560+100", 560, 0.82, "right_top_stack_no_pad")
    )


def test_poker_legends_number_prediction_round_trips_crop_evidence() -> None:
    prediction = poker_legends_numbers.PokerLegendsNumberPrediction(
        name="hero_stack",
        group="texts",
        visible=True,
        raw="$990+10",
        numbers=(990, 10),
        first_number=990,
        sum_number=1000,
        normalized_number=990,
        confidence=0.82,
        base_number=990,
        overlay_number=10,
        total_number=1000,
        crop_variant="hero_stack_no_pad",
        roi_rect=(660, 708, 210, 52),
    )

    restored = poker_legends_numbers.PokerLegendsNumberPrediction.from_dict(prediction.to_dict())

    assert restored.crop_variant == "hero_stack_no_pad"
    assert restored.roi_rect == (660, 708, 210, 52)


def test_poker_legends_number_prediction_from_dict_defaults_crop_evidence() -> None:
    restored = poker_legends_numbers.PokerLegendsNumberPrediction.from_dict(
        {
            "name": "pot",
            "group": "texts",
            "visible": True,
            "raw": "$100",
            "numbers": [100],
            "first_number": 100,
            "sum_number": None,
            "normalized_number": 100,
            "confidence": 0.90,
        }
    )

    assert restored.crop_variant == "default"
    assert restored.roi_rect is None


def test_hero_stack_crop_selector_prefers_no_pad_for_left_edge_pollution() -> None:
    selected = poker_legends_numbers._select_field_prediction(
        (
            number_prediction("28\n\n$990+10", 1000, 0.65, "default"),
            number_prediction("$990+10", 990, 0.82, "hero_stack_no_pad"),
            number_prediction("$990+10", 990, 0.82, "hero_stack_trim_right_16"),
        ),
        name="hero_stack",
        group="texts",
    )

    assert selected.crop_variant == "hero_stack_no_pad"


def test_hero_stack_crop_selector_prefers_trim_right_for_trailing_pollution() -> None:
    selected = poker_legends_numbers._select_field_prediction(
        (
            number_prediction("$890+110 4", 1994, 0.65, "default"),
            number_prediction("6890+110m", 110006890, 0.82, "hero_stack_no_pad"),
            number_prediction("$89 0+110", 890, 0.82, "hero_stack_trim_right_16"),
        ),
        name="hero_stack",
        group="texts",
    )

    assert selected.crop_variant == "hero_stack_trim_right_16"
    assert selected.normalized_number == 890


def test_hero_stack_crop_selector_detects_split_overlay_token() -> None:
    selected = poker_legends_numbers._select_field_prediction(
        (
            number_prediction("$900+1 00", 900, 0.65, "default"),
            number_prediction("$900+1004", 900, 0.65, "hero_stack_no_pad"),
            number_prediction("$900+100", 900, 0.82, "hero_stack_trim_right_16"),
        ),
        name="hero_stack",
        group="texts",
    )

    assert selected.crop_variant == "hero_stack_trim_right_16"


def number_prediction(
    raw: str,
    normalized_number: int | None,
    confidence: float,
    crop_variant: str,
) -> poker_legends_numbers.PokerLegendsNumberPrediction:
    numbers = parse_poker_legends_chip_numbers(raw)
    components = parse_poker_legends_chip_components(raw)
    return poker_legends_numbers.PokerLegendsNumberPrediction(
        name="hero_stack",
        group="texts",
        visible=True,
        raw=raw,
        numbers=numbers,
        first_number=numbers[0] if numbers else None,
        sum_number=sum(numbers) if len(numbers) >= 2 else None,
        normalized_number=normalized_number,
        confidence=confidence,
        base_number=components["base_number"],
        overlay_number=components["overlay_number"],
        total_number=components["total_number"],
        crop_variant=crop_variant,
    )


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
