import json
from pathlib import Path
from typing import Any, cast

from holdem_bot.vision.poker_legends_number_real import (
    build_poker_legends_number_real_experiment_manifest,
)


def test_number_real_manifest_keeps_only_labeled_and_explicit_negatives(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "frame_id": "frame_000",
            "group": "texts",
            "name": "hero_stack",
            "role": "hero_stack",
            "crop_variant": "default",
            "crop_path": "crops/a.png",
            "truth_canonical_text": "$100",
            "truth_visible": True,
        },
        {
            "frame_id": "frame_001",
            "group": "texts",
            "name": "hero_stack",
            "role": "hero_stack",
            "crop_variant": "default",
            "crop_path": "crops/b.png",
            "truth_canonical_text": None,
            "truth_visible": False,
        },
        {
            "frame_id": "frame_002",
            "group": "texts",
            "name": "hero_stack",
            "role": "hero_stack",
            "crop_variant": "default",
            "crop_path": "crops/c.png",
            "truth_canonical_text": None,
            "truth_visible": None,
        },
        {
            "frame_id": "frame_003",
            "group": "texts",
            "name": "pot",
            "role": "pot",
            "crop_variant": "default",
            "crop_path": "crops/d.png",
            "truth_canonical_text": "$5",
            "truth_visible": True,
        },
    ]
    manifest_path = tmp_path / "number_crop_dataset_manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "rows": rows}),
        encoding="utf-8",
    )

    summary = build_poker_legends_number_real_experiment_manifest(
        manifest_path,
        output_dir=tmp_path / "clean",
        test_frame_modulo=2,
    )

    assert summary["candidate_rows"] == 3
    assert summary["included_rows"] == 2
    assert summary["positive_rows"] == 1
    assert summary["hard_negative_rows"] == 1
    assert summary["excluded_status_counts"] == {"unlabeled_unknown": 1}
    output_manifest = json.loads(
        (tmp_path / "clean" / "number_real_experiment_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    clean_rows = cast(list[dict[str, Any]], output_manifest["rows"])
    assert [row["clean_status"] for row in clean_rows] == [
        "labeled_visible",
        "no_visible_number",
    ]
    assert [row["split"] for row in clean_rows] == ["test", "train"]


def test_number_real_manifest_applies_review_overrides(tmp_path: Path) -> None:
    rows = [
        {
            "frame_id": "frame_000",
            "group": "texts",
            "name": "hero_stack",
            "role": "hero_stack",
            "crop_variant": "default",
            "crop_path": "crops/a.png",
            "truth_canonical_text": "$100",
            "truth_visible": True,
        },
        {
            "frame_id": "frame_000",
            "group": "texts",
            "name": "hero_stack",
            "role": "hero_stack",
            "crop_variant": "trim",
            "crop_path": "crops/b.png",
            "truth_canonical_text": "$100",
            "truth_visible": True,
        },
    ]
    manifest_path = tmp_path / "number_crop_dataset_manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "rows": rows}),
        encoding="utf-8",
    )
    overrides_path = tmp_path / "review_overrides.json"
    overrides_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rows": [
                    {
                        "frame_id": "frame_000",
                        "group": "texts",
                        "name": "hero_stack",
                        "crop_variant": "*",
                        "clean_status": "no_visible_number",
                        "reason": "manual_review_no_number",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = build_poker_legends_number_real_experiment_manifest(
        manifest_path,
        output_dir=tmp_path / "clean",
        review_overrides_path=overrides_path,
    )

    assert summary["positive_rows"] == 0
    assert summary["hard_negative_rows"] == 2
    assert summary["review_override_counts"] == {"no_visible_number": 2}
    output_manifest = json.loads(
        (tmp_path / "clean" / "number_real_experiment_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    clean_rows = cast(list[dict[str, Any]], output_manifest["rows"])
    assert [row["truth_visible"] for row in clean_rows] == [False, False]
    assert [row["truth_canonical_text"] for row in clean_rows] == [None, None]


def test_number_real_manifest_applies_labeled_visible_review_override(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "frame_id": "frame_000",
            "group": "texts",
            "name": "hero_stack",
            "role": "hero_stack",
            "crop_variant": "default",
            "crop_path": "crops/a.png",
            "truth_canonical_text": None,
            "truth_visible": None,
        },
    ]
    manifest_path = tmp_path / "number_crop_dataset_manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "rows": rows}),
        encoding="utf-8",
    )
    overrides_path = tmp_path / "review_overrides.json"
    overrides_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rows": [
                    {
                        "frame_id": "frame_000",
                        "group": "texts",
                        "name": "hero_stack",
                        "crop_variant": "default",
                        "clean_status": "labeled_visible",
                        "truth_canonical_text": "$1,000",
                        "truth_normalized_number": 1000,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = build_poker_legends_number_real_experiment_manifest(
        manifest_path,
        output_dir=tmp_path / "clean",
        review_overrides_path=overrides_path,
    )

    assert summary["positive_rows"] == 1
    assert summary["hard_negative_rows"] == 0
    output_manifest = json.loads(
        (tmp_path / "clean" / "number_real_experiment_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    clean_rows = cast(list[dict[str, Any]], output_manifest["rows"])
    assert clean_rows[0]["truth_visible"] is True
    assert clean_rows[0]["truth_canonical_text"] == "$1,000"
    assert clean_rows[0]["truth_normalized_number"] == 1000
