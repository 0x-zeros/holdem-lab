"""Reviewed real-crop manifest builder for Poker Legends number OCR experiments."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

REAL_EXPERIMENT_MANIFEST = "number_real_experiment_manifest.json"
REAL_EXPERIMENT_REPORT = "number_real_experiment_manifest_report.md"


def build_poker_legends_number_real_experiment_manifest(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    field_names: Sequence[str] = ("hero_stack",),
    crop_variants: Sequence[str] | None = None,
    test_frame_modulo: int = 5,
    include_hard_negatives: bool = True,
    review_overrides_path: str | Path | None = None,
) -> dict[str, object]:
    """Build a clean, frozen manifest for offline real-crop OCR comparison.

    Positive rows require a non-empty `truth_canonical_text`. Explicit hard negatives
    are included only when the upstream manifest says `truth_visible == false`.
    Unknown unlabeled rows are counted but excluded.
    """

    source = Path(manifest_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = _read_json_object(source)
    source_rows = _mapping_sequence(source_manifest.get("rows"))
    review_overrides = _load_review_overrides(review_overrides_path)
    fields = tuple(field_names)
    variants = tuple(crop_variants) if crop_variants is not None else None
    candidates = [
        row
        for row in source_rows
        if str(row.get("group") or "") == "texts"
        and str(row.get("name") or "") in fields
        and (variants is None or str(row.get("crop_variant") or "") in variants)
    ]
    positive_split_frame_ids = _positive_split_frame_ids(candidates)
    included: list[dict[str, object]] = []
    excluded_counts: Counter[str] = Counter()
    override_counts: Counter[str] = Counter()
    for row in candidates:
        clean_row = dict(row)
        override = _review_override_for_row(clean_row, review_overrides)
        if override is not None:
            _apply_review_override(clean_row, override)
            override_counts[str(clean_row.get("clean_status") or "unknown")] += 1
        status = _clean_status(clean_row)
        if status == "labeled_visible" or (
            include_hard_negatives and status == "no_visible_number"
        ):
            clean_row["clean_status"] = status
            clean_row["crop_path"] = str(
                _resolve_source_path(source.parent, clean_row.get("crop_path"))
            )
            clean_row["source_manifest"] = str(source)
            included.append(clean_row)
            continue
        excluded_counts[status] += 1

    frame_ids = positive_split_frame_ids
    split_by_frame = _split_by_frame(frame_ids, test_frame_modulo=test_frame_modulo)
    for index, row in enumerate(included):
        row["clean_row_id"] = f"{index:04d}"
        row["split"] = split_by_frame.get(str(row.get("frame_id") or ""), "train")

    summary: dict[str, object] = {
        "schema_version": 1,
        "source_manifest": str(source),
        "manifest": REAL_EXPERIMENT_MANIFEST,
        "field_names": list(fields),
        "crop_variants": list(variants) if variants is not None else None,
        "test_frame_modulo": max(2, test_frame_modulo),
        "include_hard_negatives": include_hard_negatives,
        "review_overrides": str(review_overrides_path) if review_overrides_path else None,
        "review_override_counts": dict(sorted(override_counts.items())),
        "source_rows": len(source_rows),
        "candidate_rows": len(candidates),
        "included_rows": len(included),
        "excluded_rows": sum(excluded_counts.values()),
        "excluded_status_counts": dict(sorted(excluded_counts.items())),
        "status_counts": dict(
            sorted(Counter(str(row["clean_status"]) for row in included).items())
        ),
        "split_counts": dict(sorted(Counter(str(row["split"]) for row in included).items())),
        "field_counts": dict(
            sorted(Counter(str(row.get("name") or "") for row in included).items())
        ),
        "variant_counts": dict(
            sorted(Counter(str(row.get("crop_variant") or "") for row in included).items())
        ),
        "positive_rows": len(
            [row for row in included if row.get("clean_status") == "labeled_visible"]
        ),
        "hard_negative_rows": len(
            [row for row in included if row.get("clean_status") == "no_visible_number"]
        ),
        "artifacts": {
            "manifest": REAL_EXPERIMENT_MANIFEST,
            "report": REAL_EXPERIMENT_REPORT,
        },
    }
    clean_manifest = {
        "schema_version": 1,
        "source_manifest": str(source),
        "description": (
            "Clean real-crop OCR experiment manifest. Positive rows are labeled-visible; "
            "hard negatives require truth_visible == false; unknown unlabeled rows are excluded."
        ),
        "summary": summary,
        "rows": included,
    }
    (output / REAL_EXPERIMENT_MANIFEST).write_text(
        json.dumps(clean_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output / REAL_EXPERIMENT_REPORT, summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a clean real-crop manifest for Poker Legends number OCR experiments."
    )
    parser.add_argument("manifest", help="Source number_crop_dataset_manifest.json")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--field-name",
        action="append",
        dest="field_names",
        help="Field to include; repeatable. Defaults to hero_stack.",
    )
    parser.add_argument(
        "--crop-variant",
        action="append",
        dest="crop_variants",
        help="Crop variant to include; repeatable. Defaults to all variants.",
    )
    parser.add_argument("--test-frame-modulo", type=int, default=5)
    parser.add_argument("--exclude-hard-negatives", action="store_true")
    parser.add_argument(
        "--review-overrides",
        help="Optional JSON file with reviewed crop status overrides.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = build_poker_legends_number_real_experiment_manifest(
        args.manifest,
        output_dir=args.out,
        field_names=tuple(args.field_names) if args.field_names else ("hero_stack",),
        crop_variants=tuple(args.crop_variants) if args.crop_variants else None,
        test_frame_modulo=args.test_frame_modulo,
        include_hard_negatives=not args.exclude_hard_negatives,
        review_overrides_path=args.review_overrides,
    )
    print(json.dumps(_stdout_summary(summary), indent=2, sort_keys=True))


def _clean_status(row: Mapping[str, object]) -> str:
    explicit = row.get("clean_status")
    if explicit in {"labeled_visible", "no_visible_number", "roi_invalid"}:
        return str(explicit)
    truth = row.get("truth_canonical_text")
    if isinstance(truth, str) and truth.strip():
        return "labeled_visible"
    if row.get("truth_visible") is False:
        return "no_visible_number"
    return "unlabeled_unknown"


def _resolve_source_path(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _positive_split_frame_ids(rows: Sequence[Mapping[str, object]]) -> list[str]:
    frame_ids: list[str] = []
    for row in rows:
        if _clean_status(row) != "labeled_visible":
            continue
        frame_id = str(row.get("frame_id") or "")
        if frame_id and frame_id not in frame_ids:
            frame_ids.append(frame_id)
    return frame_ids


def _load_review_overrides(path: str | Path | None) -> tuple[Mapping[str, object], ...]:
    if path is None:
        return ()
    data = _read_json_object(path)
    return tuple(_mapping_sequence(data.get("rows")))


def _review_override_for_row(
    row: Mapping[str, object],
    overrides: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    for override in overrides:
        if _override_matches(row, override):
            return override
    return None


def _override_matches(row: Mapping[str, object], override: Mapping[str, object]) -> bool:
    for key in ("frame_id", "group", "name", "role", "crop_variant"):
        expected = override.get(key)
        if expected is None or expected == "*":
            continue
        if str(row.get(key) or "") != str(expected):
            return False
    return True


def _apply_review_override(row: dict[str, object], override: Mapping[str, object]) -> None:
    status = str(override.get("clean_status") or "")
    if status not in {"labeled_visible", "no_visible_number", "roi_invalid"}:
        raise ValueError(f"unsupported review override status: {status!r}")
    row["clean_status"] = status
    row["review_override"] = {
        key: value
        for key, value in override.items()
        if key not in {"frame_id", "group", "name", "role", "crop_variant"}
    }
    if status == "no_visible_number":
        row["truth_canonical_text"] = None
        row["truth_normalized_number"] = None
        row["truth_visible"] = False
    elif status == "roi_invalid":
        row["truth_canonical_text"] = None
        row["truth_normalized_number"] = None
        row["truth_visible"] = None


def _split_by_frame(frame_ids: Sequence[str], *, test_frame_modulo: int) -> dict[str, str]:
    modulo = max(2, test_frame_modulo)
    return {
        frame_id: "test" if index % modulo == 0 else "train"
        for index, frame_id in enumerate(frame_ids)
    }


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    lines = [
        "# Poker Legends Number Real-Crop Manifest",
        "",
        "## Summary",
        f"- Source manifest: `{summary['source_manifest']}`",
        f"- Fields: `{summary['field_names']}`",
        f"- Crop variants: `{summary['crop_variants']}`",
        f"- Review overrides: `{summary['review_overrides']}`",
        f"- Review override counts: `{summary['review_override_counts']}`",
        f"- Candidate rows: {summary['candidate_rows']}",
        f"- Included rows: {summary['included_rows']}",
        f"- Positive rows: {summary['positive_rows']}",
        f"- Hard negative rows: {summary['hard_negative_rows']}",
        f"- Excluded rows: {summary['excluded_rows']}",
        f"- Status counts: `{summary['status_counts']}`",
        f"- Excluded status counts: `{summary['excluded_status_counts']}`",
        f"- Split counts: `{summary['split_counts']}`",
        f"- Field counts: `{summary['field_counts']}`",
        f"- Variant counts: `{summary['variant_counts']}`",
        "",
        "## Rules",
        "- `labeled_visible`: non-empty `truth_canonical_text`; eligible for positive "
        "training/eval.",
        "- `no_visible_number`: explicit `truth_visible == false`; hard negative only.",
        "- `unlabeled_unknown`: excluded, not treated as absent.",
        "- Runtime OCR and click planning are not connected by this manifest.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stdout_summary(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "manifest": summary["manifest"],
        "included_rows": summary["included_rows"],
        "positive_rows": summary["positive_rows"],
        "hard_negative_rows": summary["hard_negative_rows"],
        "review_override_counts": summary["review_override_counts"],
        "excluded_status_counts": summary["excluded_status_counts"],
        "split_counts": summary["split_counts"],
        "report": REAL_EXPERIMENT_REPORT,
    }


def _read_json_object(path: str | Path) -> dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, object], data)


def _mapping_sequence(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [cast(Mapping[str, object], item) for item in value if isinstance(item, Mapping)]


if __name__ == "__main__":
    main()
