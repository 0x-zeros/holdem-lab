"""Numeric OCR helpers for Poker Legends text and button ROIs."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self, cast

import cv2
import pytesseract  # type: ignore[import-untyped]
from numpy.typing import NDArray

from holdem_bot.vision.annotations import ScreenRect
from holdem_bot.vision.poker_legends_layout import poker_legends_layout_regions

ImageArray = NDArray[Any]
RgbImage = ImageArray

_NUMERIC_WHITELIST = "$0123456789.,+KkMm "
_BUTTON_WHITELIST = "$0123456789.,+KkMmABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "


def parse_poker_legends_chip_numbers(text: str) -> tuple[int, ...]:
    return tuple(_numbers_from_text(text))


def parse_poker_legends_chip_amount(text: str) -> int | None:
    numbers = parse_poker_legends_chip_numbers(text)
    return _normalized_number(text, numbers=numbers)


def parse_poker_legends_chip_components(text: str) -> dict[str, int | None]:
    numbers = parse_poker_legends_chip_numbers(text)
    base, overlay, total = _number_components(text, numbers=numbers)
    return {
        "base_number": base,
        "overlay_number": overlay,
        "total_number": total,
    }


@dataclass(frozen=True, slots=True)
class PokerLegendsNumberPrediction:
    name: str
    group: str
    visible: bool
    raw: str
    numbers: tuple[int, ...]
    first_number: int | None
    sum_number: int | None
    normalized_number: int | None
    confidence: float
    base_number: int | None = None
    overlay_number: int | None = None
    total_number: int | None = None
    crop_variant: str = "default"
    roi_rect: tuple[int, int, int, int] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        return cls(
            name=str(data["name"]),
            group=str(data["group"]),
            visible=bool(data["visible"]),
            raw=str(data["raw"]),
            numbers=tuple(_to_int(item) for item in _sequence(data["numbers"])),
            first_number=_optional_int(data["first_number"]),
            sum_number=_optional_int(data["sum_number"]),
            normalized_number=_optional_int(data["normalized_number"]),
            confidence=_to_float(data["confidence"]),
            base_number=_optional_int(data.get("base_number")),
            overlay_number=_optional_int(data.get("overlay_number")),
            total_number=_optional_int(data.get("total_number")),
            crop_variant=str(data.get("crop_variant") or "default"),
            roi_rect=_optional_rect_tuple(data.get("roi_rect")),
        )


@dataclass(frozen=True, slots=True)
class _CropSpec:
    variant: str
    rect: ScreenRect
    pad: int


class PokerLegendsNumberRecognizer:
    def recognize(
        self,
        image_path: str | Path,
        annotation: Mapping[str, object],
        *,
        text_names: Sequence[str] = ("pot", "hero_stack", "right_top_stack"),
        button_names: Sequence[str] = ("primary_left",),
    ) -> tuple[PokerLegendsNumberPrediction, ...]:
        image = _load_rgb_image(image_path)
        selected_text_names = set(text_names)
        selected_button_names = set(button_names)
        regions = _region_groups(annotation)
        regions["texts"] = _with_canonical_text_regions(
            regions.get("texts", ()),
            selected_text_names=selected_text_names,
            image_width=int(image.shape[1]),
            image_height=int(image.shape[0]),
        )
        predictions: list[PokerLegendsNumberPrediction] = []
        for region in regions.get("texts", ()):
            name = str(region.get("name") or "")
            if name in selected_text_names:
                predictions.append(
                    self._recognize_region(
                        image,
                        _rect_from_region(region),
                        name=name,
                        group="texts",
                    )
                )
        for region in regions.get("buttons", ()):
            name = str(region.get("name") or "")
            if name in selected_button_names:
                predictions.append(
                    self._recognize_region(
                        image,
                        _rect_from_region(region),
                        name=name,
                        group="buttons",
                    )
                )
        return tuple(predictions)

    def _recognize_region(
        self,
        image: RgbImage,
        rect: ScreenRect,
        *,
        name: str,
        group: str,
    ) -> PokerLegendsNumberPrediction:
        default = self._recognize_crop(
            image,
            _default_crop_spec(rect, group=group),
            name=name,
            group=group,
        )
        specs = _fallback_crop_specs_for_prediction(default, rect, name=name, group=group)
        predictions = (default,) + tuple(
            self._recognize_crop(image, spec, name=name, group=group) for spec in specs
        )
        return _select_field_prediction(predictions, name=name, group=group)

    def _recognize_crop(
        self,
        image: RgbImage,
        spec: _CropSpec,
        *,
        name: str,
        group: str,
    ) -> PokerLegendsNumberPrediction:
        crop = _crop(image, spec.rect, pad=spec.pad)
        whitelist = _BUTTON_WHITELIST if group == "buttons" else _NUMERIC_WHITELIST
        raw = _best_numeric_text(crop, whitelist=whitelist)
        numbers = tuple(_numbers_from_text(raw))
        normalized = _normalized_number(raw, numbers=numbers)
        base, overlay, total = _number_components(raw, numbers=numbers)
        field_normalized = _field_normalized_number(
            group=group,
            name=name,
            normalized=normalized,
            base=base,
            overlay=overlay,
        )
        visible = bool(_normalize_text(raw))
        return PokerLegendsNumberPrediction(
            name=name,
            group=group,
            visible=visible,
            raw=raw,
            numbers=numbers,
            first_number=numbers[0] if numbers else None,
            sum_number=sum(numbers) if len(numbers) >= 2 else None,
            normalized_number=field_normalized,
            confidence=_confidence(raw, numbers),
            base_number=base,
            overlay_number=overlay,
            total_number=total,
            crop_variant=spec.variant,
            roi_rect=(spec.rect.x, spec.rect.y, spec.rect.width, spec.rect.height),
        )


def build_poker_legends_number_ocr_report(
    annotation_paths: Sequence[str | Path],
    *,
    image_root: str | Path,
    truth_dir: str | Path | None = None,
    output_dir: str | Path,
) -> dict[str, object]:
    annotations = [Path(path) for path in annotation_paths]
    if not annotations:
        raise ValueError("at least one annotation path is required")
    recognizer = PokerLegendsNumberRecognizer()
    image_base = Path(image_root)
    output = Path(output_dir)
    result_dir = output / "number_ocr_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    compared = 0
    correct = 0
    missing = 0
    mismatch = 0
    for annotation_path in sorted(annotations):
        annotation = _read_json_object(annotation_path)
        frame_id = str(annotation.get("frame_id") or annotation_path.stem)
        truth = None if truth_dir is None else _optional_truth(Path(truth_dir) / f"{frame_id}.json")
        predictions = recognizer.recognize(image_base / str(annotation["image"]), annotation)
        (result_dir / f"{frame_id}.json").write_text(
            json.dumps(
                {
                    "frame_id": frame_id,
                    "image": str(annotation["image"]),
                    "predictions": [prediction.to_dict() for prediction in predictions],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        for prediction in predictions:
            expected = _expected_number(truth, prediction)
            status = "not_compared"
            if expected is not None:
                compared += 1
                if prediction.normalized_number is None:
                    missing += 1
                    status = "missing"
                elif prediction.normalized_number == expected:
                    correct += 1
                    status = "match"
                else:
                    mismatch += 1
                    status = "mismatch"
            rows.append(
                {
                    "frame_id": frame_id,
                    "group": prediction.group,
                    "name": prediction.name,
                    "expected": expected,
                    "observed": prediction.normalized_number,
                    "raw": prediction.raw,
                    "numbers": list(prediction.numbers),
                    "status": status,
                    "confidence": prediction.confidence,
                }
            )
    summary: dict[str, object] = {
        "schema_version": 1,
        "frames": len(annotations),
        "predictions": len(rows),
        "compared": compared,
        "correct": correct,
        "missing": missing,
        "mismatch": mismatch,
        "accuracy": correct / compared if compared else 1.0,
        "rows": rows,
        "result_dir": str(result_dir.relative_to(output)),
    }
    (output / "number_ocr_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_number_ocr_report(output / "number_ocr_report.md", summary)
    return summary


def build_poker_legends_number_crop_dataset(
    annotation_paths: Sequence[str | Path],
    *,
    image_root: str | Path,
    output_dir: str | Path,
    truth_dir: str | Path | None = None,
    text_names: Sequence[str] = ("pot", "hero_stack", "hero_current_bet", "right_top_stack"),
    button_names: Sequence[str] = ("primary_left",),
) -> dict[str, object]:
    annotations = [Path(path) for path in annotation_paths]
    if not annotations:
        raise ValueError("at least one annotation path is required")
    image_base = Path(image_root)
    output = Path(output_dir)
    crop_root = output / "crops"
    crop_root.mkdir(parents=True, exist_ok=True)
    selected_text_names = set(text_names)
    selected_button_names = set(button_names)
    rows: list[dict[str, object]] = []
    screen_kind_counts: dict[str, int] = {}
    field_counts: dict[str, int] = {}
    variant_counts: dict[str, int] = {}
    labeled_crops = 0
    for annotation_path in sorted(annotations):
        annotation = _read_json_object(annotation_path)
        frame_id = str(annotation.get("frame_id") or annotation_path.stem)
        truth = None if truth_dir is None else _optional_truth(Path(truth_dir) / f"{frame_id}.json")
        truth_screen = _truth_screen(truth)
        screen_kind = truth_screen.get("kind")
        if isinstance(screen_kind, str) and screen_kind:
            screen_kind_counts[screen_kind] = screen_kind_counts.get(screen_kind, 0) + 1
        image_path = _resolve_image_path(image_base, annotation["image"])
        image = _load_rgb_image(image_path)
        regions = _region_groups(annotation)
        regions["texts"] = _with_canonical_text_regions(
            regions.get("texts", ()),
            selected_text_names=selected_text_names,
            image_width=int(image.shape[1]),
            image_height=int(image.shape[0]),
        )
        for group, region_names in (
            ("texts", selected_text_names),
            ("buttons", selected_button_names),
        ):
            for region in regions.get(group, ()):
                name = str(region.get("name") or "")
                if name not in region_names:
                    continue
                rect = _rect_from_region(region)
                for spec in _dataset_crop_specs_for_region(rect, group=group, name=name):
                    crop = _crop(image, spec.rect, pad=spec.pad)
                    crop_path = _crop_dataset_path(
                        crop_root,
                        frame_id=frame_id,
                        group=group,
                        name=name,
                        variant=spec.variant,
                    )
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                    truth_label = _truth_number_label(truth, group=group, name=name)
                    truth_number = _optional_int(truth_label.get("normalized_number"))
                    canonical_text = _canonical_chip_text(truth_label.get("value"), truth_number)
                    if truth_number is not None:
                        labeled_crops += 1
                    field_key = f"{group}:{name}"
                    field_counts[field_key] = field_counts.get(field_key, 0) + 1
                    variant_key = f"{field_key}:{spec.variant}"
                    variant_counts[variant_key] = variant_counts.get(variant_key, 0) + 1
                    rows.append(
                        {
                            "frame_id": frame_id,
                            "image": str(image_path),
                            "group": group,
                            "name": name,
                            "role": _number_crop_role(group, name),
                            "crop_variant": spec.variant,
                            "crop_path": str(crop_path.relative_to(output)),
                            "roi_rect": [
                                spec.rect.x,
                                spec.rect.y,
                                spec.rect.width,
                                spec.rect.height,
                            ],
                            "pad": spec.pad,
                            "screen_kind": screen_kind,
                            "blocking_reason": truth_screen.get("blocking_reason"),
                            "truth_visible": truth_label.get("visible"),
                            "truth_value": truth_label.get("value"),
                            "truth_normalized_number": truth_number,
                            "truth_canonical_text": canonical_text,
                            "truth_tokens": (
                                list(canonical_text) if canonical_text is not None else []
                            ),
                            "truth_chip_numbers": list(
                                parse_poker_legends_chip_numbers(canonical_text)
                            )
                            if canonical_text is not None
                            else [],
                        }
                    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "frames": len(annotations),
        "crops": len(rows),
        "labeled_crops": labeled_crops,
        "crop_root": str(crop_root.relative_to(output)),
        "screen_kind_counts": dict(sorted(screen_kind_counts.items())),
        "field_counts": dict(sorted(field_counts.items())),
        "variant_counts": dict(sorted(variant_counts.items())),
        "rows": rows,
    }
    (output / "number_crop_dataset_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_number_crop_dataset_report(output / "number_crop_dataset_report.md", summary)
    return summary


def evaluate_poker_legends_number_crop_dataset(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    accepted_confidence: float = 0.70,
    max_crops: int | None = None,
) -> dict[str, object]:
    manifest_file = Path(manifest_path)
    manifest = _read_json_object(manifest_file)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    overall = _new_number_crop_eval_stats()
    by_field: dict[str, dict[str, int]] = {}
    by_variant: dict[str, dict[str, int]] = {}
    by_field_variant: dict[str, dict[str, int]] = {}
    dataset_rows = _mapping_sequence(manifest.get("rows"))
    selected_rows = dataset_rows[:max_crops] if max_crops is not None else dataset_rows
    for dataset_row in selected_rows:
        group = str(dataset_row.get("group") or "")
        name = str(dataset_row.get("name") or "")
        variant = str(dataset_row.get("crop_variant") or "default")
        crop_path = _resolve_crop_path(manifest_file.parent, dataset_row.get("crop_path"))
        prediction = _recognize_crop_file(
            crop_path,
            group=group,
            name=name,
            variant=variant,
            roi_rect=_optional_rect_tuple(dataset_row.get("roi_rect")),
        )
        expected = _optional_int(dataset_row.get("truth_normalized_number"))
        status = _number_crop_eval_status(prediction.normalized_number, expected)
        accepted = _number_crop_prediction_accepted(
            prediction,
            accepted_confidence=accepted_confidence,
        )
        eval_row = {
            "frame_id": dataset_row.get("frame_id"),
            "group": group,
            "name": name,
            "role": dataset_row.get("role"),
            "crop_variant": variant,
            "crop_path": str(crop_path),
            "screen_kind": dataset_row.get("screen_kind"),
            "expected": expected,
            "observed": prediction.normalized_number,
            "raw": prediction.raw,
            "numbers": list(prediction.numbers),
            "base_number": prediction.base_number,
            "overlay_number": prediction.overlay_number,
            "total_number": prediction.total_number,
            "confidence": prediction.confidence,
            "status": status,
            "accepted": accepted,
        }
        rows.append(eval_row)
        for stats in (
            overall,
            by_field.setdefault(f"{group}:{name}", _new_number_crop_eval_stats()),
            by_variant.setdefault(variant, _new_number_crop_eval_stats()),
            by_field_variant.setdefault(
                f"{group}:{name}:{variant}",
                _new_number_crop_eval_stats(),
            ),
        ):
            _update_number_crop_eval_stats(
                stats,
                status=status,
                expected=expected,
                accepted=accepted,
            )
    summary: dict[str, object] = {
        "schema_version": 1,
        "manifest": str(manifest_file),
        "available_crops": len(dataset_rows),
        "evaluated_crops": len(selected_rows),
        "max_crops": max_crops,
        "accepted_confidence": accepted_confidence,
        "overall": _finalize_number_crop_eval_stats(overall),
        "by_field": {
            key: _finalize_number_crop_eval_stats(stats)
            for key, stats in sorted(by_field.items())
        },
        "by_variant": {
            key: _finalize_number_crop_eval_stats(stats)
            for key, stats in sorted(by_variant.items())
        },
        "by_field_variant": {
            key: _finalize_number_crop_eval_stats(stats)
            for key, stats in sorted(by_field_variant.items())
        },
        "rows": rows,
    }
    review_queue = _number_crop_ocr_review_queue(rows)
    summary["review_queue_counts"] = _number_crop_ocr_review_queue_counts(review_queue)
    summary["review_queue"] = review_queue
    (output / "number_crop_ocr_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "number_crop_ocr_review_queue.json").write_text(
        json.dumps(
            {
                "manifest": str(manifest_file),
                "counts": summary["review_queue_counts"],
                "rows": review_queue,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_number_crop_ocr_report(output / "number_crop_ocr_report.md", summary)
    _write_number_crop_ocr_review_queue_report(
        output / "number_crop_ocr_review_queue.md",
        summary,
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Poker Legends numeric OCR on text/button ROIs."
    )
    parser.add_argument(
        "annotations",
        nargs="+",
        help="Poker Legends layout annotation JSON files.",
    )
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--truth-dir")
    parser.add_argument("--out", required=True)
    return parser


def build_crop_dataset_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Poker Legends numeric ROI crop dataset for OCR training/review."
    )
    parser.add_argument(
        "annotations",
        nargs="+",
        help="Poker Legends layout annotation JSON files.",
    )
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--truth-dir")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--text-name",
        action="append",
        dest="text_names",
        help="Text ROI name to export. Repeat to override defaults.",
    )
    parser.add_argument(
        "--button-name",
        action="append",
        dest="button_names",
        help="Button ROI name to export. Repeat to override defaults.",
    )
    return parser


def build_crop_ocr_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Poker Legends numeric OCR on an exported crop dataset."
    )
    parser.add_argument("manifest", help="number_crop_dataset_manifest.json")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--accepted-confidence",
        type=float,
        default=0.70,
        help="Confidence threshold for accepted precision reporting.",
    )
    parser.add_argument(
        "--max-crops",
        type=int,
        help="Evaluate only the first N manifest rows for quick offline smoke runs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = build_poker_legends_number_ocr_report(
        args.annotations,
        image_root=args.image_root,
        truth_dir=args.truth_dir,
        output_dir=args.out,
    )
    print(json.dumps(_stdout_summary(summary), indent=2, sort_keys=True))


def crop_dataset_main(argv: Sequence[str] | None = None) -> None:
    args = build_crop_dataset_arg_parser().parse_args(argv)
    summary = build_poker_legends_number_crop_dataset(
        args.annotations,
        image_root=args.image_root,
        truth_dir=args.truth_dir,
        output_dir=args.out,
        text_names=tuple(
            args.text_names or ("pot", "hero_stack", "hero_current_bet", "right_top_stack")
        ),
        button_names=tuple(args.button_names or ("primary_left",)),
    )
    print(json.dumps(_crop_dataset_stdout_summary(summary), indent=2, sort_keys=True))


def crop_ocr_main(argv: Sequence[str] | None = None) -> None:
    args = build_crop_ocr_arg_parser().parse_args(argv)
    summary = evaluate_poker_legends_number_crop_dataset(
        args.manifest,
        output_dir=args.out,
        accepted_confidence=args.accepted_confidence,
        max_crops=args.max_crops,
    )
    print(json.dumps(_crop_ocr_stdout_summary(summary), indent=2, sort_keys=True))


def _default_crop_spec(rect: ScreenRect, *, group: str) -> _CropSpec:
    default_pad = 4 if group == "texts" else 0
    return _CropSpec("default", rect, default_pad)


def _fallback_crop_specs_for_prediction(
    default: PokerLegendsNumberPrediction,
    rect: ScreenRect,
    *,
    name: str,
    group: str,
) -> tuple[_CropSpec, ...]:
    if group != "texts" or name != "hero_stack" or default.confidence >= 0.70:
        return ()
    specs: list[_CropSpec] = []
    if _looks_like_left_edge_stack_pollution(default.raw):
        specs.append(_CropSpec("hero_stack_no_pad", rect, 0))
    if _looks_like_right_edge_stack_pollution(default.raw):
        specs.append(
            _CropSpec(
                "hero_stack_trim_right_16",
                ScreenRect(rect.x, rect.y, max(1, rect.width - 16), rect.height),
                0,
            )
        )
    if not specs:
        specs.append(_CropSpec("hero_stack_no_pad", rect, 0))
    return tuple(specs)


def _dataset_crop_specs_for_region(
    rect: ScreenRect,
    *,
    group: str,
    name: str,
) -> tuple[_CropSpec, ...]:
    default = _default_crop_spec(rect, group=group)
    if group != "texts":
        return (default,)
    if name == "hero_stack":
        return (
            default,
            _CropSpec("hero_stack_no_pad", rect, 0),
            _CropSpec(
                "hero_stack_trim_right_16",
                ScreenRect(rect.x, rect.y, max(1, rect.width - 16), rect.height),
                0,
            ),
        )
    if name == "right_top_stack":
        return (
            default,
            _CropSpec("right_top_stack_no_pad", rect, 0),
            _CropSpec(
                "right_top_stack_trim_left_16",
                ScreenRect(rect.x + 16, rect.y, max(1, rect.width - 16), rect.height),
                0,
            ),
        )
    return (default,)


def _select_field_prediction(
    predictions: tuple[PokerLegendsNumberPrediction, ...],
    *,
    name: str,
    group: str,
) -> PokerLegendsNumberPrediction:
    if not predictions:
        raise ValueError("at least one prediction is required")
    default = predictions[0]
    if group != "texts" or name != "hero_stack":
        return default
    if default.confidence >= 0.70:
        return default
    variant_by_name = {prediction.crop_variant: prediction for prediction in predictions}
    if _looks_like_right_edge_stack_pollution(default.raw):
        trim_right = variant_by_name.get("hero_stack_trim_right_16")
        if trim_right is not None and _is_safe_stack_variant(trim_right):
            return trim_right
    if _looks_like_left_edge_stack_pollution(default.raw):
        no_pad = variant_by_name.get("hero_stack_no_pad")
        if no_pad is not None and _is_safe_stack_variant(no_pad):
            return no_pad
    no_pad = variant_by_name.get("hero_stack_no_pad")
    if no_pad is not None and _is_safe_stack_variant(no_pad):
        return no_pad
    return default


def _is_safe_stack_variant(prediction: PokerLegendsNumberPrediction) -> bool:
    if prediction.normalized_number is None or prediction.confidence < 0.70:
        return False
    if prediction.normalized_number < 0 or prediction.normalized_number > 10_000:
        return False
    return not _looks_fragmented_numeric_ocr(prediction.raw)


def _looks_like_left_edge_stack_pollution(raw: str) -> bool:
    normalized = _normalize_text(raw).translate(str.maketrans("OoI|", "0011"))
    if "$" in normalized and re.search(r"[0-9KkMm]", normalized.rsplit("$", maxsplit=1)[0]):
        return True
    return "$" not in normalized and bool(re.search(r"\d", normalized))


def _looks_like_right_edge_stack_pollution(raw: str) -> bool:
    normalized = _normalize_text(raw)
    raw_source = _amount_source(normalized.translate(str.maketrans("OoI|", "0011")))
    if re.search(r"\+\s*\d+(?:\s+\d+)+", raw_source):
        return True
    return bool(re.search(r"\d[.,]?\s+\d$", normalized))


def _best_numeric_text(image: RgbImage, *, whitelist: str) -> str:
    candidates: list[str] = []
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    for prepared in _prepared_images(gray):
        for psm in (7, 8, 11):
            config = f"--psm {psm} -c tessedit_char_whitelist={whitelist}"
            text = cast(str, pytesseract.image_to_string(prepared, config=config)).strip()
            if text:
                candidates.append(text)
    if not candidates:
        return ""
    return max(
        candidates,
        key=lambda text: (len(_numbers_from_text(text)), len(_normalize_text(text))),
    )


def _prepared_images(gray: ImageArray) -> tuple[ImageArray, ...]:
    scaled = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    equalized = cv2.equalizeHist(scaled)
    _, binary = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, inverted_binary = cv2.threshold(
        equalized,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    return (scaled, equalized, binary, inverted_binary)


def _numbers_from_text(text: str) -> list[int]:
    normalized = _normalize_numeric_text(text)
    normalized = _amount_source(normalized)
    values: list[int] = []
    for match in re.finditer(r"(\d+(?:[.,]\d+)?)([KkMm]?)", normalized):
        token = match.group(1)
        suffix = match.group(2).lower()
        if suffix:
            value = float(token.replace(",", "."))
            multiplier = 1_000_000 if suffix == "m" else 1_000
            values.append(int(round(value * multiplier)))
        else:
            values.append(int(re.sub(r"[.,]", "", token)))
    return values


def _normalized_number(text: str, *, numbers: tuple[int, ...]) -> int | None:
    if not numbers:
        return None
    normalized = _normalize_numeric_text(text)
    if "+" in normalized and len(numbers) >= 2:
        return sum(numbers)
    return numbers[0]


def _number_components(
    text: str,
    *,
    numbers: tuple[int, ...],
) -> tuple[int | None, int | None, int | None]:
    if not numbers:
        return None, None, None
    normalized = _normalize_numeric_text(text)
    total = _normalized_number(text, numbers=numbers)
    if "+" in normalized and len(numbers) >= 2:
        base = numbers[0]
        overlay = sum(numbers[1:])
        return base, overlay, total
    return total, None, total


def _field_normalized_number(
    *,
    group: str,
    name: str,
    normalized: int | None,
    base: int | None,
    overlay: int | None,
) -> int | None:
    if group == "texts" and _is_stack_field_name(name) and overlay is not None:
        return base
    return normalized


def _is_stack_field_name(name: str) -> bool:
    return name == "opponent_stack" or name.endswith("_stack")


def _normalize_numeric_text(text: str) -> str:
    normalized = _normalize_text(text).translate(str.maketrans("OoI|", "0011"))
    normalized = re.sub(r"\$\s+", "$", normalized)
    normalized = re.sub(r"(?<=[\d$.,+])\s+(?=\d)", "", normalized)
    normalized = re.sub(r"(?<=\d)\s+(?=[.,+])", "", normalized)
    normalized = re.sub(r"(?<=[.,+])\s+(?=\d)", "", normalized)
    return normalized


def _confidence(raw: str, numbers: tuple[int, ...]) -> float:
    if not numbers:
        return 0.25 if _normalize_text(raw) else 0.0
    compact = _amount_source(_normalize_numeric_text(raw))
    compact = re.sub(r"[^0-9KkMm+.,]", "", compact).strip(".,+")
    confidence = 0.70
    if re.fullmatch(r"\d+(?:[KkMm])?", compact):
        confidence = 0.90
    elif re.fullmatch(r"\d{1,3}(?:,\d{3})+", compact):
        confidence = 0.90
    elif re.fullmatch(r"\d+[.,]\d{1,2}[KkMm]", compact):
        confidence = 0.86
    elif re.fullmatch(r"\d+[.,]\d+", compact):
        confidence = 0.70
    elif "+" in compact and all(_is_clean_chip_token(part) for part in compact.split("+")):
        confidence = 0.82
    elif len(numbers) >= 2:
        confidence = 0.65
    if _looks_fragmented_numeric_ocr(raw):
        return min(confidence, 0.65)
    return confidence


def _looks_fragmented_numeric_ocr(raw: str) -> bool:
    normalized = _normalize_text(raw).translate(str.maketrans("OoI|", "0011"))
    if "$" in normalized and re.search(r"[0-9KkMm]", normalized.rsplit("$", maxsplit=1)[0]):
        return True
    source = _amount_source(normalized)
    if re.match(r"^[KkMm]\D*\d", source):
        return True
    if "$" not in normalized and re.fullmatch(r"\d+,", source):
        return True
    if re.search(r"(?<=\d)\s+[KkMm]\b", source):
        return True
    if "+" in source:
        if source.count("+") >= 2:
            return True
        compact_source = re.sub(r"\s+", "", source)
        plus_parts = compact_source.split("+")
        if "$" not in normalized and "," not in compact_source and plus_parts[0].isdigit():
            if len(plus_parts[0]) >= 4:
                return True
        if (
            "$" not in normalized
            and "," not in compact_source
            and len(plus_parts) == 2
            and plus_parts[0].isdigit()
            and plus_parts[1].isdigit()
            and (len(plus_parts[0]) >= 4 or int(plus_parts[0]) < 100)
            and len(plus_parts[1]) <= 3
        ):
            return True
        return bool(re.search(r"\+\s*\d+(?:\s+\d+)+", source))
    digit_runs = re.findall(r"\d+", source)
    return len(digit_runs) >= 2 and all(len(run) <= 2 for run in digit_runs)


def _is_clean_chip_token(token: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:\d+|\d{1,3}(?:,\d{3})+|\d+[.,]\d{1,2}[KkMm]|\d+[KkMm])",
            token,
        )
    )


def _amount_source(normalized_text: str) -> str:
    if "$" in normalized_text:
        return normalized_text.rsplit("$", maxsplit=1)[1]
    return normalized_text


def _expected_number(
    truth: Mapping[str, object] | None,
    prediction: PokerLegendsNumberPrediction,
) -> int | None:
    if truth is None:
        return None
    if prediction.group == "texts":
        for text in _mapping_sequence(truth.get("texts")):
            if str(text.get("name") or "") == prediction.name and bool(text.get("visible", True)):
                return _optional_int(text.get("normalized_number"))
    if prediction.group == "buttons":
        for button in _mapping_sequence(truth.get("buttons")):
            if str(button.get("name") or "") == prediction.name and bool(
                button.get("visible", True)
            ):
                label = button.get("label")
                if isinstance(label, str):
                    numbers = tuple(_numbers_from_text(label))
                    return _normalized_number(label, numbers=numbers)
    return None


def _truth_number_label(
    truth: Mapping[str, object] | None,
    *,
    group: str,
    name: str,
) -> dict[str, object]:
    if truth is None:
        return {"visible": None, "value": None, "normalized_number": None}
    if group == "texts":
        for text in _mapping_sequence(truth.get("texts")):
            if str(text.get("name") or "") == name:
                value = text.get("value")
                normalized_number = _optional_int(text.get("normalized_number"))
                if _is_stack_field_name(name) and isinstance(value, str):
                    stack_components = parse_poker_legends_chip_components(value)
                    normalized_number = stack_components["base_number"]
                elif normalized_number is None and isinstance(value, str):
                    normalized_number = parse_poker_legends_chip_amount(value)
                return {
                    "visible": bool(text.get("visible", True)),
                    "value": value,
                    "normalized_number": normalized_number,
                }
        if name == "hero_stack":
            stack = _truth_unambiguous_hero_stack_number(truth)
            if stack is not None:
                return {"visible": True, "value": f"${stack}", "normalized_number": stack}
        if name == "hero_current_bet":
            committed = _truth_hero_seat_number(truth, "committed")
            if committed is not None and committed > 0:
                return {
                    "visible": True,
                    "value": f"${committed}",
                    "normalized_number": committed,
                }
    if group == "buttons":
        for button in _mapping_sequence(truth.get("buttons")):
            if str(button.get("name") or "") == name:
                label = button.get("label")
                return {
                    "visible": bool(button.get("visible", True)),
                    "value": label,
                    "normalized_number": parse_poker_legends_chip_amount(label)
                    if isinstance(label, str)
                    else None,
                }
    return {"visible": None, "value": None, "normalized_number": None}


def _truth_hero_seat_number(truth: Mapping[str, object], field_name: str) -> int | None:
    for seat in _mapping_sequence(truth.get("seats")):
        if not bool(seat.get("visible", True)):
            continue
        if str(seat.get("name") or "").lower() != "hero":
            continue
        return _optional_int(seat.get(field_name))
    return None


def _truth_unambiguous_hero_stack_number(truth: Mapping[str, object]) -> int | None:
    for seat in _mapping_sequence(truth.get("seats")):
        if not bool(seat.get("visible", True)):
            continue
        if str(seat.get("name") or "").lower() != "hero":
            continue
        stack = _optional_int(seat.get("stack"))
        committed = _optional_int(seat.get("committed"))
        if stack is not None and committed == 0:
            return stack
    return None


def _truth_screen(truth: Mapping[str, object] | None) -> Mapping[str, object]:
    screen = None if truth is None else truth.get("screen")
    return screen if isinstance(screen, Mapping) else {}


def _canonical_chip_text(value: object, normalized_number: int | None) -> str | None:
    if isinstance(value, str) and value.strip():
        normalized = _normalize_numeric_text(value)
        source = _amount_source(normalized)
        if "$" in normalized:
            source = f"${source}"
        source = source.strip()
        return source or None
    if normalized_number is not None:
        return str(normalized_number)
    return None


def _number_crop_role(group: str, name: str) -> str:
    if group == "buttons":
        return "call_amount" if name == "primary_left" else "button_amount"
    if name == "pot":
        return "pot"
    if name == "hero_current_bet":
        return "hero_current_bet"
    if name == "hero_stack":
        return "hero_stack"
    if name.endswith("_stack") or name == "opponent_stack":
        return "seat_stack"
    return name


def _crop_dataset_path(
    crop_root: Path,
    *,
    frame_id: str,
    group: str,
    name: str,
    variant: str,
) -> Path:
    field = f"{group}_{name}"
    return crop_root / field / f"{frame_id}__{field}__{variant}.png"


def _recognize_crop_file(
    crop_path: Path,
    *,
    group: str,
    name: str,
    variant: str,
    roi_rect: tuple[int, int, int, int] | None,
) -> PokerLegendsNumberPrediction:
    image = _load_rgb_image(crop_path)
    whitelist = _BUTTON_WHITELIST if group == "buttons" else _NUMERIC_WHITELIST
    raw = _best_numeric_text(image, whitelist=whitelist)
    numbers = tuple(_numbers_from_text(raw))
    normalized = _normalized_number(raw, numbers=numbers)
    base, overlay, total = _number_components(raw, numbers=numbers)
    return PokerLegendsNumberPrediction(
        name=name,
        group=group,
        visible=bool(_normalize_text(raw)),
        raw=raw,
        numbers=numbers,
        first_number=numbers[0] if numbers else None,
        sum_number=sum(numbers) if len(numbers) >= 2 else None,
        normalized_number=_field_normalized_number(
            group=group,
            name=name,
            normalized=normalized,
            base=base,
            overlay=overlay,
        ),
        confidence=_confidence(raw, numbers),
        base_number=base,
        overlay_number=overlay,
        total_number=total,
        crop_variant=variant,
        roi_rect=roi_rect,
    )


def _number_crop_eval_status(observed: int | None, expected: int | None) -> str:
    if expected is None:
        return "not_labeled"
    if observed is None:
        return "missing"
    if observed == expected:
        return "match"
    return "mismatch"


def _number_crop_prediction_accepted(
    prediction: PokerLegendsNumberPrediction,
    *,
    accepted_confidence: float,
) -> bool:
    if prediction.normalized_number is None or prediction.confidence < accepted_confidence:
        return False
    if prediction.group == "texts" and _is_stack_field_name(prediction.name):
        return _is_safe_stack_variant(prediction)
    return True


def _new_number_crop_eval_stats() -> dict[str, int]:
    return {
        "crops": 0,
        "labeled": 0,
        "correct": 0,
        "missing": 0,
        "mismatch": 0,
        "accepted": 0,
        "accepted_labeled": 0,
        "accepted_correct": 0,
        "accepted_wrong": 0,
    }


def _update_number_crop_eval_stats(
    stats: dict[str, int],
    *,
    status: str,
    expected: int | None,
    accepted: bool,
) -> None:
    stats["crops"] += 1
    if expected is not None:
        stats["labeled"] += 1
        if status == "match":
            stats["correct"] += 1
        elif status == "missing":
            stats["missing"] += 1
        elif status == "mismatch":
            stats["mismatch"] += 1
    if accepted:
        stats["accepted"] += 1
        if expected is not None:
            stats["accepted_labeled"] += 1
            if status == "match":
                stats["accepted_correct"] += 1
            elif status == "mismatch":
                stats["accepted_wrong"] += 1


def _finalize_number_crop_eval_stats(stats: Mapping[str, int]) -> dict[str, object]:
    labeled = stats["labeled"]
    accepted_labeled = stats["accepted_labeled"]
    return {
        "crops": stats["crops"],
        "labeled": labeled,
        "correct": stats["correct"],
        "missing": stats["missing"],
        "mismatch": stats["mismatch"],
        "accuracy": stats["correct"] / labeled if labeled else None,
        "accepted": stats["accepted"],
        "accepted_labeled": accepted_labeled,
        "accepted_correct": stats["accepted_correct"],
        "accepted_wrong": stats["accepted_wrong"],
        "accepted_precision": (
            stats["accepted_correct"] / accepted_labeled if accepted_labeled else None
        ),
    }


def _number_crop_ocr_review_queue(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    queue: list[dict[str, object]] = []
    for row in rows:
        category = _number_crop_ocr_review_category(row)
        if category is None:
            continue
        queue.append(
            {
                "category": category,
                "priority": _number_crop_ocr_review_priority(category),
                "frame_id": row.get("frame_id"),
                "group": row.get("group"),
                "name": row.get("name"),
                "role": row.get("role"),
                "crop_variant": row.get("crop_variant"),
                "crop_path": row.get("crop_path"),
                "screen_kind": row.get("screen_kind"),
                "expected": row.get("expected"),
                "observed": row.get("observed"),
                "raw": row.get("raw"),
                "numbers": row.get("numbers"),
                "confidence": row.get("confidence"),
                "accepted": row.get("accepted"),
            }
        )
    return sorted(
        queue,
        key=lambda item: (
            _to_int(item["priority"]),
            str(item.get("frame_id") or ""),
            str(item.get("group") or ""),
            str(item.get("name") or ""),
            str(item.get("crop_variant") or ""),
        ),
    )


def _number_crop_ocr_review_category(row: Mapping[str, object]) -> str | None:
    status = str(row.get("status") or "")
    expected = row.get("expected")
    accepted = bool(row.get("accepted"))
    if expected is not None and status == "mismatch" and accepted:
        return "accepted_wrong"
    if expected is not None and status == "missing":
        return "missing_labeled"
    if expected is not None and status == "mismatch":
        return "mismatch_labeled"
    if expected is None and accepted:
        return "accepted_unlabeled"
    return None


def _number_crop_ocr_review_priority(category: str) -> int:
    if category == "accepted_wrong":
        return 0
    if category == "missing_labeled":
        return 1
    if category == "mismatch_labeled":
        return 2
    if category == "accepted_unlabeled":
        return 3
    return 9


def _number_crop_ocr_review_queue_counts(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        category = str(row.get("category") or "")
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def _write_number_ocr_report(path: Path, summary: Mapping[str, object]) -> None:
    rows = _mapping_sequence(summary["rows"])
    lines = [
        "# Poker Legends Number OCR",
        "",
        "## Summary",
        f"- Frames: {summary['frames']}",
        f"- Predictions: {summary['predictions']}",
        f"- Compared: {summary['compared']}",
        f"- Correct: {summary['correct']}",
        f"- Accuracy: {_to_float(summary['accuracy']):.3f}",
        f"- Missing: {summary['missing']}",
        f"- Mismatch: {summary['mismatch']}",
        "",
        "## Conflicts",
    ]
    conflicts = [row for row in rows if row.get("status") not in {"match", "not_compared"}]
    if conflicts:
        for row in conflicts:
            lines.append(
                f"- `{row['frame_id']}` `{row['group']}.{row['name']}`: "
                f"{row['status']}; expected={row['expected']!r}, observed={row['observed']!r}, "
                f"raw={row['raw']!r}"
            )
    else:
        lines.append("No conflicts.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_number_crop_dataset_report(path: Path, summary: Mapping[str, object]) -> None:
    lines = [
        "# Poker Legends Number Crop Dataset",
        "",
        "## Summary",
        f"- Frames: {summary['frames']}",
        f"- Crops: {summary['crops']}",
        f"- Labeled crops: {summary['labeled_crops']}",
        f"- Crop root: `{summary['crop_root']}`",
        "",
        "## Screen Kind Counts",
        *_count_lines(summary.get("screen_kind_counts")),
        "",
        "## Field Counts",
        *_count_lines(summary.get("field_counts")),
        "",
        "## Variant Counts",
        *_count_lines(summary.get("variant_counts")),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_number_crop_ocr_report(path: Path, summary: Mapping[str, object]) -> None:
    overall = cast(Mapping[str, object], summary["overall"])
    rows = _mapping_sequence(summary["rows"])
    accepted_wrong = [
        row
        for row in rows
        if bool(row.get("accepted")) and row.get("status") == "mismatch"
    ]
    mismatches = [
        row
        for row in rows
        if row.get("status") in {"missing", "mismatch"}
        and row.get("expected") is not None
    ]
    lines = [
        "# Poker Legends Number Crop OCR",
        "",
        "## Summary",
        f"- Manifest: `{summary['manifest']}`",
        f"- Evaluated crops: {summary['evaluated_crops']} / {summary['available_crops']}",
        f"- Max crops: {summary['max_crops']}",
        f"- Accepted confidence: {summary['accepted_confidence']}",
        f"- Crops: {overall['crops']}",
        f"- Labeled: {overall['labeled']}",
        f"- Correct: {overall['correct']}",
        f"- Accuracy: {_optional_ratio(overall.get('accuracy'))}",
        f"- Missing: {overall['missing']}",
        f"- Mismatch: {overall['mismatch']}",
        f"- Accepted labeled: {overall['accepted_labeled']}",
        f"- Accepted correct: {overall['accepted_correct']}",
        f"- Accepted wrong: {overall['accepted_wrong']}",
        f"- Accepted precision: {_optional_ratio(overall.get('accepted_precision'))}",
        f"- Review queue: {len(_mapping_sequence(summary.get('review_queue')))}",
        "",
        "## By Field Variant",
    ]
    for key, stats in cast(Mapping[str, object], summary["by_field_variant"]).items():
        stat_map = cast(Mapping[str, object], stats)
        lines.append(
            f"- `{key}`: labeled={stat_map['labeled']}, correct={stat_map['correct']}, "
            f"missing={stat_map['missing']}, mismatch={stat_map['mismatch']}, "
            f"accepted={stat_map['accepted_labeled']}, "
            f"accepted_wrong={stat_map['accepted_wrong']}, "
            f"accepted_precision={_optional_ratio(stat_map.get('accepted_precision'))}"
        )
    lines.extend(["", "## Accepted Wrong"])
    if accepted_wrong:
        for row in accepted_wrong:
            lines.append(_crop_ocr_conflict_line(row))
    else:
        lines.append("No accepted wrong labeled crops.")
    lines.extend(["", "## Missing Or Mismatch"])
    if mismatches:
        for row in mismatches[:50]:
            lines.append(_crop_ocr_conflict_line(row))
        if len(mismatches) > 50:
            lines.append(f"- ... {len(mismatches) - 50} more")
    else:
        lines.append("No missing or mismatched labeled crops.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_number_crop_ocr_review_queue_report(
    path: Path,
    summary: Mapping[str, object],
) -> None:
    queue = _mapping_sequence(summary.get("review_queue"))
    counts = cast(Mapping[str, object], summary.get("review_queue_counts") or {})
    lines = [
        "# Poker Legends Number Crop OCR Review Queue",
        "",
        "## Summary",
        f"- Manifest: `{summary['manifest']}`",
        f"- Evaluated crops: {summary['evaluated_crops']} / {summary['available_crops']}",
        "",
        "## Counts",
        *_count_lines(counts),
        "",
        "## Rows",
    ]
    if queue:
        for row in queue[:100]:
            lines.append(_crop_ocr_review_queue_line(row))
        if len(queue) > 100:
            lines.append(f"- ... {len(queue) - 100} more")
    else:
        lines.append("No review rows.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _crop_ocr_review_queue_line(row: Mapping[str, object]) -> str:
    return (
        f"- `{row.get('category')}` `{row.get('frame_id')}` "
        f"`{row.get('group')}.{row.get('name')}` `{row.get('crop_variant')}`: "
        f"expected={row.get('expected')!r}, observed={row.get('observed')!r}, "
        f"conf={_to_float(row.get('confidence')):.2f}, raw={row.get('raw')!r}, "
        f"crop=`{row.get('crop_path')}`"
    )


def _crop_ocr_conflict_line(row: Mapping[str, object]) -> str:
    return (
        f"- `{row.get('frame_id')}` `{row.get('group')}.{row.get('name')}` "
        f"`{row.get('crop_variant')}`: {row.get('status')}; "
        f"expected={row.get('expected')!r}, observed={row.get('observed')!r}, "
        f"conf={_to_float(row.get('confidence')):.2f}, raw={row.get('raw')!r}"
    )


def _optional_ratio(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{_to_float(value):.3f}"


def _stdout_summary(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "frames": summary["frames"],
        "predictions": summary["predictions"],
        "compared": summary["compared"],
        "accuracy": summary["accuracy"],
        "report": "number_ocr_report.md",
    }


def _crop_dataset_stdout_summary(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "frames": summary["frames"],
        "crops": summary["crops"],
        "labeled_crops": summary["labeled_crops"],
        "manifest": "number_crop_dataset_manifest.json",
        "report": "number_crop_dataset_report.md",
    }


def _crop_ocr_stdout_summary(summary: Mapping[str, object]) -> dict[str, object]:
    overall = cast(Mapping[str, object], summary["overall"])
    return {
        "available_crops": summary["available_crops"],
        "evaluated_crops": summary["evaluated_crops"],
        "max_crops": summary["max_crops"],
        "crops": overall["crops"],
        "labeled": overall["labeled"],
        "accuracy": overall["accuracy"],
        "accepted_labeled": overall["accepted_labeled"],
        "accepted_precision": overall["accepted_precision"],
        "accepted_wrong": overall["accepted_wrong"],
        "review_queue": len(_mapping_sequence(summary.get("review_queue"))),
        "report": "number_crop_ocr_report.md",
        "review_report": "number_crop_ocr_review_queue.md",
    }


def _count_lines(value: object) -> list[str]:
    if not isinstance(value, Mapping) or not value:
        return ["- none"]
    return [f"- {key}: {count}" for key, count in sorted(value.items())]


def _region_groups(annotation: Mapping[str, object]) -> dict[str, list[Mapping[str, object]]]:
    raw = annotation.get("regions")
    if not isinstance(raw, Mapping):
        return {}
    groups: dict[str, list[Mapping[str, object]]] = {}
    for group, regions in raw.items():
        if isinstance(group, str):
            groups[group] = _mapping_sequence(regions)
    return groups


def _with_canonical_text_regions(
    regions: Sequence[Mapping[str, object]],
    *,
    selected_text_names: set[str],
    image_width: int,
    image_height: int,
) -> list[Mapping[str, object]]:
    merged = list(regions)
    existing = {str(region.get("name") or "") for region in merged}
    missing = selected_text_names - existing
    if not missing:
        return merged
    for region in poker_legends_layout_regions(image_width, image_height).get("texts", ()):
        if str(region.get("name") or "") in missing:
            merged.append(region)
    return merged


def _rect_from_region(region: Mapping[str, object]) -> ScreenRect:
    raw_rect = region.get("rect")
    if not isinstance(raw_rect, Mapping):
        raise ValueError(f"region has no rect: {region}")
    return ScreenRect(
        x=_to_int(raw_rect["x"]),
        y=_to_int(raw_rect["y"]),
        width=_to_int(raw_rect["width"]),
        height=_to_int(raw_rect["height"]),
    )


def _load_rgb_image(path: str | Path) -> RgbImage:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cast(RgbImage, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _resolve_image_path(image_root: Path, image_value: object) -> Path:
    image_path = Path(str(image_value))
    if image_path.is_absolute():
        return image_path
    return image_root / image_path


def _resolve_crop_path(manifest_dir: Path, crop_value: object) -> Path:
    crop_path = Path(str(crop_value))
    if crop_path.is_absolute():
        return crop_path
    return manifest_dir / crop_path


def _crop(image: RgbImage, rect: ScreenRect, *, pad: int) -> RgbImage:
    height, width = image.shape[:2]
    x = max(0, rect.x - pad)
    y = max(0, rect.y - pad)
    right = min(width, rect.x + rect.width + pad)
    bottom = min(height, rect.y + rect.height + pad)
    return image[y:bottom, x:right]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _optional_truth(path: Path) -> dict[str, object] | None:
    return _read_json_object(path) if path.exists() else None


def _read_json_object(path: str | Path) -> dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, object], data)


def _mapping_sequence(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [cast(Mapping[str, object], item) for item in value if isinstance(item, Mapping)]


def _sequence(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _to_int(value: object) -> int:
    if type(value) is int:
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip():
        return int(value)
    raise TypeError(f"expected int-like value: {value!r}")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return _to_int(value)


def _optional_rect_tuple(value: object) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    items = _sequence(value)
    if len(items) != 4:
        return None
    return (_to_int(items[0]), _to_int(items[1]), _to_int(items[2]), _to_int(items[3]))


def _to_float(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        return float(value)
    raise TypeError(f"expected float-like value: {value!r}")


if __name__ == "__main__":
    main()
