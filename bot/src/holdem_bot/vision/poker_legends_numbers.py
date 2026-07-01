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

ImageArray = NDArray[Any]
RgbImage = ImageArray

_NUMERIC_WHITELIST = "$0123456789.,+KkMm "
_BUTTON_WHITELIST = "$0123456789.,+KkMmABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "


def parse_poker_legends_chip_numbers(text: str) -> tuple[int, ...]:
    return tuple(_numbers_from_text(text))


def parse_poker_legends_chip_amount(text: str) -> int | None:
    numbers = parse_poker_legends_chip_numbers(text)
    return _normalized_number(text, numbers=numbers)


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
        )


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
        regions = _region_groups(annotation)
        predictions: list[PokerLegendsNumberPrediction] = []
        selected_text_names = set(text_names)
        selected_button_names = set(button_names)
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
        crop = _crop(image, rect, pad=4 if group == "texts" else 0)
        whitelist = _BUTTON_WHITELIST if group == "buttons" else _NUMERIC_WHITELIST
        raw = _best_numeric_text(crop, whitelist=whitelist)
        numbers = tuple(_numbers_from_text(raw))
        normalized = _normalized_number(raw, numbers=numbers)
        visible = bool(_normalize_text(raw))
        return PokerLegendsNumberPrediction(
            name=name,
            group=group,
            visible=visible,
            raw=raw,
            numbers=numbers,
            first_number=numbers[0] if numbers else None,
            sum_number=sum(numbers) if len(numbers) >= 2 else None,
            normalized_number=normalized,
            confidence=_confidence(raw, numbers),
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
        frame_id = str(annotation.get("frame_id") or Path(str(annotation["image"])).stem)
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


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = build_poker_legends_number_ocr_report(
        args.annotations,
        image_root=args.image_root,
        truth_dir=args.truth_dir,
        output_dir=args.out,
    )
    print(json.dumps(_stdout_summary(summary), indent=2, sort_keys=True))


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
    if re.fullmatch(r"\d+(?:[KkMm])?", compact):
        return 0.90
    if re.fullmatch(r"\d+[.,]\d{1,2}[KkMm]", compact):
        return 0.86
    if re.fullmatch(r"\d+[.,]\d+", compact):
        return 0.70
    if "+" in compact and all(part.isdigit() for part in compact.split("+")):
        return 0.82
    if len(numbers) >= 2:
        return 0.65
    return 0.70


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


def _stdout_summary(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "frames": summary["frames"],
        "predictions": summary["predictions"],
        "compared": summary["compared"],
        "accuracy": summary["accuracy"],
        "report": "number_ocr_report.md",
    }


def _region_groups(annotation: Mapping[str, object]) -> dict[str, list[Mapping[str, object]]]:
    raw = annotation.get("regions")
    if not isinstance(raw, Mapping):
        return {}
    groups: dict[str, list[Mapping[str, object]]] = {}
    for group, regions in raw.items():
        if isinstance(group, str):
            groups[group] = _mapping_sequence(regions)
    return groups


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


def _to_float(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        return float(value)
    raise TypeError(f"expected float-like value: {value!r}")


if __name__ == "__main__":
    main()
