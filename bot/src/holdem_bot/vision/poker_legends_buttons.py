"""Primary action button recognition for Poker Legends."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from holdem_bot.screen_state import ScreenKind
from holdem_bot.vision.annotations import ScreenRect
from holdem_bot.vision.poker_legends_screen import detect_poker_legends_screen_state

RgbImage = NDArray[np.uint8]
GrayFeature = NDArray[np.float32]

DEFAULT_NORMALIZED_WIDTH = 96
DEFAULT_NORMALIZED_HEIGHT = 64
DEFAULT_LEFT_MAX_DISTANCE = 0.050
PRIMARY_BUTTON_ACTIONS = {
    "primary_middle": "raise",
    "primary_right": "fold",
}
LEFT_SLOT = "primary_left"
BUTTON_TEMPLATE_MANIFEST = "button_template_manifest.json"


@dataclass(frozen=True, slots=True)
class PokerLegendsButtonTemplate:
    action_type: str
    frame_id: str
    slot: str
    image: str
    source_image: str
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        return cls(
            action_type=str(data["action_type"]),
            frame_id=str(data["frame_id"]),
            slot=str(data["slot"]),
            image=str(data["image"]),
            source_image=str(data["source_image"]),
            sha256=str(data["sha256"]),
        )


@dataclass(frozen=True, slots=True)
class PokerLegendsButtonTemplateManifest:
    schema_version: int
    normalized_width: int
    normalized_height: int
    left_max_distance: float
    templates: tuple[PokerLegendsButtonTemplate, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported button template manifest schema version")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "normalized_width": self.normalized_width,
            "normalized_height": self.normalized_height,
            "left_max_distance": self.left_max_distance,
            "templates": [template.to_dict() for template in self.templates],
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read_json(cls, path: str | Path) -> Self:
        data = _read_json_object(path)
        return cls(
            schema_version=_to_int(data["schema_version"]),
            normalized_width=_to_int(data["normalized_width"]),
            normalized_height=_to_int(data["normalized_height"]),
            left_max_distance=_to_float(data["left_max_distance"]),
            templates=tuple(
                PokerLegendsButtonTemplate.from_dict(item)
                for item in _mapping_sequence(data["templates"])
            ),
        )


@dataclass(frozen=True, slots=True)
class PokerLegendsButtonPrediction:
    frame_id: str
    slot: str
    visible: bool
    action_type: str | None
    confidence: float
    distance: float | None = None
    matched_template: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class PokerLegendsButtonRecognizer:
    def __init__(
        self,
        manifest: PokerLegendsButtonTemplateManifest,
        *,
        template_root: str | Path,
    ) -> None:
        self.manifest = manifest
        self.template_root = Path(template_root)
        self._templates = tuple(
            (template, _load_feature(self.template_root / template.image))
            for template in manifest.templates
        )

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> Self:
        path = Path(manifest_path)
        return cls(PokerLegendsButtonTemplateManifest.read_json(path), template_root=path.parent)

    def recognize(
        self,
        image_path: str | Path,
        annotation: Mapping[str, object],
        *,
        frame_id: str,
        exclude_frame_id: str | None = None,
    ) -> tuple[PokerLegendsButtonPrediction, ...]:
        screen_detection = detect_poker_legends_screen_state(
            image_path, layout_annotation=annotation
        )
        if screen_detection.screen.kind is not ScreenKind.ACTIONABLE_TABLE:
            return ()
        image = _load_rgb_image(image_path)
        regions = _button_regions_by_name(annotation)
        predictions: list[PokerLegendsButtonPrediction] = []
        left_region = regions.get(LEFT_SLOT)
        if left_region is not None:
            predictions.append(
                self._recognize_left_button(
                    image,
                    left_region,
                    frame_id=frame_id,
                    exclude_frame_id=exclude_frame_id,
                )
            )
        for slot, action_type in PRIMARY_BUTTON_ACTIONS.items():
            if slot in regions:
                predictions.append(
                    PokerLegendsButtonPrediction(
                        frame_id=frame_id,
                        slot=slot,
                        visible=True,
                        action_type=action_type,
                        confidence=0.90,
                    )
                )
        return tuple(predictions)

    def _recognize_left_button(
        self,
        image: RgbImage,
        region: Mapping[str, object],
        *,
        frame_id: str,
        exclude_frame_id: str | None,
    ) -> PokerLegendsButtonPrediction:
        feature = _feature_from_crop(_crop(image, _rect_from_region(region)))
        match = self._best_match(feature, exclude_frame_id=exclude_frame_id)
        if match is None or match[1] > self.manifest.left_max_distance:
            return PokerLegendsButtonPrediction(
                frame_id=frame_id,
                slot=LEFT_SLOT,
                visible=False,
                action_type=None,
                confidence=0.0,
                distance=None if match is None else match[1],
                matched_template=None,
            )
        template, distance = match
        return PokerLegendsButtonPrediction(
            frame_id=frame_id,
            slot=LEFT_SLOT,
            visible=True,
            action_type=template.action_type,
            confidence=_confidence_from_distance(distance, self.manifest.left_max_distance),
            distance=distance,
            matched_template=template.image,
        )

    def _best_match(
        self,
        feature: GrayFeature,
        *,
        exclude_frame_id: str | None,
    ) -> tuple[PokerLegendsButtonTemplate, float] | None:
        best: tuple[PokerLegendsButtonTemplate, float] | None = None
        for template, template_feature in self._templates:
            if exclude_frame_id is not None and template.frame_id == exclude_frame_id:
                continue
            distance = float(np.mean((feature - template_feature) ** 2))
            if best is None or distance < best[1]:
                best = (template, distance)
        return best


def build_poker_legends_button_template_library(
    truth_paths: Sequence[str | Path],
    *,
    annotation_dir: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    left_max_distance: float = DEFAULT_LEFT_MAX_DISTANCE,
) -> PokerLegendsButtonTemplateManifest:
    output = Path(output_dir)
    template_dir = output / "button_templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    annotations = Path(annotation_dir)
    images = Path(image_root)
    templates: list[PokerLegendsButtonTemplate] = []
    for truth_path in sorted(Path(path) for path in truth_paths):
        truth = _read_json_object(truth_path)
        frame_id = str(truth["frame_id"])
        if _truth_screen_kind(truth) != ScreenKind.ACTIONABLE_TABLE.value:
            continue
        left = _truth_button(truth, LEFT_SLOT)
        if left is None or not bool(left.get("visible")):
            continue
        action_type = str(left.get("action_type") or "")
        if action_type not in {"check", "call"}:
            continue
        annotation = _read_json_object(annotations / f"{frame_id}.json")
        region = _button_regions_by_name(annotation).get(LEFT_SLOT)
        if region is None:
            continue
        image = _load_rgb_image(images / str(annotation["image"]))
        feature = _feature_from_crop(_crop(image, _rect_from_region(region)))
        relative_image = Path("button_templates") / action_type / f"{frame_id}__{LEFT_SLOT}.png"
        target = output / relative_image
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_feature_png(target, feature)
        templates.append(
            PokerLegendsButtonTemplate(
                action_type=action_type,
                frame_id=frame_id,
                slot=LEFT_SLOT,
                image=str(relative_image),
                source_image=str(annotation["image"]),
                sha256=_sha256_file(target),
            )
        )
    manifest = PokerLegendsButtonTemplateManifest(
        schema_version=1,
        normalized_width=DEFAULT_NORMALIZED_WIDTH,
        normalized_height=DEFAULT_NORMALIZED_HEIGHT,
        left_max_distance=left_max_distance,
        templates=tuple(templates),
    )
    manifest.write_json(output / BUTTON_TEMPLATE_MANIFEST)
    return manifest


def evaluate_poker_legends_button_templates(
    truth_paths: Sequence[str | Path],
    *,
    annotation_dir: str | Path,
    image_root: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    exclude_same_frame: bool = False,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    recognizer = PokerLegendsButtonRecognizer.from_manifest(manifest_path)
    annotations = Path(annotation_dir)
    images = Path(image_root)
    rows: list[dict[str, object]] = []
    correct = 0
    total = 0
    predicted = 0
    for truth_path in sorted(Path(path) for path in truth_paths):
        truth = _read_json_object(truth_path)
        frame_id = str(truth["frame_id"])
        if _truth_screen_kind(truth) != ScreenKind.ACTIONABLE_TABLE.value:
            continue
        annotation = _read_json_object(annotations / f"{frame_id}.json")
        predictions = {
            prediction.slot: prediction
            for prediction in recognizer.recognize(
                images / str(annotation["image"]),
                annotation,
                frame_id=frame_id,
                exclude_frame_id=frame_id if exclude_same_frame else None,
            )
        }
        expected = _expected_primary_buttons(truth)
        for slot, expected_action in expected.items():
            total += 1
            prediction = predictions.get(slot)
            observed = None if prediction is None else prediction.action_type
            if observed is not None:
                predicted += 1
            if observed == expected_action:
                correct += 1
                status = "match"
            elif observed is None:
                status = "missing"
            else:
                status = "mismatch"
            rows.append(
                {
                    "frame_id": frame_id,
                    "slot": slot,
                    "expected": expected_action,
                    "observed": observed,
                    "status": status,
                    "confidence": None if prediction is None else prediction.confidence,
                    "distance": None if prediction is None else prediction.distance,
                }
            )
    summary: dict[str, object] = {
        "schema_version": 1,
        "mode": "leave_same_frame_out" if exclude_same_frame else "template_self",
        "frames": len({str(row["frame_id"]) for row in rows}),
        "total": total,
        "predicted": predicted,
        "correct": correct,
        "accuracy": correct / total if total else 1.0,
        "precision": correct / predicted if predicted else 1.0,
        "coverage": predicted / total if total else 1.0,
        "rows": rows,
    }
    result_name = (
        "button_recognition_leave_frame.json"
        if exclude_same_frame
        else "button_recognition_self.json"
    )
    report_name = (
        "button_recognition_leave_frame.md" if exclude_same_frame else "button_recognition_self.md"
    )
    (output / result_name).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_button_recognition_report(output / report_name, summary)
    return summary


def build_and_evaluate_poker_legends_button_templates(
    truth_paths: Sequence[str | Path],
    *,
    annotation_dir: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    left_max_distance: float = DEFAULT_LEFT_MAX_DISTANCE,
) -> dict[str, object]:
    manifest = build_poker_legends_button_template_library(
        truth_paths,
        annotation_dir=annotation_dir,
        image_root=image_root,
        output_dir=output_dir,
        left_max_distance=left_max_distance,
    )
    output = Path(output_dir)
    self_eval = evaluate_poker_legends_button_templates(
        truth_paths,
        annotation_dir=annotation_dir,
        image_root=image_root,
        manifest_path=output / BUTTON_TEMPLATE_MANIFEST,
        output_dir=output,
        exclude_same_frame=False,
    )
    leave_frame_eval = evaluate_poker_legends_button_templates(
        truth_paths,
        annotation_dir=annotation_dir,
        image_root=image_root,
        manifest_path=output / BUTTON_TEMPLATE_MANIFEST,
        output_dir=output,
        exclude_same_frame=True,
    )
    action_counts = _action_counts(manifest.templates)
    summary = {
        "schema_version": 1,
        "templates": len(manifest.templates),
        "action_counts": action_counts,
        "left_max_distance": manifest.left_max_distance,
        "manifest": BUTTON_TEMPLATE_MANIFEST,
        "self_eval": _summary_without_rows(self_eval),
        "leave_frame_eval": _summary_without_rows(leave_frame_eval),
    }
    (output / "button_template_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_button_template_report(output / "button_template_report.md", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and evaluate Poker Legends primary button recognition."
    )
    parser.add_argument("truth_overlays", nargs="+", help="Truth overlay JSON files.")
    parser.add_argument(
        "--annotation-dir",
        required=True,
        help="Directory containing Poker Legends draft annotation JSON files.",
    )
    parser.add_argument("--image-root", required=True, help="Root for annotation image paths.")
    parser.add_argument("--out", required=True, help="Output directory for button artifacts.")
    parser.add_argument(
        "--left-max-distance",
        type=float,
        default=DEFAULT_LEFT_MAX_DISTANCE,
        help="Maximum template distance accepted for primary_left check/call.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = build_and_evaluate_poker_legends_button_templates(
        args.truth_overlays,
        annotation_dir=args.annotation_dir,
        image_root=args.image_root,
        output_dir=args.out,
        left_max_distance=args.left_max_distance,
    )
    print(json.dumps(_stdout_summary(summary), indent=2, sort_keys=True))


def _expected_primary_buttons(truth: Mapping[str, object]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for button in _mapping_sequence(truth.get("buttons")):
        name = str(button.get("name") or "")
        action_type = button.get("action_type")
        if name in {LEFT_SLOT, *PRIMARY_BUTTON_ACTIONS.keys()} and isinstance(action_type, str):
            expected[name] = action_type
    return expected


def _truth_button(truth: Mapping[str, object], name: str) -> Mapping[str, object] | None:
    for button in _mapping_sequence(truth.get("buttons")):
        if str(button.get("name") or "") == name:
            return button
    return None


def _button_regions_by_name(annotation: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    raw = annotation.get("regions")
    if not isinstance(raw, Mapping):
        return {}
    return {str(region.get("name")): region for region in _mapping_sequence(raw.get("buttons"))}


def _truth_screen_kind(truth: Mapping[str, object]) -> str:
    screen = truth.get("screen")
    if not isinstance(screen, Mapping):
        return ScreenKind.UNKNOWN_OR_TRANSITION.value
    return str(screen.get("kind") or ScreenKind.UNKNOWN_OR_TRANSITION.value)


def _action_counts(templates: Sequence[PokerLegendsButtonTemplate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for template in templates:
        counts[template.action_type] = counts.get(template.action_type, 0) + 1
    return dict(sorted(counts.items()))


def _feature_from_crop(crop: RgbImage) -> GrayFeature:
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(
        gray,
        (DEFAULT_NORMALIZED_WIDTH, DEFAULT_NORMALIZED_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    equalized = cv2.equalizeHist(resized)
    return cast(GrayFeature, equalized.astype(np.float32) / 255.0)


def _load_feature(path: str | Path) -> GrayFeature:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read template image: {path}")
    if image.shape != (DEFAULT_NORMALIZED_HEIGHT, DEFAULT_NORMALIZED_WIDTH):
        image = cv2.resize(
            image,
            (DEFAULT_NORMALIZED_WIDTH, DEFAULT_NORMALIZED_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
    return cast(GrayFeature, image.astype(np.float32) / 255.0)


def _write_feature_png(path: str | Path, feature: GrayFeature) -> None:
    image = np.clip(feature * 255.0, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not write template image: {path}")


def _load_rgb_image(path: str | Path) -> RgbImage:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cast(RgbImage, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _crop(image: RgbImage, rect: ScreenRect) -> RgbImage:
    height, width = image.shape[:2]
    x = max(0, rect.x)
    y = max(0, rect.y)
    right = min(width, rect.x + rect.width)
    bottom = min(height, rect.y + rect.height)
    return image[y:bottom, x:right]


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


def _confidence_from_distance(distance: float, max_distance: float) -> float:
    if max_distance <= 0:
        return 1.0 if distance == 0 else 0.0
    return max(0.0, min(1.0, 1.0 - (distance / max_distance)))


def _summary_without_rows(summary: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in summary.items() if key != "rows"}


def _stdout_summary(summary: Mapping[str, object]) -> dict[str, object]:
    self_eval = cast(Mapping[str, object], summary["self_eval"])
    leave_eval = cast(Mapping[str, object], summary["leave_frame_eval"])
    return {
        "templates": summary["templates"],
        "action_counts": summary["action_counts"],
        "left_max_distance": summary["left_max_distance"],
        "self_accuracy": self_eval["accuracy"],
        "leave_frame_precision": leave_eval["precision"],
        "leave_frame_coverage": leave_eval["coverage"],
        "manifest": summary["manifest"],
    }


def _write_button_recognition_report(path: Path, summary: Mapping[str, object]) -> None:
    rows = _mapping_sequence(summary["rows"])
    lines = [
        "# Poker Legends Button Recognition",
        "",
        "## Summary",
        f"- Mode: `{summary['mode']}`",
        f"- Frames: {summary['frames']}",
        f"- Buttons: {summary['total']}",
        f"- Correct: {summary['correct']}",
        f"- Accuracy: {_to_float(summary['accuracy']):.3f}",
        f"- Precision: {_to_float(summary['precision']):.3f}",
        f"- Coverage: {_to_float(summary['coverage']):.3f}",
        "",
        "## Conflicts",
    ]
    conflicts = [row for row in rows if row.get("status") != "match"]
    if conflicts:
        for row in conflicts:
            lines.append(
                f"- `{row['frame_id']}` `{row['slot']}`: {row['status']}; "
                f"expected={row['expected']!r}, observed={row['observed']!r}, "
                f"distance={row['distance']!r}"
            )
    else:
        lines.append("No conflicts.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_button_template_report(path: Path, summary: Mapping[str, object]) -> None:
    self_eval = cast(Mapping[str, object], summary["self_eval"])
    leave_eval = cast(Mapping[str, object], summary["leave_frame_eval"])
    action_counts = cast(Mapping[str, object], summary["action_counts"])
    lines = [
        "# Poker Legends Button Template Library",
        "",
        "## Scope",
        f"- Templates: {summary['templates']}",
        f"- Action counts: {', '.join(f'{key}={value}' for key, value in action_counts.items())}",
        f"- Left button max distance: {summary['left_max_distance']}",
        f"- Manifest: `{summary['manifest']}`",
        "",
        "## Evaluation",
        f"- Self accuracy: {_to_float(self_eval['accuracy']):.3f}",
        f"- Leave-frame precision: {_to_float(leave_eval['precision']):.3f}",
        f"- Leave-frame coverage: {_to_float(leave_eval['coverage']):.3f}",
        "",
        "## Notes",
        "- v0 recognizes only the three primary action buttons.",
        "- `primary_middle` and `primary_right` are position-mapped to raise/fold.",
        "- `primary_left` uses templates to separate check/call because OCR is unreliable.",
        "- Raise shortcut buttons are intentionally ignored; they are not required for a legal "
        "GameState action set and are visually close to non-action blind controls.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _read_json_object(path: str | Path) -> dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, object], data)


def _mapping_sequence(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [cast(Mapping[str, object], item) for item in value if isinstance(item, Mapping)]


def _to_int(value: object) -> int:
    if type(value) is int:
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"expected int-like value: {value!r}")


def _to_float(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        return float(value)
    raise TypeError(f"expected float-like value: {value!r}")
