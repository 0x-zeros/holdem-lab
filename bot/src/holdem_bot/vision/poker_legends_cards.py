"""Card template extraction and recognition for Poker Legends."""

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

RgbImage = NDArray[np.uint8]
GrayFeature = NDArray[np.float32]

DEFAULT_NORMALIZED_WIDTH = 64
DEFAULT_NORMALIZED_HEIGHT = 88
DEFAULT_MAX_DISTANCE = 0.010
DEFAULT_SCREEN_KINDS = (
    ScreenKind.ACTIONABLE_TABLE.value,
    ScreenKind.TABLE_OBSERVE.value,
)
TEMPLATE_MANIFEST = "card_template_manifest.json"


@dataclass(frozen=True, slots=True)
class PokerLegendsCardTemplate:
    card: str
    frame_id: str
    group: str
    slot: str
    image: str
    source_image: str
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        return cls(
            card=str(data["card"]),
            frame_id=str(data["frame_id"]),
            group=str(data["group"]),
            slot=str(data["slot"]),
            image=str(data["image"]),
            source_image=str(data["source_image"]),
            sha256=str(data["sha256"]),
        )


@dataclass(frozen=True, slots=True)
class PokerLegendsCardTemplateManifest:
    schema_version: int
    normalized_width: int
    normalized_height: int
    max_distance: float
    screen_kinds: tuple[str, ...]
    templates: tuple[PokerLegendsCardTemplate, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported card template manifest schema version")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "normalized_width": self.normalized_width,
            "normalized_height": self.normalized_height,
            "max_distance": self.max_distance,
            "screen_kinds": list(self.screen_kinds),
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
            max_distance=_to_float(data["max_distance"]),
            screen_kinds=tuple(str(item) for item in _sequence(data.get("screen_kinds"))),
            templates=tuple(
                PokerLegendsCardTemplate.from_dict(item)
                for item in _mapping_sequence(data["templates"])
            ),
        )


@dataclass(frozen=True, slots=True)
class PokerLegendsCardPrediction:
    frame_id: str
    group: str
    slot: str
    visible: bool
    card: str | None
    confidence: float
    distance: float | None
    matched_template: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class PokerLegendsCardTemplateRecognizer:
    def __init__(
        self,
        manifest: PokerLegendsCardTemplateManifest,
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
        return cls(PokerLegendsCardTemplateManifest.read_json(path), template_root=path.parent)

    def recognize(
        self,
        image_path: str | Path,
        annotation: Mapping[str, object],
        *,
        frame_id: str,
        exclude_frame_id: str | None = None,
    ) -> tuple[PokerLegendsCardPrediction, ...]:
        image = _load_rgb_image(image_path)
        predictions: list[PokerLegendsCardPrediction] = []
        for truth_group, region_group in (("hero_hole_cards", "cards"), ("board", "board")):
            for region in _region_group(annotation, region_group):
                slot = str(region["name"])
                feature = _feature_from_crop(_crop(image, _rect_from_region(region)))
                match = self._best_match(feature, exclude_frame_id=exclude_frame_id)
                if match is None:
                    predictions.append(
                        PokerLegendsCardPrediction(
                            frame_id=frame_id,
                            group=truth_group,
                            slot=slot,
                            visible=False,
                            card=None,
                            confidence=0.0,
                            distance=None,
                            matched_template=None,
                        )
                    )
                    continue
                template, distance = match
                is_visible = distance <= self.manifest.max_distance
                predictions.append(
                    PokerLegendsCardPrediction(
                        frame_id=frame_id,
                        group=truth_group,
                        slot=slot,
                        visible=is_visible,
                        card=template.card if is_visible else None,
                        confidence=_confidence_from_distance(distance, self.manifest.max_distance)
                        if is_visible
                        else 0.0,
                        distance=distance,
                        matched_template=template.image if is_visible else None,
                    )
                )
        return tuple(predictions)

    def _best_match(
        self,
        feature: GrayFeature,
        *,
        exclude_frame_id: str | None,
    ) -> tuple[PokerLegendsCardTemplate, float] | None:
        best: tuple[PokerLegendsCardTemplate, float] | None = None
        for template, template_feature in self._templates:
            if exclude_frame_id is not None and template.frame_id == exclude_frame_id:
                continue
            distance = float(np.mean((feature - template_feature) ** 2))
            if best is None or distance < best[1]:
                best = (template, distance)
        return best


def build_poker_legends_card_template_library(
    truth_paths: Sequence[str | Path],
    *,
    annotation_dir: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    screen_kinds: Sequence[str] = DEFAULT_SCREEN_KINDS,
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> PokerLegendsCardTemplateManifest:
    selected_screen_kinds = _normalize_screen_kinds(screen_kinds)
    output = Path(output_dir)
    template_dir = output / "card_templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    annotations = Path(annotation_dir)
    images = Path(image_root)
    templates: list[PokerLegendsCardTemplate] = []

    for truth_path in sorted(Path(path) for path in truth_paths):
        truth = _read_json_object(truth_path)
        frame_id = str(truth["frame_id"])
        screen_kind = _truth_screen_kind(truth)
        if screen_kind not in selected_screen_kinds:
            continue
        annotation = _read_json_object(annotations / f"{frame_id}.json")
        image_path = images / str(annotation["image"])
        image = _load_rgb_image(image_path)
        regions = _regions_by_truth_group(annotation)
        for item in _visible_truth_cards(truth):
            region = regions.get((item.group, item.slot))
            if region is None:
                continue
            feature = _feature_from_crop(_crop(image, _rect_from_region(region)))
            relative_image = (
                Path("card_templates") / item.card / f"{frame_id}__{item.group}__{item.slot}.png"
            )
            target = output / relative_image
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_feature_png(target, feature)
            templates.append(
                PokerLegendsCardTemplate(
                    card=item.card,
                    frame_id=frame_id,
                    group=item.group,
                    slot=item.slot,
                    image=str(relative_image),
                    source_image=str(annotation["image"]),
                    sha256=_sha256_file(target),
                )
            )

    manifest = PokerLegendsCardTemplateManifest(
        schema_version=1,
        normalized_width=DEFAULT_NORMALIZED_WIDTH,
        normalized_height=DEFAULT_NORMALIZED_HEIGHT,
        max_distance=max_distance,
        screen_kinds=tuple(selected_screen_kinds),
        templates=tuple(templates),
    )
    manifest.write_json(output / TEMPLATE_MANIFEST)
    return manifest


def evaluate_poker_legends_card_templates(
    truth_paths: Sequence[str | Path],
    *,
    annotation_dir: str | Path,
    image_root: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    exclude_same_frame: bool = False,
) -> dict[str, object]:
    manifest_file = Path(manifest_path)
    recognizer = PokerLegendsCardTemplateRecognizer.from_manifest(manifest_file)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    annotations = Path(annotation_dir)
    images = Path(image_root)
    rows: list[dict[str, object]] = []
    visible_total = 0
    visible_predicted = 0
    visible_correct = 0
    hidden_total = 0
    hidden_false_positive = 0

    for truth_path in sorted(Path(path) for path in truth_paths):
        truth = _read_json_object(truth_path)
        frame_id = str(truth["frame_id"])
        if _truth_screen_kind(truth) not in recognizer.manifest.screen_kinds:
            continue
        annotation = _read_json_object(annotations / f"{frame_id}.json")
        predictions = recognizer.recognize(
            images / str(annotation["image"]),
            annotation,
            frame_id=frame_id,
            exclude_frame_id=frame_id if exclude_same_frame else None,
        )
        expected_by_slot = {
            (item.group, item.slot): item for item in _truth_cards_for_scoring(truth, annotation)
        }
        for prediction in predictions:
            expected = expected_by_slot.get((prediction.group, prediction.slot))
            if expected is None or not expected.visible:
                hidden_total += 1
                status = "hidden_match" if prediction.card is None else "false_positive"
                if prediction.card is not None:
                    hidden_false_positive += 1
                expected_card = None
            else:
                visible_total += 1
                expected_card = expected.card
                if prediction.card is None:
                    status = "missing"
                elif prediction.card == expected.card:
                    status = "match"
                    visible_predicted += 1
                    visible_correct += 1
                else:
                    status = "mismatch"
                    visible_predicted += 1
            rows.append(
                {
                    "frame_id": frame_id,
                    "group": prediction.group,
                    "slot": prediction.slot,
                    "expected": expected_card,
                    "observed": prediction.card,
                    "status": status,
                    "confidence": prediction.confidence,
                    "distance": prediction.distance,
                    "matched_template": prediction.matched_template,
                }
            )

    summary: dict[str, object] = {
        "schema_version": 1,
        "mode": "leave_same_frame_out" if exclude_same_frame else "template_self",
        "frames": len({str(row["frame_id"]) for row in rows}),
        "visible_total": visible_total,
        "visible_predicted": visible_predicted,
        "visible_correct": visible_correct,
        "visible_accuracy": visible_correct / visible_total if visible_total else 1.0,
        "visible_precision": visible_correct / visible_predicted if visible_predicted else 1.0,
        "visible_coverage": visible_predicted / visible_total if visible_total else 1.0,
        "hidden_total": hidden_total,
        "hidden_false_positive": hidden_false_positive,
        "hidden_false_positive_rate": hidden_false_positive / hidden_total if hidden_total else 0.0,
        "rows": rows,
    }
    result_name = (
        "card_recognition_leave_frame.json" if exclude_same_frame else "card_recognition_self.json"
    )
    report_name = (
        "card_recognition_leave_frame.md" if exclude_same_frame else "card_recognition_self.md"
    )
    (output / result_name).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_card_recognition_report(output / report_name, summary)
    return summary


def build_and_evaluate_poker_legends_card_templates(
    truth_paths: Sequence[str | Path],
    *,
    annotation_dir: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    screen_kinds: Sequence[str] = DEFAULT_SCREEN_KINDS,
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> dict[str, object]:
    manifest = build_poker_legends_card_template_library(
        truth_paths,
        annotation_dir=annotation_dir,
        image_root=image_root,
        output_dir=output_dir,
        screen_kinds=screen_kinds,
        max_distance=max_distance,
    )
    output = Path(output_dir)
    self_eval = evaluate_poker_legends_card_templates(
        truth_paths,
        annotation_dir=annotation_dir,
        image_root=image_root,
        manifest_path=output / TEMPLATE_MANIFEST,
        output_dir=output,
        exclude_same_frame=False,
    )
    leave_frame_eval = evaluate_poker_legends_card_templates(
        truth_paths,
        annotation_dir=annotation_dir,
        image_root=image_root,
        manifest_path=output / TEMPLATE_MANIFEST,
        output_dir=output,
        exclude_same_frame=True,
    )
    unique_cards = sorted({template.card for template in manifest.templates})
    summary = {
        "schema_version": 1,
        "templates": len(manifest.templates),
        "unique_cards": len(unique_cards),
        "unique_card_codes": unique_cards,
        "screen_kinds": list(manifest.screen_kinds),
        "max_distance": manifest.max_distance,
        "manifest": TEMPLATE_MANIFEST,
        "self_eval": _summary_without_rows(self_eval),
        "leave_frame_eval": _summary_without_rows(leave_frame_eval),
    }
    (output / "card_template_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_template_summary_report(output / "card_template_report.md", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and evaluate Poker Legends card template recognition."
    )
    parser.add_argument("truth_overlays", nargs="+", help="Truth overlay JSON files.")
    parser.add_argument(
        "--annotation-dir",
        required=True,
        help="Directory containing Poker Legends draft annotation JSON files.",
    )
    parser.add_argument("--image-root", required=True, help="Root for annotation image paths.")
    parser.add_argument("--out", required=True, help="Output directory for card artifacts.")
    parser.add_argument(
        "--screen-kinds",
        default=",".join(DEFAULT_SCREEN_KINDS),
        help="Comma-separated screen kinds to use for card templates, or 'all'.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=DEFAULT_MAX_DISTANCE,
        help="Maximum normalized template distance accepted as a card prediction.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = build_and_evaluate_poker_legends_card_templates(
        args.truth_overlays,
        annotation_dir=args.annotation_dir,
        image_root=args.image_root,
        output_dir=args.out,
        screen_kinds=_parse_screen_kinds(args.screen_kinds),
        max_distance=args.max_distance,
    )
    print(json.dumps(_stdout_summary(summary), indent=2, sort_keys=True))


@dataclass(frozen=True, slots=True)
class _TruthCard:
    group: str
    slot: str
    visible: bool
    card: str | None


@dataclass(frozen=True, slots=True)
class _VisibleTruthCard:
    group: str
    slot: str
    card: str


def _visible_truth_cards(truth: Mapping[str, object]) -> tuple[_VisibleTruthCard, ...]:
    items: list[_VisibleTruthCard] = []
    for group in ("hero_hole_cards", "board"):
        if group in _ignored_fields(truth):
            continue
        for item in _mapping_sequence(truth.get(group)):
            if bool(item.get("visible")) and isinstance(item.get("card"), str):
                items.append(
                    _VisibleTruthCard(
                        group=group,
                        slot=str(item["slot"]),
                        card=str(item["card"]),
                    )
                )
    return tuple(items)


def _truth_cards_for_scoring(
    truth: Mapping[str, object],
    annotation: Mapping[str, object],
) -> tuple[_TruthCard, ...]:
    cards: dict[tuple[str, str], _TruthCard] = {}
    ignored = _ignored_fields(truth)
    for truth_group, region_group in (("hero_hole_cards", "cards"), ("board", "board")):
        for region in _region_group(annotation, region_group):
            slot = str(region["name"])
            cards[(truth_group, slot)] = _TruthCard(
                group=truth_group,
                slot=slot,
                visible=False,
                card=None,
            )
        if truth_group in ignored:
            continue
        for item in _mapping_sequence(truth.get(truth_group)):
            slot = str(item["slot"])
            cards[(truth_group, slot)] = _TruthCard(
                group=truth_group,
                slot=slot,
                visible=bool(item.get("visible")) and isinstance(item.get("card"), str),
                card=str(item["card"]) if isinstance(item.get("card"), str) else None,
            )
    return tuple(cards.values())


def _regions_by_truth_group(
    annotation: Mapping[str, object],
) -> dict[tuple[str, str], Mapping[str, object]]:
    regions: dict[tuple[str, str], Mapping[str, object]] = {}
    for region in _region_group(annotation, "cards"):
        regions[("hero_hole_cards", str(region["name"]))] = region
    for region in _region_group(annotation, "board"):
        regions[("board", str(region["name"]))] = region
    return regions


def _region_group(annotation: Mapping[str, object], group: str) -> list[Mapping[str, object]]:
    raw = annotation.get("regions")
    if not isinstance(raw, Mapping):
        return []
    regions = raw.get(group)
    return _mapping_sequence(regions)


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


def _truth_screen_kind(truth: Mapping[str, object]) -> str:
    screen = truth.get("screen")
    if not isinstance(screen, Mapping):
        return ScreenKind.UNKNOWN_OR_TRANSITION.value
    return str(screen.get("kind") or ScreenKind.UNKNOWN_OR_TRANSITION.value)


def _ignored_fields(truth: Mapping[str, object]) -> frozenset[str]:
    return frozenset(str(item) for item in _sequence(truth.get("ignored_fields")))


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
        "unique_cards": summary["unique_cards"],
        "max_distance": summary["max_distance"],
        "self_visible_accuracy": self_eval["visible_accuracy"],
        "leave_frame_visible_precision": leave_eval["visible_precision"],
        "leave_frame_visible_coverage": leave_eval["visible_coverage"],
        "manifest": summary["manifest"],
    }


def _write_card_recognition_report(path: Path, summary: Mapping[str, object]) -> None:
    rows = _mapping_sequence(summary["rows"])
    lines = [
        "# Poker Legends Card Recognition",
        "",
        "## Summary",
        f"- Mode: `{summary['mode']}`",
        f"- Frames: {summary['frames']}",
        f"- Visible cards: {summary['visible_total']}",
        f"- Visible correct: {summary['visible_correct']}",
        f"- Visible accuracy: {_to_float(summary['visible_accuracy']):.3f}",
        f"- Visible precision: {_to_float(summary['visible_precision']):.3f}",
        f"- Visible coverage: {_to_float(summary['visible_coverage']):.3f}",
        f"- Hidden false positives: {summary['hidden_false_positive']} / {summary['hidden_total']}",
        "",
        "## Conflicts",
    ]
    conflicts = [row for row in rows if row.get("status") not in {"match", "hidden_match"}]
    if conflicts:
        for row in conflicts:
            lines.append(
                f"- `{row['frame_id']}` `{row['group']}.{row['slot']}`: "
                f"{row['status']}; expected={row['expected']!r}, observed={row['observed']!r}, "
                f"distance={row['distance']!r}"
            )
    else:
        lines.append("No conflicts.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_template_summary_report(path: Path, summary: Mapping[str, object]) -> None:
    self_eval = cast(Mapping[str, object], summary["self_eval"])
    leave_eval = cast(Mapping[str, object], summary["leave_frame_eval"])
    codes = ", ".join(str(code) for code in _sequence(summary["unique_card_codes"]))
    lines = [
        "# Poker Legends Card Template Library",
        "",
        "## Scope",
        f"- Templates: {summary['templates']}",
        f"- Unique card codes: {summary['unique_cards']} / 52",
        f"- Screen kinds: {', '.join(str(kind) for kind in _sequence(summary['screen_kinds']))}",
        f"- Max distance: {summary['max_distance']}",
        f"- Manifest: `{summary['manifest']}`",
        "",
        "## Covered Cards",
        codes or "-",
        "",
        "## Evaluation",
        f"- Self visible accuracy: {_to_float(self_eval['visible_accuracy']):.3f}",
        f"- Self hidden false-positive rate: "
        f"{_to_float(self_eval['hidden_false_positive_rate']):.3f}",
        f"- Leave-frame visible precision: {_to_float(leave_eval['visible_precision']):.3f}",
        f"- Leave-frame visible coverage: {_to_float(leave_eval['visible_coverage']):.3f}",
        f"- Leave-frame hidden false-positive rate: "
        f"{_to_float(leave_eval['hidden_false_positive_rate']):.3f}",
        "",
        "## Notes",
        "- Self evaluation proves the template pipeline is wired correctly; it is not a "
        "generalization estimate.",
        "- Leave-frame evaluation excludes templates from the same frame and is intentionally "
        "conservative. Low coverage means the current 20-frame truth set is still too small "
        "for real-time card recognition without a stronger classifier or more labeled cards.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_screen_kinds(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() == "all":
        return tuple(kind.value for kind in ScreenKind)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _normalize_screen_kinds(screen_kinds: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(
        dict.fromkeys(str(kind).strip() for kind in screen_kinds if str(kind).strip())
    )
    if not normalized:
        raise ValueError("at least one screen kind is required")
    valid = {kind.value for kind in ScreenKind}
    invalid = sorted(set(normalized) - valid)
    if invalid:
        raise ValueError(f"unsupported screen kinds: {', '.join(invalid)}")
    return normalized


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
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"expected int-like value: {value!r}")


def _to_float(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        return float(value)
    raise TypeError(f"expected float-like value: {value!r}")
