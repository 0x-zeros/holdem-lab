"""Rank/suit part template recognition for Poker Legends cards."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Self, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from holdem_bot.screen_state import ScreenKind
from holdem_bot.vision.annotations import ScreenRect

RgbImage = NDArray[np.uint8]
FeatureImage = NDArray[np.float32]
CardPart = Literal["rank", "suit"]

DEFAULT_RANK_NORMALIZED_WIDTH = 44
DEFAULT_RANK_NORMALIZED_HEIGHT = 34
DEFAULT_SUIT_NORMALIZED_WIDTH = 40
DEFAULT_SUIT_NORMALIZED_HEIGHT = 40
DEFAULT_RANK_MAX_DISTANCE = 0.040
DEFAULT_SUIT_MAX_DISTANCE = 0.040
DEFAULT_SCREEN_KINDS = (
    ScreenKind.ACTIONABLE_TABLE.value,
    ScreenKind.TABLE_OBSERVE.value,
)
PART_TEMPLATE_MANIFEST = "card_part_template_manifest.json"
CARD_RANKS = frozenset("AKQJT98765432")
CARD_SUITS = frozenset("SHDC")


@dataclass(frozen=True, slots=True)
class PokerLegendsCardPartTemplate:
    part: CardPart
    label: str
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
        part = str(data["part"])
        if part not in {"rank", "suit"}:
            raise ValueError(f"unsupported card part: {part}")
        return cls(
            part=cast(CardPart, part),
            label=str(data["label"]),
            card=str(data["card"]),
            frame_id=str(data["frame_id"]),
            group=str(data["group"]),
            slot=str(data["slot"]),
            image=str(data["image"]),
            source_image=str(data["source_image"]),
            sha256=str(data["sha256"]),
        )


@dataclass(frozen=True, slots=True)
class PokerLegendsCardPartTemplateManifest:
    schema_version: int
    rank_normalized_width: int
    rank_normalized_height: int
    suit_normalized_width: int
    suit_normalized_height: int
    rank_max_distance: float
    suit_max_distance: float
    screen_kinds: tuple[str, ...]
    templates: tuple[PokerLegendsCardPartTemplate, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported card part template manifest schema version")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "rank_normalized_width": self.rank_normalized_width,
            "rank_normalized_height": self.rank_normalized_height,
            "suit_normalized_width": self.suit_normalized_width,
            "suit_normalized_height": self.suit_normalized_height,
            "rank_max_distance": self.rank_max_distance,
            "suit_max_distance": self.suit_max_distance,
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
            rank_normalized_width=_to_int(data["rank_normalized_width"]),
            rank_normalized_height=_to_int(data["rank_normalized_height"]),
            suit_normalized_width=_to_int(data["suit_normalized_width"]),
            suit_normalized_height=_to_int(data["suit_normalized_height"]),
            rank_max_distance=_to_float(data["rank_max_distance"]),
            suit_max_distance=_to_float(data["suit_max_distance"]),
            screen_kinds=tuple(str(item) for item in _sequence(data.get("screen_kinds"))),
            templates=tuple(
                PokerLegendsCardPartTemplate.from_dict(item)
                for item in _mapping_sequence(data["templates"])
            ),
        )


@dataclass(frozen=True, slots=True)
class PokerLegendsCardPartPrediction:
    frame_id: str
    group: str
    slot: str
    visible: bool
    card: str | None
    rank: str | None
    suit: str | None
    confidence: float
    rank_distance: float | None
    suit_distance: float | None
    matched_rank_template: str | None
    matched_suit_template: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class PokerLegendsCardPartTemplateRecognizer:
    def __init__(
        self,
        manifest: PokerLegendsCardPartTemplateManifest,
        *,
        template_root: str | Path,
    ) -> None:
        self.manifest = manifest
        self.template_root = Path(template_root)
        self._rank_templates = tuple(
            (template, _load_feature(self.template_root / template.image, part="rank"))
            for template in manifest.templates
            if template.part == "rank"
        )
        self._suit_templates = tuple(
            (template, _load_feature(self.template_root / template.image, part="suit"))
            for template in manifest.templates
            if template.part == "suit"
        )

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> Self:
        path = Path(manifest_path)
        return cls(
            PokerLegendsCardPartTemplateManifest.read_json(path),
            template_root=path.parent,
        )

    def recognize(
        self,
        image_path: str | Path,
        annotation: Mapping[str, object],
        *,
        frame_id: str,
        exclude_frame_id: str | None = None,
        exclude_card: str | None = None,
    ) -> tuple[PokerLegendsCardPartPrediction, ...]:
        image = _load_rgb_image(image_path)
        predictions: list[PokerLegendsCardPartPrediction] = []
        for truth_group, region_group in (("hero_hole_cards", "cards"), ("board", "board")):
            for region in _region_group(annotation, region_group):
                slot = str(region["name"])
                crop = _crop(image, _rect_from_region(region))
                rank_match = self._best_match(
                    _feature_from_card_crop(crop, part="rank"),
                    part="rank",
                    exclude_frame_id=exclude_frame_id,
                    exclude_card=exclude_card,
                )
                suit_match = self._best_match(
                    _feature_from_card_crop(crop, part="suit"),
                    part="suit",
                    exclude_frame_id=exclude_frame_id,
                    exclude_card=exclude_card,
                )
                predictions.append(
                    _prediction_from_matches(
                        frame_id=frame_id,
                        group=truth_group,
                        slot=slot,
                        rank_match=rank_match,
                        suit_match=suit_match,
                        rank_max_distance=self.manifest.rank_max_distance,
                        suit_max_distance=self.manifest.suit_max_distance,
                    )
                )
        return tuple(predictions)

    def _best_match(
        self,
        feature: FeatureImage,
        *,
        part: CardPart,
        exclude_frame_id: str | None,
        exclude_card: str | None,
    ) -> tuple[PokerLegendsCardPartTemplate, float] | None:
        templates = self._rank_templates if part == "rank" else self._suit_templates
        best: tuple[PokerLegendsCardPartTemplate, float] | None = None
        for template, template_feature in templates:
            if exclude_frame_id is not None and template.frame_id == exclude_frame_id:
                continue
            if exclude_card is not None and template.card == exclude_card:
                continue
            distance = float(np.mean((feature - template_feature) ** 2))
            if best is None or distance < best[1]:
                best = (template, distance)
        return best


def build_poker_legends_card_part_template_library(
    truth_paths: Sequence[str | Path],
    *,
    annotation_dir: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    screen_kinds: Sequence[str] = DEFAULT_SCREEN_KINDS,
    rank_max_distance: float = DEFAULT_RANK_MAX_DISTANCE,
    suit_max_distance: float = DEFAULT_SUIT_MAX_DISTANCE,
) -> PokerLegendsCardPartTemplateManifest:
    selected_screen_kinds = _normalize_screen_kinds(screen_kinds)
    output = Path(output_dir)
    template_dir = output / "card_part_templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    annotations = Path(annotation_dir)
    images = Path(image_root)
    templates: list[PokerLegendsCardPartTemplate] = []

    for truth_path in sorted(Path(path) for path in truth_paths):
        truth = _read_json_object(truth_path)
        frame_id = str(truth["frame_id"])
        if _truth_screen_kind(truth) not in selected_screen_kinds:
            continue
        annotation = _read_json_object(annotations / f"{frame_id}.json")
        image_path = images / str(annotation["image"])
        image = _load_rgb_image(image_path)
        regions = _regions_by_truth_group(annotation)
        for item in _visible_truth_cards(truth):
            region = regions.get((item.group, item.slot))
            if region is None:
                continue
            crop = _crop(image, _rect_from_region(region))
            for part, label in (("rank", item.card[0]), ("suit", item.card[1])):
                typed_part = cast(CardPart, part)
                feature = _feature_from_card_crop(crop, part=typed_part)
                relative_image = (
                    Path("card_part_templates")
                    / typed_part
                    / label
                    / f"{frame_id}__{item.group}__{item.slot}.png"
                )
                target = output / relative_image
                target.parent.mkdir(parents=True, exist_ok=True)
                _write_feature_png(target, feature)
                templates.append(
                    PokerLegendsCardPartTemplate(
                        part=typed_part,
                        label=label,
                        card=item.card,
                        frame_id=frame_id,
                        group=item.group,
                        slot=item.slot,
                        image=str(relative_image),
                        source_image=str(annotation["image"]),
                        sha256=_sha256_file(target),
                    )
                )

    manifest = PokerLegendsCardPartTemplateManifest(
        schema_version=1,
        rank_normalized_width=DEFAULT_RANK_NORMALIZED_WIDTH,
        rank_normalized_height=DEFAULT_RANK_NORMALIZED_HEIGHT,
        suit_normalized_width=DEFAULT_SUIT_NORMALIZED_WIDTH,
        suit_normalized_height=DEFAULT_SUIT_NORMALIZED_HEIGHT,
        rank_max_distance=rank_max_distance,
        suit_max_distance=suit_max_distance,
        screen_kinds=tuple(selected_screen_kinds),
        templates=tuple(templates),
    )
    manifest.write_json(output / PART_TEMPLATE_MANIFEST)
    return manifest


def evaluate_poker_legends_card_part_templates(
    truth_paths: Sequence[str | Path],
    *,
    annotation_dir: str | Path,
    image_root: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    exclude_same_frame: bool = False,
    exclude_same_card: bool = False,
) -> dict[str, object]:
    if exclude_same_frame and exclude_same_card:
        raise ValueError("only one exclusion mode can be enabled")
    manifest_file = Path(manifest_path)
    recognizer = PokerLegendsCardPartTemplateRecognizer.from_manifest(manifest_file)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    annotations = Path(annotation_dir)
    images = Path(image_root)
    rows: list[dict[str, object]] = []
    visible_total = 0
    visible_predicted = 0
    visible_correct = 0
    rank_total = 0
    rank_correct = 0
    suit_total = 0
    suit_correct = 0
    hidden_total = 0
    hidden_false_positive = 0

    for truth_path in sorted(Path(path) for path in truth_paths):
        truth = _read_json_object(truth_path)
        frame_id = str(truth["frame_id"])
        if _truth_screen_kind(truth) not in recognizer.manifest.screen_kinds:
            continue
        annotation = _read_json_object(annotations / f"{frame_id}.json")
        expected_by_slot = {
            (item.group, item.slot): item for item in _truth_cards_for_scoring(truth, annotation)
        }
        expected_cards = {key: item.card for key, item in expected_by_slot.items() if item.visible}
        predictions: list[PokerLegendsCardPartPrediction] = []
        for prediction in recognizer.recognize(
            images / str(annotation["image"]),
            annotation,
            frame_id=frame_id,
            exclude_frame_id=frame_id if exclude_same_frame else None,
        ):
            expected_card = expected_cards.get((prediction.group, prediction.slot))
            if exclude_same_card and expected_card is not None:
                prediction = recognizer.recognize(
                    images / str(annotation["image"]),
                    _single_region_annotation(annotation, prediction.group, prediction.slot),
                    frame_id=frame_id,
                    exclude_card=expected_card,
                )[0]
            predictions.append(prediction)

        for prediction in predictions:
            expected = expected_by_slot.get((prediction.group, prediction.slot))
            if expected is None or not expected.visible or expected.card is None:
                hidden_total += 1
                status = "hidden_match" if prediction.card is None else "false_positive"
                if prediction.card is not None:
                    hidden_false_positive += 1
                expected_card = None
            else:
                visible_total += 1
                expected_card = expected.card
                rank_total += 1
                suit_total += 1
                if prediction.rank == expected.card[0]:
                    rank_correct += 1
                if prediction.suit == expected.card[1]:
                    suit_correct += 1
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
                    "observed_rank": prediction.rank,
                    "observed_suit": prediction.suit,
                    "status": status,
                    "confidence": prediction.confidence,
                    "rank_distance": prediction.rank_distance,
                    "suit_distance": prediction.suit_distance,
                    "matched_rank_template": prediction.matched_rank_template,
                    "matched_suit_template": prediction.matched_suit_template,
                }
            )

    summary: dict[str, object] = {
        "schema_version": 1,
        "mode": _evaluation_mode(
            exclude_same_frame=exclude_same_frame,
            exclude_same_card=exclude_same_card,
        ),
        "frames": len({str(row["frame_id"]) for row in rows}),
        "visible_total": visible_total,
        "visible_predicted": visible_predicted,
        "visible_correct": visible_correct,
        "visible_accuracy": visible_correct / visible_total if visible_total else 1.0,
        "visible_precision": visible_correct / visible_predicted if visible_predicted else 1.0,
        "visible_coverage": visible_predicted / visible_total if visible_total else 1.0,
        "rank_accuracy": rank_correct / rank_total if rank_total else 1.0,
        "suit_accuracy": suit_correct / suit_total if suit_total else 1.0,
        "hidden_total": hidden_total,
        "hidden_false_positive": hidden_false_positive,
        "hidden_false_positive_rate": hidden_false_positive / hidden_total if hidden_total else 0.0,
        "rows": rows,
    }
    result_name = f"card_part_recognition_{summary['mode']}.json"
    report_name = f"card_part_recognition_{summary['mode']}.md"
    (output / result_name).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_card_part_recognition_report(output / report_name, summary)
    return summary


def build_and_evaluate_poker_legends_card_part_templates(
    truth_paths: Sequence[str | Path],
    *,
    annotation_dir: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    screen_kinds: Sequence[str] = DEFAULT_SCREEN_KINDS,
    rank_max_distance: float = DEFAULT_RANK_MAX_DISTANCE,
    suit_max_distance: float = DEFAULT_SUIT_MAX_DISTANCE,
) -> dict[str, object]:
    manifest = build_poker_legends_card_part_template_library(
        truth_paths,
        annotation_dir=annotation_dir,
        image_root=image_root,
        output_dir=output_dir,
        screen_kinds=screen_kinds,
        rank_max_distance=rank_max_distance,
        suit_max_distance=suit_max_distance,
    )
    output = Path(output_dir)
    self_eval = evaluate_poker_legends_card_part_templates(
        truth_paths,
        annotation_dir=annotation_dir,
        image_root=image_root,
        manifest_path=output / PART_TEMPLATE_MANIFEST,
        output_dir=output,
    )
    leave_frame_eval = evaluate_poker_legends_card_part_templates(
        truth_paths,
        annotation_dir=annotation_dir,
        image_root=image_root,
        manifest_path=output / PART_TEMPLATE_MANIFEST,
        output_dir=output,
        exclude_same_frame=True,
    )
    leave_card_eval = evaluate_poker_legends_card_part_templates(
        truth_paths,
        annotation_dir=annotation_dir,
        image_root=image_root,
        manifest_path=output / PART_TEMPLATE_MANIFEST,
        output_dir=output,
        exclude_same_card=True,
    )
    rank_labels = sorted(
        {template.label for template in manifest.templates if template.part == "rank"}
    )
    suit_labels = sorted(
        {template.label for template in manifest.templates if template.part == "suit"}
    )
    card_codes = sorted({template.card for template in manifest.templates})
    summary = {
        "schema_version": 1,
        "templates": len(manifest.templates),
        "rank_templates": len([item for item in manifest.templates if item.part == "rank"]),
        "suit_templates": len([item for item in manifest.templates if item.part == "suit"]),
        "unique_cards": len(card_codes),
        "unique_card_codes": card_codes,
        "rank_labels": rank_labels,
        "suit_labels": suit_labels,
        "screen_kinds": list(manifest.screen_kinds),
        "rank_max_distance": manifest.rank_max_distance,
        "suit_max_distance": manifest.suit_max_distance,
        "manifest": PART_TEMPLATE_MANIFEST,
        "self_eval": _summary_without_rows(self_eval),
        "leave_frame_eval": _summary_without_rows(leave_frame_eval),
        "leave_card_eval": _summary_without_rows(leave_card_eval),
    }
    (output / "card_part_template_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_card_part_template_summary_report(output / "card_part_template_report.md", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and evaluate Poker Legends rank/suit part template recognition."
    )
    parser.add_argument("truth_overlays", nargs="+", help="Truth overlay JSON files.")
    parser.add_argument(
        "--annotation-dir",
        required=True,
        help="Directory containing Poker Legends draft annotation JSON files.",
    )
    parser.add_argument("--image-root", required=True, help="Root for annotation image paths.")
    parser.add_argument("--out", required=True, help="Output directory for card part artifacts.")
    parser.add_argument(
        "--screen-kinds",
        default=",".join(DEFAULT_SCREEN_KINDS),
        help="Comma-separated screen kinds to use for part templates, or 'all'.",
    )
    parser.add_argument(
        "--rank-max-distance",
        type=float,
        default=DEFAULT_RANK_MAX_DISTANCE,
        help="Maximum normalized distance accepted as a rank prediction.",
    )
    parser.add_argument(
        "--suit-max-distance",
        type=float,
        default=DEFAULT_SUIT_MAX_DISTANCE,
        help="Maximum normalized distance accepted as a suit prediction.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = build_and_evaluate_poker_legends_card_part_templates(
        args.truth_overlays,
        annotation_dir=args.annotation_dir,
        image_root=args.image_root,
        output_dir=args.out,
        screen_kinds=_parse_screen_kinds(args.screen_kinds),
        rank_max_distance=args.rank_max_distance,
        suit_max_distance=args.suit_max_distance,
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


def _prediction_from_matches(
    *,
    frame_id: str,
    group: str,
    slot: str,
    rank_match: tuple[PokerLegendsCardPartTemplate, float] | None,
    suit_match: tuple[PokerLegendsCardPartTemplate, float] | None,
    rank_max_distance: float,
    suit_max_distance: float,
) -> PokerLegendsCardPartPrediction:
    if rank_match is None or suit_match is None:
        return PokerLegendsCardPartPrediction(
            frame_id=frame_id,
            group=group,
            slot=slot,
            visible=False,
            card=None,
            rank=None,
            suit=None,
            confidence=0.0,
            rank_distance=None if rank_match is None else rank_match[1],
            suit_distance=None if suit_match is None else suit_match[1],
            matched_rank_template=None,
            matched_suit_template=None,
        )
    rank_template, rank_distance = rank_match
    suit_template, suit_distance = suit_match
    rank_visible = rank_distance <= rank_max_distance
    suit_visible = suit_distance <= suit_max_distance
    if not rank_visible or not suit_visible:
        return PokerLegendsCardPartPrediction(
            frame_id=frame_id,
            group=group,
            slot=slot,
            visible=False,
            card=None,
            rank=rank_template.label if rank_visible else None,
            suit=suit_template.label if suit_visible else None,
            confidence=0.0,
            rank_distance=rank_distance,
            suit_distance=suit_distance,
            matched_rank_template=rank_template.image if rank_visible else None,
            matched_suit_template=suit_template.image if suit_visible else None,
        )
    rank_confidence = _confidence_from_distance(rank_distance, rank_max_distance)
    suit_confidence = _confidence_from_distance(suit_distance, suit_max_distance)
    rank = rank_template.label
    suit = suit_template.label
    return PokerLegendsCardPartPrediction(
        frame_id=frame_id,
        group=group,
        slot=slot,
        visible=True,
        card=f"{rank}{suit}",
        rank=rank,
        suit=suit,
        confidence=min(rank_confidence, suit_confidence),
        rank_distance=rank_distance,
        suit_distance=suit_distance,
        matched_rank_template=rank_template.image,
        matched_suit_template=suit_template.image,
    )


def _feature_from_card_crop(crop: RgbImage, *, part: CardPart) -> FeatureImage:
    part_crop = _crop_relative(crop, _part_rect(part))
    size = _normalized_size(part)
    if part == "suit":
        resized_color = cv2.resize(part_crop, size, interpolation=cv2.INTER_AREA)
        return cast(FeatureImage, resized_color.astype(np.float32) / 255.0)
    gray = cv2.cvtColor(part_crop, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    equalized = cv2.equalizeHist(resized)
    return cast(FeatureImage, equalized.astype(np.float32) / 255.0)


def _part_rect(part: CardPart) -> tuple[float, float, float, float]:
    if part == "rank":
        return (0.06, 0.03, 0.58, 0.38)
    return (0.10, 0.36, 0.60, 0.48)


def _normalized_size(part: CardPart) -> tuple[int, int]:
    if part == "rank":
        return (DEFAULT_RANK_NORMALIZED_WIDTH, DEFAULT_RANK_NORMALIZED_HEIGHT)
    return (DEFAULT_SUIT_NORMALIZED_WIDTH, DEFAULT_SUIT_NORMALIZED_HEIGHT)


def _crop_relative(image: RgbImage, rect: tuple[float, float, float, float]) -> RgbImage:
    height, width = image.shape[:2]
    x = max(0, min(width - 1, int(round(width * rect[0]))))
    y = max(0, min(height - 1, int(round(height * rect[1]))))
    right = max(x + 1, min(width, int(round(width * (rect[0] + rect[2])))))
    bottom = max(y + 1, min(height, int(round(height * (rect[1] + rect[3])))))
    return image[y:bottom, x:right]


def _load_feature(path: str | Path, *, part: CardPart) -> FeatureImage:
    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR if part == "suit" else cv2.IMREAD_GRAYSCALE,
    )
    if image is None:
        raise FileNotFoundError(f"could not read template image: {path}")
    height = DEFAULT_RANK_NORMALIZED_HEIGHT if part == "rank" else DEFAULT_SUIT_NORMALIZED_HEIGHT
    width = DEFAULT_RANK_NORMALIZED_WIDTH if part == "rank" else DEFAULT_SUIT_NORMALIZED_WIDTH
    if part == "suit":
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return cast(FeatureImage, cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    if image.shape != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return cast(FeatureImage, image.astype(np.float32) / 255.0)


def _write_feature_png(path: str | Path, feature: FeatureImage) -> None:
    image = np.clip(feature * 255.0, 0, 255).astype(np.uint8)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not write template image: {path}")


def _visible_truth_cards(truth: Mapping[str, object]) -> tuple[_VisibleTruthCard, ...]:
    items: list[_VisibleTruthCard] = []
    for group in ("hero_hole_cards", "board"):
        if group in _ignored_fields(truth):
            continue
        for item in _mapping_sequence(truth.get(group)):
            raw_card = item.get("card")
            if bool(item.get("visible")) and isinstance(raw_card, str):
                card = _normalize_card_code(raw_card)
                if card is None:
                    continue
                items.append(
                    _VisibleTruthCard(
                        group=group,
                        slot=str(item["slot"]),
                        card=card,
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
            card = _normalize_card_code(item.get("card"))
            cards[(truth_group, slot)] = _TruthCard(
                group=truth_group,
                slot=slot,
                visible=bool(item.get("visible")) and card is not None,
                card=card,
            )
    return tuple(cards.values())


def _single_region_annotation(
    annotation: Mapping[str, object],
    group: str,
    slot: str,
) -> dict[str, object]:
    region_group = "cards" if group == "hero_hole_cards" else "board"
    regions = [
        dict(region)
        for region in _region_group(annotation, region_group)
        if str(region.get("name")) == slot
    ]
    return {
        "image": annotation["image"],
        "regions": {
            "cards": regions if group == "hero_hole_cards" else [],
            "board": regions if group == "board" else [],
        },
    }


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


def _normalize_card_code(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    card = value.strip().upper()
    if len(card) != 2:
        return None
    if card[0] not in CARD_RANKS or card[1] not in CARD_SUITS:
        return None
    return card


def _confidence_from_distance(distance: float, max_distance: float) -> float:
    if max_distance <= 0:
        return 1.0 if distance == 0 else 0.0
    return max(0.0, min(1.0, 1.0 - (distance / max_distance)))


def _evaluation_mode(*, exclude_same_frame: bool, exclude_same_card: bool) -> str:
    if exclude_same_frame:
        return "leave_frame"
    if exclude_same_card:
        return "leave_card"
    return "self"


def _summary_without_rows(summary: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in summary.items() if key != "rows"}


def _stdout_summary(summary: Mapping[str, object]) -> dict[str, object]:
    self_eval = cast(Mapping[str, object], summary["self_eval"])
    leave_frame_eval = cast(Mapping[str, object], summary["leave_frame_eval"])
    leave_card_eval = cast(Mapping[str, object], summary["leave_card_eval"])
    return {
        "templates": summary["templates"],
        "unique_cards": summary["unique_cards"],
        "rank_labels": summary["rank_labels"],
        "suit_labels": summary["suit_labels"],
        "rank_max_distance": summary["rank_max_distance"],
        "suit_max_distance": summary["suit_max_distance"],
        "self_visible_accuracy": self_eval["visible_accuracy"],
        "leave_frame_visible_precision": leave_frame_eval["visible_precision"],
        "leave_frame_visible_coverage": leave_frame_eval["visible_coverage"],
        "leave_card_visible_accuracy": leave_card_eval["visible_accuracy"],
        "leave_card_visible_precision": leave_card_eval["visible_precision"],
        "leave_card_visible_coverage": leave_card_eval["visible_coverage"],
        "manifest": summary["manifest"],
    }


def _write_card_part_recognition_report(path: Path, summary: Mapping[str, object]) -> None:
    rows = _mapping_sequence(summary["rows"])
    lines = [
        "# Poker Legends Card Part Recognition",
        "",
        "## Summary",
        f"- Mode: `{summary['mode']}`",
        f"- Frames: {summary['frames']}",
        f"- Visible cards: {summary['visible_total']}",
        f"- Visible correct: {summary['visible_correct']}",
        f"- Visible accuracy: {_to_float(summary['visible_accuracy']):.3f}",
        f"- Visible precision: {_to_float(summary['visible_precision']):.3f}",
        f"- Visible coverage: {_to_float(summary['visible_coverage']):.3f}",
        f"- Rank accuracy: {_to_float(summary['rank_accuracy']):.3f}",
        f"- Suit accuracy: {_to_float(summary['suit_accuracy']):.3f}",
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
                f"rank={row['observed_rank']!r}, suit={row['observed_suit']!r}, "
                f"rank_distance={row['rank_distance']!r}, suit_distance={row['suit_distance']!r}"
            )
    else:
        lines.append("No conflicts.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_card_part_template_summary_report(
    path: Path,
    summary: Mapping[str, object],
) -> None:
    self_eval = cast(Mapping[str, object], summary["self_eval"])
    leave_frame_eval = cast(Mapping[str, object], summary["leave_frame_eval"])
    leave_card_eval = cast(Mapping[str, object], summary["leave_card_eval"])
    codes = ", ".join(str(code) for code in _sequence(summary["unique_card_codes"]))
    lines = [
        "# Poker Legends Card Part Template Library",
        "",
        "## Scope",
        f"- Templates: {summary['templates']}",
        f"- Rank templates: {summary['rank_templates']}",
        f"- Suit templates: {summary['suit_templates']}",
        f"- Unique card codes: {summary['unique_cards']} / 52",
        f"- Rank labels: {', '.join(str(item) for item in _sequence(summary['rank_labels']))}",
        f"- Suit labels: {', '.join(str(item) for item in _sequence(summary['suit_labels']))}",
        f"- Screen kinds: {', '.join(str(kind) for kind in _sequence(summary['screen_kinds']))}",
        f"- Rank max distance: {summary['rank_max_distance']}",
        f"- Suit max distance: {summary['suit_max_distance']}",
        f"- Manifest: `{summary['manifest']}`",
        "",
        "## Covered Cards",
        codes or "-",
        "",
        "## Evaluation",
        f"- Self visible accuracy: {_to_float(self_eval['visible_accuracy']):.3f}",
        f"- Self hidden false-positive rate: "
        f"{_to_float(self_eval['hidden_false_positive_rate']):.3f}",
        f"- Leave-frame visible precision: {_to_float(leave_frame_eval['visible_precision']):.3f}",
        f"- Leave-frame visible coverage: {_to_float(leave_frame_eval['visible_coverage']):.3f}",
        f"- Leave-card visible precision: {_to_float(leave_card_eval['visible_precision']):.3f}",
        f"- Leave-card visible coverage: {_to_float(leave_card_eval['visible_coverage']):.3f}",
        f"- Leave-card rank accuracy: {_to_float(leave_card_eval['rank_accuracy']):.3f}",
        f"- Leave-card suit accuracy: {_to_float(leave_card_eval['suit_accuracy']):.3f}",
        "",
        "## Notes",
        "- Leave-card evaluation excludes templates from the exact expected card code. It is a "
        "rough proxy for whether separate rank and suit templates can generalize to unseen "
        "rank+suit combinations.",
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
