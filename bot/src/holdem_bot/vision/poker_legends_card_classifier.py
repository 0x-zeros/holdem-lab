"""Local rank/suit classification for Poker Legends cards."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
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
FloatVector = NDArray[np.float32]
ClassifierPart = Literal["visibility", "rank", "suit"]

DEFAULT_VISIBILITY_NORMALIZED_WIDTH = 48
DEFAULT_VISIBILITY_NORMALIZED_HEIGHT = 64
DEFAULT_RANK_NORMALIZED_WIDTH = 44
DEFAULT_RANK_NORMALIZED_HEIGHT = 34
DEFAULT_SUIT_NORMALIZED_WIDTH = 40
DEFAULT_SUIT_NORMALIZED_HEIGHT = 40
DEFAULT_VISIBILITY_K = 7
DEFAULT_RANK_K = 7
DEFAULT_SUIT_K = 7
DEFAULT_VISIBILITY_MIN_WEIGHT_RATIO = 0.55
DEFAULT_RANK_MIN_WEIGHT_RATIO = 0.50
DEFAULT_SUIT_MIN_WEIGHT_RATIO = 0.50
DEFAULT_VISIBILITY_MAX_DISTANCE = 0.180
DEFAULT_RANK_MAX_DISTANCE = 0.200
DEFAULT_SUIT_MAX_DISTANCE = 0.220
DEFAULT_SCREEN_KINDS = (
    ScreenKind.ACTIONABLE_TABLE.value,
    ScreenKind.TABLE_OBSERVE.value,
)
CARD_CLASSIFIER_MANIFEST = "card_classifier_manifest.json"
CARD_RANKS = frozenset("AKQJT98765432")
CARD_SUITS = frozenset("SHDC")


@dataclass(frozen=True, slots=True)
class PokerLegendsCardClassifierSample:
    part: ClassifierPart
    label: str
    card: str | None
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
        if part not in {"visibility", "rank", "suit"}:
            raise ValueError(f"unsupported card classifier part: {part}")
        card = data.get("card")
        return cls(
            part=cast(ClassifierPart, part),
            label=str(data["label"]),
            card=str(card) if isinstance(card, str) else None,
            frame_id=str(data["frame_id"]),
            group=str(data["group"]),
            slot=str(data["slot"]),
            image=str(data["image"]),
            source_image=str(data["source_image"]),
            sha256=str(data["sha256"]),
        )


@dataclass(frozen=True, slots=True)
class PokerLegendsCardClassifierManifest:
    schema_version: int
    visibility_normalized_width: int
    visibility_normalized_height: int
    rank_normalized_width: int
    rank_normalized_height: int
    suit_normalized_width: int
    suit_normalized_height: int
    visibility_k: int
    rank_k: int
    suit_k: int
    visibility_min_weight_ratio: float
    rank_min_weight_ratio: float
    suit_min_weight_ratio: float
    visibility_max_distance: float
    rank_max_distance: float
    suit_max_distance: float
    screen_kinds: tuple[str, ...]
    samples: tuple[PokerLegendsCardClassifierSample, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported card classifier manifest schema version")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "visibility_normalized_width": self.visibility_normalized_width,
            "visibility_normalized_height": self.visibility_normalized_height,
            "rank_normalized_width": self.rank_normalized_width,
            "rank_normalized_height": self.rank_normalized_height,
            "suit_normalized_width": self.suit_normalized_width,
            "suit_normalized_height": self.suit_normalized_height,
            "visibility_k": self.visibility_k,
            "rank_k": self.rank_k,
            "suit_k": self.suit_k,
            "visibility_min_weight_ratio": self.visibility_min_weight_ratio,
            "rank_min_weight_ratio": self.rank_min_weight_ratio,
            "suit_min_weight_ratio": self.suit_min_weight_ratio,
            "visibility_max_distance": self.visibility_max_distance,
            "rank_max_distance": self.rank_max_distance,
            "suit_max_distance": self.suit_max_distance,
            "screen_kinds": list(self.screen_kinds),
            "samples": [sample.to_dict() for sample in self.samples],
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
            visibility_normalized_width=_to_int(data["visibility_normalized_width"]),
            visibility_normalized_height=_to_int(data["visibility_normalized_height"]),
            rank_normalized_width=_to_int(data["rank_normalized_width"]),
            rank_normalized_height=_to_int(data["rank_normalized_height"]),
            suit_normalized_width=_to_int(data["suit_normalized_width"]),
            suit_normalized_height=_to_int(data["suit_normalized_height"]),
            visibility_k=_to_int(data["visibility_k"]),
            rank_k=_to_int(data["rank_k"]),
            suit_k=_to_int(data["suit_k"]),
            visibility_min_weight_ratio=_to_float(data["visibility_min_weight_ratio"]),
            rank_min_weight_ratio=_to_float(data["rank_min_weight_ratio"]),
            suit_min_weight_ratio=_to_float(data["suit_min_weight_ratio"]),
            visibility_max_distance=_to_float(data["visibility_max_distance"]),
            rank_max_distance=_to_float(data["rank_max_distance"]),
            suit_max_distance=_to_float(data["suit_max_distance"]),
            screen_kinds=tuple(str(item) for item in _sequence(data.get("screen_kinds"))),
            samples=tuple(
                PokerLegendsCardClassifierSample.from_dict(item)
                for item in _mapping_sequence(data["samples"])
            ),
        )


@dataclass(frozen=True, slots=True)
class PokerLegendsCardClassifierPrediction:
    frame_id: str
    group: str
    slot: str
    visible: bool
    card: str | None
    rank: str | None
    suit: str | None
    confidence: float
    visibility_label: str | None
    visibility_confidence: float
    visibility_distance: float | None
    visibility_weight_ratio: float | None
    rank_confidence: float
    rank_distance: float | None
    rank_weight_ratio: float | None
    suit_confidence: float
    suit_distance: float | None
    suit_weight_ratio: float | None
    matched_visibility_samples: tuple[str, ...]
    matched_rank_samples: tuple[str, ...]
    matched_suit_samples: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class PokerLegendsCardClassifier:
    def __init__(
        self,
        manifest: PokerLegendsCardClassifierManifest,
        *,
        sample_root: str | Path,
    ) -> None:
        self.manifest = manifest
        self.sample_root = Path(sample_root)
        self._samples = tuple(
            _LoadedSample(
                sample=sample,
                vector=_vector_from_sample_image(
                    _load_rgb_image(self.sample_root / sample.image),
                    part=sample.part,
                ),
            )
            for sample in manifest.samples
        )

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> Self:
        path = Path(manifest_path)
        return cls(
            PokerLegendsCardClassifierManifest.read_json(path),
            sample_root=path.parent,
        )

    def recognize(
        self,
        image_path: str | Path,
        annotation: Mapping[str, object],
        *,
        frame_id: str,
        exclude_frame_id: str | None = None,
        exclude_card: str | None = None,
    ) -> tuple[PokerLegendsCardClassifierPrediction, ...]:
        image = _load_rgb_image(image_path)
        predictions: list[PokerLegendsCardClassifierPrediction] = []
        for truth_group, region_group in (("hero_hole_cards", "cards"), ("board", "board")):
            for region in _region_group(annotation, region_group):
                slot = str(region["name"])
                crop = _crop(image, _rect_from_region(region))
                predictions.append(
                    self._recognize_crop(
                        crop,
                        frame_id=frame_id,
                        group=truth_group,
                        slot=slot,
                        exclude_frame_id=exclude_frame_id,
                        exclude_card=exclude_card,
                    )
                )
        return tuple(predictions)

    def _recognize_crop(
        self,
        crop: RgbImage,
        *,
        frame_id: str,
        group: str,
        slot: str,
        exclude_frame_id: str | None,
        exclude_card: str | None,
    ) -> PokerLegendsCardClassifierPrediction:
        visibility_vote = self._predict(
            _visibility_vector_from_card_crop(crop),
            part="visibility",
            k=self.manifest.visibility_k,
            exclude_frame_id=exclude_frame_id,
            exclude_card=exclude_card,
        )
        visibility_ok = _accepted_vote(
            visibility_vote,
            expected_label="visible",
            min_weight_ratio=self.manifest.visibility_min_weight_ratio,
            max_distance=self.manifest.visibility_max_distance,
        )
        if not visibility_ok:
            return _prediction_from_votes(
                frame_id=frame_id,
                group=group,
                slot=slot,
                visibility_vote=visibility_vote,
                rank_vote=None,
                suit_vote=None,
                rank_ok=False,
                suit_ok=False,
            )

        rank_vote = self._predict(
            _rank_vector_from_card_crop(crop),
            part="rank",
            k=self.manifest.rank_k,
            exclude_frame_id=exclude_frame_id,
            exclude_card=exclude_card,
        )
        suit_vote = self._predict(
            _suit_vector_from_card_crop(crop),
            part="suit",
            k=self.manifest.suit_k,
            exclude_frame_id=exclude_frame_id,
            exclude_card=exclude_card,
        )
        rank_ok = _accepted_vote(
            rank_vote,
            expected_label=None,
            min_weight_ratio=self.manifest.rank_min_weight_ratio,
            max_distance=self.manifest.rank_max_distance,
        )
        suit_ok = _accepted_vote(
            suit_vote,
            expected_label=None,
            min_weight_ratio=self.manifest.suit_min_weight_ratio,
            max_distance=self.manifest.suit_max_distance,
        )
        return _prediction_from_votes(
            frame_id=frame_id,
            group=group,
            slot=slot,
            visibility_vote=visibility_vote,
            rank_vote=rank_vote,
            suit_vote=suit_vote,
            rank_ok=rank_ok,
            suit_ok=suit_ok,
        )

    def _predict(
        self,
        vector: FloatVector,
        *,
        part: ClassifierPart,
        k: int,
        exclude_frame_id: str | None,
        exclude_card: str | None,
    ) -> _KnnVote | None:
        candidates: list[tuple[_LoadedSample, float]] = []
        for loaded in self._samples:
            sample = loaded.sample
            if sample.part != part:
                continue
            if exclude_frame_id is not None and sample.frame_id == exclude_frame_id:
                continue
            if exclude_card is not None and sample.card == exclude_card:
                continue
            candidates.append((loaded, _mean_square_distance(vector, loaded.vector)))
        if not candidates:
            return None
        nearest = sorted(candidates, key=lambda item: item[1])[: max(1, min(k, len(candidates)))]
        label_weights: defaultdict[str, float] = defaultdict(float)
        label_distances: defaultdict[str, list[float]] = defaultdict(list)
        label_counts: Counter[str] = Counter()
        for loaded, distance in nearest:
            weight = 1.0 / max(distance, 1e-9)
            label = loaded.sample.label
            label_weights[label] += weight
            label_distances[label].append(distance)
            label_counts[label] += 1
        label = max(
            label_weights,
            key=lambda item: (
                label_weights[item],
                label_counts[item],
                -float(np.mean(label_distances[item])),
            ),
        )
        winning_distances = label_distances[label]
        total_weight = sum(label_weights.values())
        return _KnnVote(
            label=label,
            confidence=_confidence_from_vote(
                weight_ratio=label_weights[label] / total_weight if total_weight else 0.0,
                distance=float(np.mean(winning_distances)),
                max_distance=_default_max_distance(part, self.manifest),
            ),
            vote_ratio=label_counts[label] / len(nearest),
            weight_ratio=label_weights[label] / total_weight if total_weight else 0.0,
            mean_distance=float(np.mean(winning_distances)),
            nearest_distance=min(winning_distances),
            sample_count=len(nearest),
            neighbor_images=tuple(loaded.sample.image for loaded, _distance in nearest),
        )


def build_poker_legends_card_classifier(
    truth_paths: Sequence[str | Path],
    *,
    annotation_dir: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    screen_kinds: Sequence[str] = DEFAULT_SCREEN_KINDS,
    visibility_k: int = DEFAULT_VISIBILITY_K,
    rank_k: int = DEFAULT_RANK_K,
    suit_k: int = DEFAULT_SUIT_K,
    visibility_min_weight_ratio: float = DEFAULT_VISIBILITY_MIN_WEIGHT_RATIO,
    rank_min_weight_ratio: float = DEFAULT_RANK_MIN_WEIGHT_RATIO,
    suit_min_weight_ratio: float = DEFAULT_SUIT_MIN_WEIGHT_RATIO,
    visibility_max_distance: float = DEFAULT_VISIBILITY_MAX_DISTANCE,
    rank_max_distance: float = DEFAULT_RANK_MAX_DISTANCE,
    suit_max_distance: float = DEFAULT_SUIT_MAX_DISTANCE,
) -> PokerLegendsCardClassifierManifest:
    selected_screen_kinds = _normalize_screen_kinds(screen_kinds)
    output = Path(output_dir)
    sample_dir = output / "card_classifier_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    annotations = Path(annotation_dir)
    images = Path(image_root)
    samples: list[PokerLegendsCardClassifierSample] = []

    for truth_path in sorted(Path(path) for path in truth_paths):
        truth = _read_json_object(truth_path)
        frame_id = str(truth["frame_id"])
        if _truth_screen_kind(truth) not in selected_screen_kinds:
            continue
        annotation = _read_json_object(annotations / f"{frame_id}.json")
        image_path = images / str(annotation["image"])
        image = _load_rgb_image(image_path)
        regions = _regions_by_truth_group(annotation)

        for item in _truth_cards_for_training(truth, annotation):
            region = regions.get((item.group, item.slot))
            if region is None:
                continue
            crop = _crop(image, _rect_from_region(region))
            visibility_label = "visible" if item.visible else "hidden"
            samples.append(
                _write_classifier_sample(
                    sample_dir=sample_dir,
                    output_root=output,
                    part="visibility",
                    label=visibility_label,
                    card=item.card,
                    frame_id=frame_id,
                    group=item.group,
                    slot=item.slot,
                    image=_normalized_visibility_image_from_card_crop(crop),
                    source_image=str(annotation["image"]),
                )
            )
            if not item.visible or item.card is None:
                continue
            for part, label, sample_image in (
                ("rank", item.card[0], _normalized_part_image_from_card_crop(crop, part="rank")),
                ("suit", item.card[1], _normalized_part_image_from_card_crop(crop, part="suit")),
            ):
                typed_part = cast(ClassifierPart, part)
                samples.append(
                    _write_classifier_sample(
                        sample_dir=sample_dir,
                        output_root=output,
                        part=typed_part,
                        label=label,
                        card=item.card,
                        frame_id=frame_id,
                        group=item.group,
                        slot=item.slot,
                        image=sample_image,
                        source_image=str(annotation["image"]),
                    )
                )

    manifest = PokerLegendsCardClassifierManifest(
        schema_version=1,
        visibility_normalized_width=DEFAULT_VISIBILITY_NORMALIZED_WIDTH,
        visibility_normalized_height=DEFAULT_VISIBILITY_NORMALIZED_HEIGHT,
        rank_normalized_width=DEFAULT_RANK_NORMALIZED_WIDTH,
        rank_normalized_height=DEFAULT_RANK_NORMALIZED_HEIGHT,
        suit_normalized_width=DEFAULT_SUIT_NORMALIZED_WIDTH,
        suit_normalized_height=DEFAULT_SUIT_NORMALIZED_HEIGHT,
        visibility_k=visibility_k,
        rank_k=rank_k,
        suit_k=suit_k,
        visibility_min_weight_ratio=visibility_min_weight_ratio,
        rank_min_weight_ratio=rank_min_weight_ratio,
        suit_min_weight_ratio=suit_min_weight_ratio,
        visibility_max_distance=visibility_max_distance,
        rank_max_distance=rank_max_distance,
        suit_max_distance=suit_max_distance,
        screen_kinds=tuple(selected_screen_kinds),
        samples=tuple(samples),
    )
    manifest.write_json(output / CARD_CLASSIFIER_MANIFEST)
    return manifest


def evaluate_poker_legends_card_classifier(
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
    classifier = PokerLegendsCardClassifier.from_manifest(manifest_file)
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
    visibility_total = 0
    visibility_correct = 0
    visibility_visible_total = 0
    visibility_visible_correct = 0
    visibility_hidden_total = 0
    visibility_hidden_correct = 0

    for truth_path in sorted(Path(path) for path in truth_paths):
        truth = _read_json_object(truth_path)
        frame_id = str(truth["frame_id"])
        if _truth_screen_kind(truth) not in classifier.manifest.screen_kinds:
            continue
        annotation = _read_json_object(annotations / f"{frame_id}.json")
        expected_by_slot = {
            (item.group, item.slot): item for item in _truth_cards_for_scoring(truth, annotation)
        }
        expected_cards = {key: item.card for key, item in expected_by_slot.items() if item.visible}
        predictions: list[PokerLegendsCardClassifierPrediction] = []
        for prediction in classifier.recognize(
            images / str(annotation["image"]),
            annotation,
            frame_id=frame_id,
            exclude_frame_id=frame_id if exclude_same_frame else None,
        ):
            expected_card = expected_cards.get((prediction.group, prediction.slot))
            if exclude_same_card and expected_card is not None:
                prediction = classifier.recognize(
                    images / str(annotation["image"]),
                    _single_region_annotation(annotation, prediction.group, prediction.slot),
                    frame_id=frame_id,
                    exclude_card=expected_card,
                )[0]
            predictions.append(prediction)

        for prediction in predictions:
            expected = expected_by_slot.get((prediction.group, prediction.slot))
            expected_visible = (
                expected is not None and expected.visible and expected.card is not None
            )
            predicted_visible_gate = prediction.visibility_label == "visible"
            visibility_total += 1
            if expected_visible:
                visibility_visible_total += 1
                if predicted_visible_gate:
                    visibility_correct += 1
                    visibility_visible_correct += 1
            else:
                visibility_hidden_total += 1
                if not predicted_visible_gate:
                    visibility_correct += 1
                    visibility_hidden_correct += 1

            if not expected_visible:
                hidden_total += 1
                status = "hidden_match" if prediction.card is None else "false_positive"
                if prediction.card is not None:
                    hidden_false_positive += 1
                expected_card = None
            else:
                visible_total += 1
                expected_item = cast(_TruthCard, expected)
                if expected_item.card is None:
                    raise AssertionError("visible expected card must have a card code")
                expected_card = expected_item.card
                rank_total += 1
                suit_total += 1
                if prediction.rank == expected_card[0]:
                    rank_correct += 1
                if prediction.suit == expected_card[1]:
                    suit_correct += 1
                if prediction.card is None:
                    status = "missing"
                elif prediction.card == expected_card:
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
                    "visibility_label": prediction.visibility_label,
                    "visibility_confidence": prediction.visibility_confidence,
                    "visibility_distance": prediction.visibility_distance,
                    "visibility_weight_ratio": prediction.visibility_weight_ratio,
                    "rank_confidence": prediction.rank_confidence,
                    "rank_distance": prediction.rank_distance,
                    "rank_weight_ratio": prediction.rank_weight_ratio,
                    "suit_confidence": prediction.suit_confidence,
                    "suit_distance": prediction.suit_distance,
                    "suit_weight_ratio": prediction.suit_weight_ratio,
                    "matched_visibility_samples": list(prediction.matched_visibility_samples),
                    "matched_rank_samples": list(prediction.matched_rank_samples),
                    "matched_suit_samples": list(prediction.matched_suit_samples),
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
        "visibility_accuracy": visibility_correct / visibility_total if visibility_total else 1.0,
        "visibility_visible_accuracy": (
            visibility_visible_correct / visibility_visible_total
            if visibility_visible_total
            else 1.0
        ),
        "visibility_hidden_accuracy": (
            visibility_hidden_correct / visibility_hidden_total if visibility_hidden_total else 1.0
        ),
        "rows": rows,
    }
    result_name = f"card_classifier_recognition_{summary['mode']}.json"
    report_name = f"card_classifier_recognition_{summary['mode']}.md"
    (output / result_name).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_card_classifier_recognition_report(output / report_name, summary)
    return summary


def build_and_evaluate_poker_legends_card_classifier(
    truth_paths: Sequence[str | Path],
    *,
    annotation_dir: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    screen_kinds: Sequence[str] = DEFAULT_SCREEN_KINDS,
    visibility_k: int = DEFAULT_VISIBILITY_K,
    rank_k: int = DEFAULT_RANK_K,
    suit_k: int = DEFAULT_SUIT_K,
    visibility_min_weight_ratio: float = DEFAULT_VISIBILITY_MIN_WEIGHT_RATIO,
    rank_min_weight_ratio: float = DEFAULT_RANK_MIN_WEIGHT_RATIO,
    suit_min_weight_ratio: float = DEFAULT_SUIT_MIN_WEIGHT_RATIO,
    visibility_max_distance: float = DEFAULT_VISIBILITY_MAX_DISTANCE,
    rank_max_distance: float = DEFAULT_RANK_MAX_DISTANCE,
    suit_max_distance: float = DEFAULT_SUIT_MAX_DISTANCE,
) -> dict[str, object]:
    manifest = build_poker_legends_card_classifier(
        truth_paths,
        annotation_dir=annotation_dir,
        image_root=image_root,
        output_dir=output_dir,
        screen_kinds=screen_kinds,
        visibility_k=visibility_k,
        rank_k=rank_k,
        suit_k=suit_k,
        visibility_min_weight_ratio=visibility_min_weight_ratio,
        rank_min_weight_ratio=rank_min_weight_ratio,
        suit_min_weight_ratio=suit_min_weight_ratio,
        visibility_max_distance=visibility_max_distance,
        rank_max_distance=rank_max_distance,
        suit_max_distance=suit_max_distance,
    )
    output = Path(output_dir)
    self_eval = evaluate_poker_legends_card_classifier(
        truth_paths,
        annotation_dir=annotation_dir,
        image_root=image_root,
        manifest_path=output / CARD_CLASSIFIER_MANIFEST,
        output_dir=output,
    )
    leave_frame_eval = evaluate_poker_legends_card_classifier(
        truth_paths,
        annotation_dir=annotation_dir,
        image_root=image_root,
        manifest_path=output / CARD_CLASSIFIER_MANIFEST,
        output_dir=output,
        exclude_same_frame=True,
    )
    leave_card_eval = evaluate_poker_legends_card_classifier(
        truth_paths,
        annotation_dir=annotation_dir,
        image_root=image_root,
        manifest_path=output / CARD_CLASSIFIER_MANIFEST,
        output_dir=output,
        exclude_same_card=True,
    )
    sample_counts = Counter(sample.part for sample in manifest.samples)
    visibility_labels = Counter(
        sample.label for sample in manifest.samples if sample.part == "visibility"
    )
    card_codes = sorted(
        {sample.card for sample in manifest.samples if sample.part == "rank" and sample.card}
    )
    rank_labels = sorted({sample.label for sample in manifest.samples if sample.part == "rank"})
    suit_labels = sorted({sample.label for sample in manifest.samples if sample.part == "suit"})
    summary = {
        "schema_version": 1,
        "samples": len(manifest.samples),
        "visibility_samples": sample_counts["visibility"],
        "rank_samples": sample_counts["rank"],
        "suit_samples": sample_counts["suit"],
        "visible_visibility_samples": visibility_labels["visible"],
        "hidden_visibility_samples": visibility_labels["hidden"],
        "unique_cards": len(card_codes),
        "unique_card_codes": card_codes,
        "rank_labels": rank_labels,
        "suit_labels": suit_labels,
        "screen_kinds": list(manifest.screen_kinds),
        "visibility_k": manifest.visibility_k,
        "rank_k": manifest.rank_k,
        "suit_k": manifest.suit_k,
        "visibility_min_weight_ratio": manifest.visibility_min_weight_ratio,
        "rank_min_weight_ratio": manifest.rank_min_weight_ratio,
        "suit_min_weight_ratio": manifest.suit_min_weight_ratio,
        "visibility_max_distance": manifest.visibility_max_distance,
        "rank_max_distance": manifest.rank_max_distance,
        "suit_max_distance": manifest.suit_max_distance,
        "manifest": CARD_CLASSIFIER_MANIFEST,
        "self_eval": _summary_without_rows(self_eval),
        "leave_frame_eval": _summary_without_rows(leave_frame_eval),
        "leave_card_eval": _summary_without_rows(leave_card_eval),
    }
    (output / "card_classifier_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_card_classifier_summary_report(output / "card_classifier_report.md", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and evaluate a local Poker Legends card rank/suit classifier."
    )
    parser.add_argument("truth_overlays", nargs="+", help="Truth overlay JSON files.")
    parser.add_argument(
        "--annotation-dir",
        required=True,
        help="Directory containing Poker Legends draft annotation JSON files.",
    )
    parser.add_argument("--image-root", required=True, help="Root for annotation image paths.")
    parser.add_argument("--out", required=True, help="Output directory for classifier artifacts.")
    parser.add_argument(
        "--screen-kinds",
        default=",".join(DEFAULT_SCREEN_KINDS),
        help="Comma-separated screen kinds to use for samples, or 'all'.",
    )
    parser.add_argument("--visibility-k", type=int, default=DEFAULT_VISIBILITY_K)
    parser.add_argument("--rank-k", type=int, default=DEFAULT_RANK_K)
    parser.add_argument("--suit-k", type=int, default=DEFAULT_SUIT_K)
    parser.add_argument(
        "--visibility-min-weight-ratio",
        type=float,
        default=DEFAULT_VISIBILITY_MIN_WEIGHT_RATIO,
    )
    parser.add_argument(
        "--rank-min-weight-ratio", type=float, default=DEFAULT_RANK_MIN_WEIGHT_RATIO
    )
    parser.add_argument(
        "--suit-min-weight-ratio", type=float, default=DEFAULT_SUIT_MIN_WEIGHT_RATIO
    )
    parser.add_argument(
        "--visibility-max-distance", type=float, default=DEFAULT_VISIBILITY_MAX_DISTANCE
    )
    parser.add_argument("--rank-max-distance", type=float, default=DEFAULT_RANK_MAX_DISTANCE)
    parser.add_argument("--suit-max-distance", type=float, default=DEFAULT_SUIT_MAX_DISTANCE)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = build_and_evaluate_poker_legends_card_classifier(
        args.truth_overlays,
        annotation_dir=args.annotation_dir,
        image_root=args.image_root,
        output_dir=args.out,
        screen_kinds=_parse_screen_kinds(args.screen_kinds),
        visibility_k=args.visibility_k,
        rank_k=args.rank_k,
        suit_k=args.suit_k,
        visibility_min_weight_ratio=args.visibility_min_weight_ratio,
        rank_min_weight_ratio=args.rank_min_weight_ratio,
        suit_min_weight_ratio=args.suit_min_weight_ratio,
        visibility_max_distance=args.visibility_max_distance,
        rank_max_distance=args.rank_max_distance,
        suit_max_distance=args.suit_max_distance,
    )
    print(json.dumps(_stdout_summary(summary), indent=2, sort_keys=True))


@dataclass(frozen=True, slots=True)
class _LoadedSample:
    sample: PokerLegendsCardClassifierSample
    vector: FloatVector


@dataclass(frozen=True, slots=True)
class _KnnVote:
    label: str
    confidence: float
    vote_ratio: float
    weight_ratio: float
    mean_distance: float
    nearest_distance: float
    sample_count: int
    neighbor_images: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _TruthCard:
    group: str
    slot: str
    visible: bool
    card: str | None


def _write_classifier_sample(
    *,
    sample_dir: Path,
    output_root: Path,
    part: ClassifierPart,
    label: str,
    card: str | None,
    frame_id: str,
    group: str,
    slot: str,
    image: RgbImage,
    source_image: str,
) -> PokerLegendsCardClassifierSample:
    relative_image = (
        Path("card_classifier_samples") / part / label / f"{frame_id}__{group}__{slot}.png"
    )
    target = output_root / relative_image
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_rgb_png(target, image)
    return PokerLegendsCardClassifierSample(
        part=part,
        label=label,
        card=card,
        frame_id=frame_id,
        group=group,
        slot=slot,
        image=str(relative_image),
        source_image=source_image,
        sha256=_sha256_file(target),
    )


def _prediction_from_votes(
    *,
    frame_id: str,
    group: str,
    slot: str,
    visibility_vote: _KnnVote | None,
    rank_vote: _KnnVote | None,
    suit_vote: _KnnVote | None,
    rank_ok: bool,
    suit_ok: bool,
) -> PokerLegendsCardClassifierPrediction:
    visibility_ok = visibility_vote is not None and visibility_vote.label == "visible"
    rank = rank_vote.label if rank_vote is not None and rank_ok else None
    suit = suit_vote.label if suit_vote is not None and suit_ok else None
    visible = visibility_ok and rank is not None and suit is not None
    confidence = 0.0
    if visible:
        confidence = min(
            visibility_vote.confidence if visibility_vote is not None else 0.0,
            rank_vote.confidence if rank_vote is not None else 0.0,
            suit_vote.confidence if suit_vote is not None else 0.0,
        )
    return PokerLegendsCardClassifierPrediction(
        frame_id=frame_id,
        group=group,
        slot=slot,
        visible=visible,
        card=f"{rank}{suit}" if visible and rank is not None and suit is not None else None,
        rank=rank,
        suit=suit,
        confidence=confidence,
        visibility_label=None if visibility_vote is None else visibility_vote.label,
        visibility_confidence=0.0 if visibility_vote is None else visibility_vote.confidence,
        visibility_distance=None if visibility_vote is None else visibility_vote.mean_distance,
        visibility_weight_ratio=None if visibility_vote is None else visibility_vote.weight_ratio,
        rank_confidence=0.0 if rank_vote is None else rank_vote.confidence,
        rank_distance=None if rank_vote is None else rank_vote.mean_distance,
        rank_weight_ratio=None if rank_vote is None else rank_vote.weight_ratio,
        suit_confidence=0.0 if suit_vote is None else suit_vote.confidence,
        suit_distance=None if suit_vote is None else suit_vote.mean_distance,
        suit_weight_ratio=None if suit_vote is None else suit_vote.weight_ratio,
        matched_visibility_samples=()
        if visibility_vote is None
        else visibility_vote.neighbor_images,
        matched_rank_samples=() if rank_vote is None else rank_vote.neighbor_images,
        matched_suit_samples=() if suit_vote is None else suit_vote.neighbor_images,
    )


def _accepted_vote(
    vote: _KnnVote | None,
    *,
    expected_label: str | None,
    min_weight_ratio: float,
    max_distance: float,
) -> bool:
    if vote is None:
        return False
    if expected_label is not None and vote.label != expected_label:
        return False
    return vote.weight_ratio >= min_weight_ratio and vote.mean_distance <= max_distance


def _normalized_visibility_image_from_card_crop(crop: RgbImage) -> RgbImage:
    return cast(
        RgbImage,
        cv2.resize(
            crop,
            (DEFAULT_VISIBILITY_NORMALIZED_WIDTH, DEFAULT_VISIBILITY_NORMALIZED_HEIGHT),
            interpolation=cv2.INTER_AREA,
        ),
    )


def _normalized_part_image_from_card_crop(
    crop: RgbImage, *, part: Literal["rank", "suit"]
) -> RgbImage:
    part_crop = _crop_relative(crop, _part_rect(part))
    width, height = _normalized_size(part)
    return cast(RgbImage, cv2.resize(part_crop, (width, height), interpolation=cv2.INTER_AREA))


def _visibility_vector_from_card_crop(crop: RgbImage) -> FloatVector:
    return _visibility_vector_from_image(_normalized_visibility_image_from_card_crop(crop))


def _rank_vector_from_card_crop(crop: RgbImage) -> FloatVector:
    return _rank_vector_from_image(_normalized_part_image_from_card_crop(crop, part="rank"))


def _suit_vector_from_card_crop(crop: RgbImage) -> FloatVector:
    return _suit_vector_from_image(_normalized_part_image_from_card_crop(crop, part="suit"))


def _vector_from_sample_image(image: RgbImage, *, part: ClassifierPart) -> FloatVector:
    if part == "visibility":
        return _visibility_vector_from_image(image)
    if part == "rank":
        return _rank_vector_from_image(image)
    return _suit_vector_from_image(image)


def _visibility_vector_from_image(image: RgbImage) -> FloatVector:
    resized = cast(
        RgbImage,
        cv2.resize(
            image,
            (DEFAULT_VISIBILITY_NORMALIZED_WIDTH, DEFAULT_VISIBILITY_NORMALIZED_HEIGHT),
            interpolation=cv2.INTER_AREA,
        ),
    )
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray).astype(np.float32) / 255.0
    color = resized.astype(np.float32) / 255.0
    mask = _ink_mask(resized).astype(np.float32) / 255.0
    vector = np.concatenate(
        (color.reshape(-1) * 0.55, equalized.reshape(-1) * 0.30, mask.reshape(-1) * 0.15)
    )
    return cast(FloatVector, vector.astype(np.float32))


def _rank_vector_from_image(image: RgbImage) -> FloatVector:
    resized = cast(
        RgbImage,
        cv2.resize(
            image,
            (DEFAULT_RANK_NORMALIZED_WIDTH, DEFAULT_RANK_NORMALIZED_HEIGHT),
            interpolation=cv2.INTER_AREA,
        ),
    )
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray).astype(np.float32) / 255.0
    mask = _ink_mask(resized).astype(np.float32) / 255.0
    edges = cv2.Canny(gray, 40, 120).astype(np.float32) / 255.0
    vector = np.concatenate(
        (equalized.reshape(-1) * 0.50, mask.reshape(-1) * 0.35, edges.reshape(-1) * 0.15)
    )
    return cast(FloatVector, vector.astype(np.float32))


def _suit_vector_from_image(image: RgbImage) -> FloatVector:
    resized = cast(
        RgbImage,
        cv2.resize(
            image,
            (DEFAULT_SUIT_NORMALIZED_WIDTH, DEFAULT_SUIT_NORMALIZED_HEIGHT),
            interpolation=cv2.INTER_AREA,
        ),
    )
    color = resized.astype(np.float32) / 255.0
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray).astype(np.float32) / 255.0
    mask = _ink_mask(resized).astype(np.float32) / 255.0
    vector = np.concatenate(
        (
            color.reshape(-1) * 0.45,
            equalized.reshape(-1) * 0.25,
            mask.reshape(-1) * 0.30,
        )
    )
    return cast(FloatVector, vector.astype(np.float32))


def _ink_mask(image: RgbImage) -> NDArray[np.uint8]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _threshold, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    return cast(NDArray[np.uint8], mask)


def _part_rect(part: Literal["rank", "suit"]) -> tuple[float, float, float, float]:
    if part == "rank":
        return (0.06, 0.03, 0.58, 0.38)
    return (0.10, 0.36, 0.60, 0.48)


def _normalized_size(part: Literal["rank", "suit"]) -> tuple[int, int]:
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


def _mean_square_distance(left: FloatVector, right: FloatVector) -> float:
    return float(np.mean((left - right) ** 2))


def _default_max_distance(
    part: ClassifierPart,
    manifest: PokerLegendsCardClassifierManifest,
) -> float:
    if part == "visibility":
        return manifest.visibility_max_distance
    if part == "rank":
        return manifest.rank_max_distance
    return manifest.suit_max_distance


def _confidence_from_vote(*, weight_ratio: float, distance: float, max_distance: float) -> float:
    if max_distance <= 0:
        distance_confidence = 1.0 if distance == 0 else 0.0
    else:
        distance_confidence = max(0.0, min(1.0, 1.0 - (distance / max_distance)))
    return max(0.0, min(1.0, weight_ratio * distance_confidence))


def _truth_cards_for_training(
    truth: Mapping[str, object],
    annotation: Mapping[str, object],
) -> tuple[_TruthCard, ...]:
    cards: dict[tuple[str, str], _TruthCard] = {}
    ignored = _ignored_fields(truth)
    for truth_group, region_group in (("hero_hole_cards", "cards"), ("board", "board")):
        if truth_group in ignored:
            continue
        for region in _region_group(annotation, region_group):
            slot = str(region["name"])
            cards[(truth_group, slot)] = _TruthCard(
                group=truth_group,
                slot=slot,
                visible=False,
                card=None,
            )
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


def _write_rgb_png(path: str | Path, image: RgbImage) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"could not write sample image: {path}")


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
        "samples": summary["samples"],
        "unique_cards": summary["unique_cards"],
        "rank_labels": summary["rank_labels"],
        "suit_labels": summary["suit_labels"],
        "self_visible_accuracy": self_eval["visible_accuracy"],
        "self_hidden_false_positive_rate": self_eval["hidden_false_positive_rate"],
        "leave_frame_visible_precision": leave_frame_eval["visible_precision"],
        "leave_frame_visible_coverage": leave_frame_eval["visible_coverage"],
        "leave_frame_hidden_false_positive_rate": leave_frame_eval["hidden_false_positive_rate"],
        "leave_card_visible_accuracy": leave_card_eval["visible_accuracy"],
        "leave_card_visible_precision": leave_card_eval["visible_precision"],
        "leave_card_visible_coverage": leave_card_eval["visible_coverage"],
        "leave_card_hidden_false_positive_rate": leave_card_eval["hidden_false_positive_rate"],
        "manifest": summary["manifest"],
    }


def _write_card_classifier_recognition_report(path: Path, summary: Mapping[str, object]) -> None:
    rows = _mapping_sequence(summary["rows"])
    lines = [
        "# Poker Legends Card Classifier Recognition",
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
        f"- Visibility accuracy: {_to_float(summary['visibility_accuracy']):.3f}",
        f"- Visibility visible accuracy: {_to_float(summary['visibility_visible_accuracy']):.3f}",
        f"- Visibility hidden accuracy: {_to_float(summary['visibility_hidden_accuracy']):.3f}",
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
                f"visibility={row['visibility_label']!r}, "
                f"rank_distance={row['rank_distance']!r}, suit_distance={row['suit_distance']!r}"
            )
    else:
        lines.append("No conflicts.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_card_classifier_summary_report(
    path: Path,
    summary: Mapping[str, object],
) -> None:
    self_eval = cast(Mapping[str, object], summary["self_eval"])
    leave_frame_eval = cast(Mapping[str, object], summary["leave_frame_eval"])
    leave_card_eval = cast(Mapping[str, object], summary["leave_card_eval"])
    codes = ", ".join(str(code) for code in _sequence(summary["unique_card_codes"]))
    lines = [
        "# Poker Legends Card Classifier",
        "",
        "## Scope",
        f"- Samples: {summary['samples']}",
        f"- Visibility samples: {summary['visibility_samples']} "
        f"({summary['visible_visibility_samples']} visible / "
        f"{summary['hidden_visibility_samples']} hidden)",
        f"- Rank samples: {summary['rank_samples']}",
        f"- Suit samples: {summary['suit_samples']}",
        f"- Unique card codes: {summary['unique_cards']} / 52",
        f"- Rank labels: {', '.join(str(item) for item in _sequence(summary['rank_labels']))}",
        f"- Suit labels: {', '.join(str(item) for item in _sequence(summary['suit_labels']))}",
        f"- Screen kinds: {', '.join(str(kind) for kind in _sequence(summary['screen_kinds']))}",
        f"- K: visibility={summary['visibility_k']}, rank={summary['rank_k']}, "
        f"suit={summary['suit_k']}",
        f"- Min weight ratio: visibility={summary['visibility_min_weight_ratio']}, "
        f"rank={summary['rank_min_weight_ratio']}, suit={summary['suit_min_weight_ratio']}",
        f"- Max distance: visibility={summary['visibility_max_distance']}, "
        f"rank={summary['rank_max_distance']}, suit={summary['suit_max_distance']}",
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
        f"- Leave-frame hidden false-positive rate: "
        f"{_to_float(leave_frame_eval['hidden_false_positive_rate']):.3f}",
        f"- Leave-card visible precision: {_to_float(leave_card_eval['visible_precision']):.3f}",
        f"- Leave-card visible coverage: {_to_float(leave_card_eval['visible_coverage']):.3f}",
        f"- Leave-card rank accuracy: {_to_float(leave_card_eval['rank_accuracy']):.3f}",
        f"- Leave-card suit accuracy: {_to_float(leave_card_eval['suit_accuracy']):.3f}",
        f"- Leave-card hidden false-positive rate: "
        f"{_to_float(leave_card_eval['hidden_false_positive_rate']):.3f}",
        "",
        "## Notes",
        "- This classifier uses a local visibility gate before rank/suit recognition. It is "
        "intended to be composed with ScreenState and full-card template matching as a "
        "fail-closed signal, not as an always-click oracle.",
        "- Leave-card evaluation excludes samples from the exact expected card code for rank/suit. "
        "It is a proxy for recognizing rank+suit combinations not present as complete cards.",
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
