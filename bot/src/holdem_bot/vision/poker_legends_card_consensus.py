"""Consensus card recognition for Poker Legends cards."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self, cast

from holdem_bot.screen_state import ScreenKind
from holdem_bot.vision.poker_legends_card_classifier import (
    PokerLegendsCardClassifier,
    PokerLegendsCardClassifierPrediction,
    _mapping_sequence,
    _normalize_screen_kinds,
    _read_json_object,
    _sequence,
    _single_region_annotation,
    _summary_without_rows,
    _to_float,
    _truth_cards_for_scoring,
    _truth_screen_kind,
    _TruthCard,
)
from holdem_bot.vision.poker_legends_card_parts import (
    PokerLegendsCardPartPrediction,
    PokerLegendsCardPartTemplateRecognizer,
)
from holdem_bot.vision.poker_legends_cards import (
    PokerLegendsCardPrediction,
    PokerLegendsCardTemplateRecognizer,
)

DEFAULT_SCREEN_KINDS = (ScreenKind.ACTIONABLE_TABLE.value,)


@dataclass(frozen=True, slots=True)
class PokerLegendsCardConsensusPrediction:
    frame_id: str
    group: str
    slot: str
    visible: bool
    card: str | None
    confidence: float
    method: str
    full_card: str | None
    part_card: str | None
    classifier_card: str | None
    full_confidence: float | None
    part_confidence: float | None
    classifier_confidence: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class PokerLegendsCardConsensusRecognizer:
    def __init__(
        self,
        *,
        part_recognizer: PokerLegendsCardPartTemplateRecognizer,
        classifier: PokerLegendsCardClassifier,
        full_card_recognizer: PokerLegendsCardTemplateRecognizer | None = None,
    ) -> None:
        self.part_recognizer = part_recognizer
        self.classifier = classifier
        self.full_card_recognizer = full_card_recognizer

    @classmethod
    def from_manifests(
        cls,
        *,
        card_part_manifest: str | Path,
        card_classifier_manifest: str | Path,
        card_template_manifest: str | Path | None = None,
    ) -> Self:
        return cls(
            part_recognizer=PokerLegendsCardPartTemplateRecognizer.from_manifest(
                card_part_manifest
            ),
            classifier=PokerLegendsCardClassifier.from_manifest(card_classifier_manifest),
            full_card_recognizer=None
            if card_template_manifest is None
            else PokerLegendsCardTemplateRecognizer.from_manifest(card_template_manifest),
        )

    def recognize(
        self,
        image_path: str | Path,
        annotation: Mapping[str, object],
        *,
        frame_id: str,
        exclude_frame_id: str | None = None,
        exclude_card: str | None = None,
    ) -> tuple[PokerLegendsCardConsensusPrediction, ...]:
        full_predictions = _prediction_map(
            ()
            if self.full_card_recognizer is None or exclude_card is not None
            else self.full_card_recognizer.recognize(
                image_path,
                annotation,
                frame_id=frame_id,
                exclude_frame_id=exclude_frame_id,
            )
        )
        part_predictions = _prediction_map(
            self.part_recognizer.recognize(
                image_path,
                annotation,
                frame_id=frame_id,
                exclude_frame_id=exclude_frame_id,
                exclude_card=exclude_card,
            )
        )
        classifier_predictions = _prediction_map(
            self.classifier.recognize(
                image_path,
                annotation,
                frame_id=frame_id,
                exclude_frame_id=exclude_frame_id,
                exclude_card=exclude_card,
            )
        )
        keys = tuple(dict.fromkeys([*part_predictions, *classifier_predictions, *full_predictions]))
        return tuple(
            _consensus_prediction(
                frame_id=frame_id,
                group=group,
                slot=slot,
                full_prediction=cast(PokerLegendsCardPrediction | None, full_predictions.get(key)),
                part_prediction=cast(PokerLegendsCardPartPrediction, part_predictions[key]),
                classifier_prediction=cast(
                    PokerLegendsCardClassifierPrediction,
                    classifier_predictions[key],
                ),
            )
            for key in keys
            for group, slot in (key,)
            if key in part_predictions and key in classifier_predictions
        )


def evaluate_poker_legends_card_consensus(
    truth_paths: Sequence[str | Path],
    *,
    annotation_dir: str | Path,
    image_root: str | Path,
    card_part_manifest: str | Path,
    card_classifier_manifest: str | Path,
    output_dir: str | Path,
    card_template_manifest: str | Path | None = None,
    screen_kinds: Sequence[str] = DEFAULT_SCREEN_KINDS,
    exclude_same_frame: bool = False,
    exclude_same_card: bool = False,
) -> dict[str, object]:
    if exclude_same_frame and exclude_same_card:
        raise ValueError("only one exclusion mode can be enabled")
    selected_screen_kinds = _normalize_screen_kinds(screen_kinds)
    recognizer = PokerLegendsCardConsensusRecognizer.from_manifests(
        card_part_manifest=card_part_manifest,
        card_classifier_manifest=card_classifier_manifest,
        card_template_manifest=None if exclude_same_card else card_template_manifest,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    annotations = Path(annotation_dir)
    images = Path(image_root)
    rows: list[dict[str, object]] = []
    method_counts: Counter[str] = Counter()
    visible_total = 0
    visible_predicted = 0
    visible_correct = 0
    hidden_total = 0
    hidden_false_positive = 0

    for truth_path in sorted(Path(path) for path in truth_paths):
        truth = _read_json_object(truth_path)
        frame_id = str(truth["frame_id"])
        if _truth_screen_kind(truth) not in selected_screen_kinds:
            continue
        annotation = _read_json_object(annotations / f"{frame_id}.json")
        image_path = images / str(annotation["image"])
        expected_by_slot = {
            (item.group, item.slot): item for item in _truth_cards_for_scoring(truth, annotation)
        }
        predictions: list[PokerLegendsCardConsensusPrediction] = []
        for prediction in recognizer.recognize(
            image_path,
            annotation,
            frame_id=frame_id,
            exclude_frame_id=frame_id if exclude_same_frame else None,
        ):
            expected = expected_by_slot.get((prediction.group, prediction.slot))
            expected_card = expected.card if expected is not None and expected.visible else None
            if exclude_same_card and expected_card is not None:
                prediction = recognizer.recognize(
                    image_path,
                    _single_region_annotation(annotation, prediction.group, prediction.slot),
                    frame_id=frame_id,
                    exclude_card=expected_card,
                )[0]
            predictions.append(prediction)

        for prediction in predictions:
            expected = expected_by_slot.get((prediction.group, prediction.slot))
            expected_card = _expected_card(expected)
            method_counts[prediction.method] += 1
            if expected_card is None:
                hidden_total += 1
                status = "hidden_match" if prediction.card is None else "false_positive"
                if prediction.card is not None:
                    hidden_false_positive += 1
            else:
                visible_total += 1
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
                    "status": status,
                    "confidence": prediction.confidence,
                    "method": prediction.method,
                    "full_card": prediction.full_card,
                    "part_card": prediction.part_card,
                    "classifier_card": prediction.classifier_card,
                    "full_confidence": prediction.full_confidence,
                    "part_confidence": prediction.part_confidence,
                    "classifier_confidence": prediction.classifier_confidence,
                }
            )

    mode = _evaluation_mode(
        exclude_same_frame=exclude_same_frame,
        exclude_same_card=exclude_same_card,
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "mode": mode,
        "frames": len({str(row["frame_id"]) for row in rows}),
        "screen_kinds": list(selected_screen_kinds),
        "visible_total": visible_total,
        "visible_predicted": visible_predicted,
        "visible_correct": visible_correct,
        "visible_accuracy": visible_correct / visible_total if visible_total else 1.0,
        "visible_precision": visible_correct / visible_predicted if visible_predicted else 1.0,
        "visible_coverage": visible_predicted / visible_total if visible_total else 1.0,
        "hidden_total": hidden_total,
        "hidden_false_positive": hidden_false_positive,
        "hidden_false_positive_rate": hidden_false_positive / hidden_total if hidden_total else 0.0,
        "method_counts": dict(sorted(method_counts.items())),
        "card_template_manifest": None
        if exclude_same_card
        else (None if card_template_manifest is None else str(card_template_manifest)),
        "card_part_manifest": str(card_part_manifest),
        "card_classifier_manifest": str(card_classifier_manifest),
        "rows": rows,
    }
    (output / f"card_consensus_recognition_{mode}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_card_consensus_recognition_report(
        output / f"card_consensus_recognition_{mode}.md",
        summary,
    )
    return summary


def evaluate_poker_legends_card_consensus_suite(
    truth_paths: Sequence[str | Path],
    *,
    annotation_dir: str | Path,
    image_root: str | Path,
    card_part_manifest: str | Path,
    card_classifier_manifest: str | Path,
    output_dir: str | Path,
    card_template_manifest: str | Path | None = None,
    screen_kinds: Sequence[str] = DEFAULT_SCREEN_KINDS,
) -> dict[str, object]:
    output = Path(output_dir)
    self_eval = evaluate_poker_legends_card_consensus(
        truth_paths,
        annotation_dir=annotation_dir,
        image_root=image_root,
        card_part_manifest=card_part_manifest,
        card_classifier_manifest=card_classifier_manifest,
        card_template_manifest=card_template_manifest,
        output_dir=output,
        screen_kinds=screen_kinds,
    )
    leave_frame_eval = evaluate_poker_legends_card_consensus(
        truth_paths,
        annotation_dir=annotation_dir,
        image_root=image_root,
        card_part_manifest=card_part_manifest,
        card_classifier_manifest=card_classifier_manifest,
        card_template_manifest=card_template_manifest,
        output_dir=output,
        screen_kinds=screen_kinds,
        exclude_same_frame=True,
    )
    leave_card_eval = evaluate_poker_legends_card_consensus(
        truth_paths,
        annotation_dir=annotation_dir,
        image_root=image_root,
        card_part_manifest=card_part_manifest,
        card_classifier_manifest=card_classifier_manifest,
        card_template_manifest=card_template_manifest,
        output_dir=output,
        screen_kinds=screen_kinds,
        exclude_same_card=True,
    )
    summary = {
        "schema_version": 1,
        "screen_kinds": list(_normalize_screen_kinds(screen_kinds)),
        "card_template_manifest": None
        if card_template_manifest is None
        else str(card_template_manifest),
        "card_part_manifest": str(card_part_manifest),
        "card_classifier_manifest": str(card_classifier_manifest),
        "self_eval": _summary_without_rows(self_eval),
        "leave_frame_eval": _summary_without_rows(leave_frame_eval),
        "leave_card_eval": _summary_without_rows(leave_card_eval),
    }
    (output / "card_consensus_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_card_consensus_summary_report(output / "card_consensus_report.md", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Poker Legends full-card + rank/suit consensus recognition."
    )
    parser.add_argument("truth_overlays", nargs="+", help="Truth overlay JSON files.")
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--card-template-manifest")
    parser.add_argument("--card-part-manifest", required=True)
    parser.add_argument("--card-classifier-manifest", required=True)
    parser.add_argument(
        "--screen-kinds",
        default=",".join(DEFAULT_SCREEN_KINDS),
        help="Comma-separated screen kinds to evaluate, or 'all'.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = evaluate_poker_legends_card_consensus_suite(
        args.truth_overlays,
        annotation_dir=args.annotation_dir,
        image_root=args.image_root,
        output_dir=args.out,
        card_template_manifest=args.card_template_manifest,
        card_part_manifest=args.card_part_manifest,
        card_classifier_manifest=args.card_classifier_manifest,
        screen_kinds=_parse_screen_kinds(args.screen_kinds),
    )
    print(json.dumps(_stdout_summary(summary), indent=2, sort_keys=True))


def _prediction_map(
    predictions: Sequence[
        PokerLegendsCardPrediction
        | PokerLegendsCardPartPrediction
        | PokerLegendsCardClassifierPrediction
    ],
) -> dict[tuple[str, str], object]:
    return {(prediction.group, prediction.slot): prediction for prediction in predictions}


def _consensus_prediction(
    *,
    frame_id: str,
    group: str,
    slot: str,
    full_prediction: PokerLegendsCardPrediction | None,
    part_prediction: PokerLegendsCardPartPrediction,
    classifier_prediction: PokerLegendsCardClassifierPrediction,
) -> PokerLegendsCardConsensusPrediction:
    if full_prediction is not None and full_prediction.card is not None:
        return PokerLegendsCardConsensusPrediction(
            frame_id=frame_id,
            group=group,
            slot=slot,
            visible=True,
            card=full_prediction.card,
            confidence=full_prediction.confidence,
            method="full_card",
            full_card=full_prediction.card,
            part_card=part_prediction.card,
            classifier_card=classifier_prediction.card,
            full_confidence=full_prediction.confidence,
            part_confidence=part_prediction.confidence,
            classifier_confidence=classifier_prediction.confidence,
        )
    if (
        part_prediction.card is not None
        and classifier_prediction.card is not None
        and part_prediction.card == classifier_prediction.card
    ):
        confidence = min(part_prediction.confidence, classifier_prediction.confidence)
        return PokerLegendsCardConsensusPrediction(
            frame_id=frame_id,
            group=group,
            slot=slot,
            visible=True,
            card=part_prediction.card,
            confidence=confidence,
            method="part_classifier_consensus",
            full_card=None if full_prediction is None else full_prediction.card,
            part_card=part_prediction.card,
            classifier_card=classifier_prediction.card,
            full_confidence=None if full_prediction is None else full_prediction.confidence,
            part_confidence=part_prediction.confidence,
            classifier_confidence=classifier_prediction.confidence,
        )
    return PokerLegendsCardConsensusPrediction(
        frame_id=frame_id,
        group=group,
        slot=slot,
        visible=False,
        card=None,
        confidence=0.0,
        method="none",
        full_card=None if full_prediction is None else full_prediction.card,
        part_card=part_prediction.card,
        classifier_card=classifier_prediction.card,
        full_confidence=None if full_prediction is None else full_prediction.confidence,
        part_confidence=part_prediction.confidence,
        classifier_confidence=classifier_prediction.confidence,
    )


def _expected_card(expected: _TruthCard | None) -> str | None:
    if expected is None or not expected.visible or expected.card is None:
        return None
    return expected.card


def _evaluation_mode(*, exclude_same_frame: bool, exclude_same_card: bool) -> str:
    if exclude_same_frame:
        return "leave_frame"
    if exclude_same_card:
        return "leave_card"
    return "self"


def _parse_screen_kinds(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() == "all":
        return tuple(kind.value for kind in ScreenKind)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _stdout_summary(summary: Mapping[str, object]) -> dict[str, object]:
    self_eval = cast(Mapping[str, object], summary["self_eval"])
    leave_frame_eval = cast(Mapping[str, object], summary["leave_frame_eval"])
    leave_card_eval = cast(Mapping[str, object], summary["leave_card_eval"])
    return {
        "self_visible_accuracy": self_eval["visible_accuracy"],
        "self_hidden_false_positive_rate": self_eval["hidden_false_positive_rate"],
        "leave_frame_visible_precision": leave_frame_eval["visible_precision"],
        "leave_frame_visible_coverage": leave_frame_eval["visible_coverage"],
        "leave_frame_hidden_false_positive_rate": leave_frame_eval["hidden_false_positive_rate"],
        "leave_card_visible_precision": leave_card_eval["visible_precision"],
        "leave_card_visible_coverage": leave_card_eval["visible_coverage"],
        "leave_card_hidden_false_positive_rate": leave_card_eval["hidden_false_positive_rate"],
    }


def _write_card_consensus_recognition_report(
    path: Path,
    summary: Mapping[str, object],
) -> None:
    rows = _mapping_sequence(summary["rows"])
    lines = [
        "# Poker Legends Card Consensus Recognition",
        "",
        "## Summary",
        f"- Mode: `{summary['mode']}`",
        f"- Frames: {summary['frames']}",
        f"- Screen kinds: {', '.join(str(kind) for kind in _sequence(summary['screen_kinds']))}",
        f"- Visible cards: {summary['visible_total']}",
        f"- Visible correct: {summary['visible_correct']}",
        f"- Visible accuracy: {_to_float(summary['visible_accuracy']):.3f}",
        f"- Visible precision: {_to_float(summary['visible_precision']):.3f}",
        f"- Visible coverage: {_to_float(summary['visible_coverage']):.3f}",
        f"- Hidden false positives: {summary['hidden_false_positive']} / {summary['hidden_total']}",
        f"- Method counts: {summary['method_counts']}",
        "",
        "## Conflicts",
    ]
    conflicts = [row for row in rows if row.get("status") not in {"match", "hidden_match"}]
    if conflicts:
        for row in conflicts:
            lines.append(
                f"- `{row['frame_id']}` `{row['group']}.{row['slot']}`: "
                f"{row['status']}; expected={row['expected']!r}, observed={row['observed']!r}, "
                f"method={row['method']!r}, full={row['full_card']!r}, "
                f"part={row['part_card']!r}, classifier={row['classifier_card']!r}"
            )
    else:
        lines.append("No conflicts.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_card_consensus_summary_report(
    path: Path,
    summary: Mapping[str, object],
) -> None:
    self_eval = cast(Mapping[str, object], summary["self_eval"])
    leave_frame_eval = cast(Mapping[str, object], summary["leave_frame_eval"])
    leave_card_eval = cast(Mapping[str, object], summary["leave_card_eval"])
    lines = [
        "# Poker Legends Card Consensus",
        "",
        "## Scope",
        f"- Screen kinds: {', '.join(str(kind) for kind in _sequence(summary['screen_kinds']))}",
        f"- Card template manifest: `{summary['card_template_manifest']}`",
        f"- Card part manifest: `{summary['card_part_manifest']}`",
        f"- Card classifier manifest: `{summary['card_classifier_manifest']}`",
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
        f"- Leave-card hidden false-positive rate: "
        f"{_to_float(leave_card_eval['hidden_false_positive_rate']):.3f}",
        "",
        "## Notes",
        "- Runtime strategy: use full-card template when it passes; otherwise only accept a card "
        "when the part-template recognizer and classifier agree on the same rank+suit.",
        "- Leave-card evaluation disables full-card templates so it measures the "
        "unseen-combination fallback rather than exact full-card memorization.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
