"""Compare Poker Legends ROI/OCR output against LLM candidate annotations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self, cast

from holdem_bot.vision.annotations import ScreenRect
from holdem_bot.vision.roi_ocr import (
    RoiOcrConfig,
    RoiOcrRecognizer,
    _action_type_from_text,
    _crop,
    _first_int,
    _ints_from_text,
    _load_rgb_image,
    _mean_brightness,
    _normalize_text,
)


@dataclass(frozen=True, slots=True)
class PokerLegendsRoiCard:
    slot: str
    visible: bool
    card: str | None
    confidence: float


@dataclass(frozen=True, slots=True)
class PokerLegendsRoiButton:
    name: str
    visible: bool
    label: str | None
    action_type: str | None
    confidence: float


@dataclass(frozen=True, slots=True)
class PokerLegendsRoiText:
    name: str
    visible: bool
    raw: str
    numbers: tuple[int, ...]
    first_number: int | None
    sum_number: int | None
    confidence: float


@dataclass(frozen=True, slots=True)
class PokerLegendsRoiResult:
    frame_id: str
    image: str
    board: tuple[PokerLegendsRoiCard, ...]
    hero_hole_cards: tuple[PokerLegendsRoiCard, ...]
    buttons: tuple[PokerLegendsRoiButton, ...]
    texts: tuple[PokerLegendsRoiText, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        return cls(
            frame_id=str(data["frame_id"]),
            image=str(data["image"]),
            board=tuple(_roi_card_from_dict(item) for item in _mapping_list(data["board"])),
            hero_hole_cards=tuple(
                _roi_card_from_dict(item) for item in _mapping_list(data["hero_hole_cards"])
            ),
            buttons=tuple(_roi_button_from_dict(item) for item in _mapping_list(data["buttons"])),
            texts=tuple(_roi_text_from_dict(item) for item in _mapping_list(data["texts"])),
        )


@dataclass(frozen=True, slots=True)
class FieldComparison:
    frame_id: str
    group: str
    field: str
    expected: object | None
    observed: object | None
    status: str
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class _Metric:
    correct: int = 0
    total: int = 0
    missing: int = 0
    mismatch: int = 0
    false_positive: int = 0

    @property
    def agreement(self) -> float:
        if self.total == 0:
            return 1.0
        return self.correct / self.total

    def record(self, status: str) -> None:
        if status == "not_comparable":
            return
        self.total += 1
        if status == "match":
            self.correct += 1
        elif status == "missing":
            self.missing += 1
        elif status == "mismatch":
            self.mismatch += 1
        elif status == "false_positive":
            self.false_positive += 1

    def to_dict(self) -> dict[str, object]:
        return {
            "agreement": self.agreement,
            "correct": self.correct,
            "total": self.total,
            "missing": self.missing,
            "mismatch": self.mismatch,
            "false_positive": self.false_positive,
        }


class PokerLegendsRoiOcrRecognizer:
    """First-pass ROI/OCR recognizer for Poker Legends draft-region annotations."""

    def __init__(self, config: RoiOcrConfig | None = None) -> None:
        self.config = config or RoiOcrConfig()
        self._ocr = RoiOcrRecognizer(self.config)

    def recognize(
        self,
        image_path: str | Path,
        annotation: Mapping[str, object],
    ) -> PokerLegendsRoiResult:
        image = _load_rgb_image(image_path)
        regions = _region_groups(annotation)
        return PokerLegendsRoiResult(
            frame_id=Path(str(annotation["image"])).stem,
            image=str(annotation["image"]),
            board=tuple(
                self._recognize_card(image, _rect_from_region(region), str(region["name"]))
                for region in regions.get("board", ())
            ),
            hero_hole_cards=tuple(
                self._recognize_card(image, _rect_from_region(region), str(region["name"]))
                for region in regions.get("cards", ())
            ),
            buttons=tuple(
                self._recognize_button(image, _rect_from_region(region), str(region["name"]))
                for region in regions.get("buttons", ())
            ),
            texts=tuple(
                self._recognize_text(image, _rect_from_region(region), str(region["name"]))
                for region in regions.get("texts", ())
            ),
        )

    def _recognize_card(
        self,
        image: Any,
        rect: ScreenRect,
        slot: str,
    ) -> PokerLegendsRoiCard:
        roi = _crop(image, rect, pad=0)
        visible = _mean_brightness(roi) >= self.config.face_brightness_threshold
        if not visible:
            return PokerLegendsRoiCard(slot=slot, visible=False, card=None, confidence=0.90)
        return PokerLegendsRoiCard(
            slot=slot,
            visible=True,
            card=_card_or_none(self._ocr._ocr_card_code(image, rect)),
            confidence=0.65,
        )

    def _recognize_button(
        self,
        image: Any,
        rect: ScreenRect,
        name: str,
    ) -> PokerLegendsRoiButton:
        raw_text = self._ocr._ocr_button_text(_crop(image, rect, pad=0))
        label = _normalize_text(raw_text)
        action_type = _action_type_from_text(raw_text)
        return PokerLegendsRoiButton(
            name=name,
            visible=bool(label),
            label=label or None,
            action_type=action_type,
            confidence=0.60 if action_type is not None else 0.25,
        )

    def _recognize_text(
        self,
        image: Any,
        rect: ScreenRect,
        name: str,
    ) -> PokerLegendsRoiText:
        raw_text = self._ocr._ocr_text(_crop(image, rect, self.config.crop_padding))
        numbers = tuple(_ints_from_text(raw_text))
        return PokerLegendsRoiText(
            name=name,
            visible=bool(_normalize_text(raw_text)),
            raw=raw_text,
            numbers=numbers,
            first_number=_first_int(raw_text),
            sum_number=sum(numbers) if len(numbers) >= 2 else None,
            confidence=0.70 if numbers else 0.30,
        )


def build_poker_legends_recognition_comparison(
    annotation_paths: Sequence[str | Path],
    *,
    image_root: str | Path,
    candidate_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    annotations = [Path(path) for path in annotation_paths]
    if not annotations:
        raise ValueError("at least one annotation path is required")

    image_base = Path(image_root)
    candidates = Path(candidate_dir)
    output = Path(output_dir)
    roi_dir = output / "roi_ocr_results"
    roi_dir.mkdir(parents=True, exist_ok=True)

    recognizer = PokerLegendsRoiOcrRecognizer()
    frame_rows: list[dict[str, object]] = []
    comparisons: list[FieldComparison] = []
    candidate_count = 0
    actionable_count = 0
    uncertain_count = 0

    for annotation_path in annotations:
        annotation = _read_json_object(annotation_path)
        frame_id = Path(str(annotation["image"])).stem
        candidate_path = candidates / f"{frame_id}.json"
        if not candidate_path.exists():
            raise FileNotFoundError(f"missing LLM candidate annotation: {candidate_path}")
        candidate = _read_json_object(candidate_path)
        candidate_count += 1
        state = _mapping_or_empty(candidate.get("table_state"))
        if bool(state.get("is_actionable")):
            actionable_count += 1
        uncertain = candidate.get("uncertain", [])
        if isinstance(uncertain, list):
            uncertain_count += len(uncertain)

        roi = recognizer.recognize(image_base / str(annotation["image"]), annotation)
        roi.write_json(roi_dir / f"{frame_id}.json")
        frame_comparisons = compare_llm_candidate_to_roi(candidate, roi)
        comparisons.extend(frame_comparisons)
        frame_rows.append(_frame_report_row(frame_id, candidate, roi, frame_comparisons))

    metrics = _build_metrics(comparisons)
    comparison_json = {
        "schema_version": 1,
        "frames": len(annotations),
        "llm_candidates": candidate_count,
        "llm_actionable_frames": actionable_count,
        "llm_non_actionable_frames": candidate_count - actionable_count,
        "llm_uncertain_fields": uncertain_count,
        "metrics": {group: metric.to_dict() for group, metric in sorted(metrics.items())},
        "comparisons": [comparison.to_dict() for comparison in comparisons],
        "roi_ocr_results": str(roi_dir.relative_to(output)),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(comparison_json, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_comparison_report(
        output / "comparison_report.md",
        summary=comparison_json,
        frame_rows=frame_rows,
        comparisons=comparisons,
    )
    return comparison_json


def compare_llm_candidate_to_roi(
    candidate: Mapping[str, object],
    roi: PokerLegendsRoiResult,
) -> list[FieldComparison]:
    frame_id = str(candidate.get("frame_id") or roi.frame_id)
    comparisons: list[FieldComparison] = []
    comparisons.extend(
        _compare_cards(
            frame_id,
            "board",
            _candidate_items(candidate, "board"),
            {card.slot: card for card in roi.board},
        )
    )
    comparisons.extend(
        _compare_cards(
            frame_id,
            "hero_hole_cards",
            _candidate_items(candidate, "hero_hole_cards"),
            {card.slot: card for card in roi.hero_hole_cards},
        )
    )
    comparisons.extend(
        _compare_buttons(
            frame_id,
            _candidate_items(candidate, "buttons"),
            {button.name: button for button in roi.buttons},
        )
    )
    comparisons.extend(
        _compare_texts(
            frame_id,
            _candidate_items(candidate, "texts"),
            {text.name: text for text in roi.texts},
        )
    )
    return comparisons


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Poker Legends ROI/OCR output with LLM candidate annotations."
    )
    parser.add_argument("annotations", nargs="+", help="Poker Legends draft annotation JSON files.")
    parser.add_argument("--image-root", required=True, help="Root for annotation image paths.")
    parser.add_argument(
        "--candidate-dir",
        required=True,
        help="Directory containing LLM candidate annotation JSON files.",
    )
    parser.add_argument("--out", required=True, help="Output directory for comparison artifacts.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = build_poker_legends_recognition_comparison(
        args.annotations,
        image_root=args.image_root,
        candidate_dir=args.candidate_dir,
        output_dir=args.out,
    )
    print(json.dumps(_summary_for_stdout(summary), indent=2, sort_keys=True))


def _compare_cards(
    frame_id: str,
    group: str,
    expected_items: Sequence[Mapping[str, object]],
    observed_by_slot: Mapping[str, PokerLegendsRoiCard],
) -> list[FieldComparison]:
    comparisons: list[FieldComparison] = []
    expected_slots: set[str] = set()
    for item in expected_items:
        slot = _roi_name(item.get("slot") or item.get("name"))
        if not slot:
            continue
        expected_slots.add(slot)
        expected = _card_or_none(item.get("card"))
        visible = bool(item.get("visible"))
        observed = observed_by_slot.get(slot)
        observed_card = None if observed is None else observed.card
        if visible and expected is not None:
            status = _value_status(expected, observed_card)
            comparisons.append(
                FieldComparison(frame_id, group, slot, expected, observed_card, status)
            )
        elif observed_card is not None:
            comparisons.append(
                FieldComparison(
                    frame_id,
                    group,
                    slot,
                    None,
                    observed_card,
                    "false_positive",
                    "LLM candidate marks this card as hidden or absent.",
                )
            )
    for slot, observed in observed_by_slot.items():
        if slot not in expected_slots and observed.card is not None:
            comparisons.append(
                FieldComparison(
                    frame_id,
                    group,
                    slot,
                    None,
                    observed.card,
                    "false_positive",
                    "ROI/OCR produced a card outside the LLM candidate list.",
                )
            )
    return comparisons


def _compare_buttons(
    frame_id: str,
    expected_items: Sequence[Mapping[str, object]],
    observed_by_name: Mapping[str, PokerLegendsRoiButton],
) -> list[FieldComparison]:
    comparisons: list[FieldComparison] = []
    expected_names: set[str] = set()
    for item in expected_items:
        name = _roi_name(item.get("name"))
        if not name:
            continue
        expected_names.add(name)
        expected = _string_or_none(item.get("action_type"))
        visible = bool(item.get("visible"))
        observed = observed_by_name.get(name)
        observed_action = None if observed is None else observed.action_type
        if visible and expected is not None:
            status = _value_status(expected, observed_action)
            comparisons.append(
                FieldComparison(frame_id, "buttons", name, expected, observed_action, status)
            )
        elif observed_action is not None:
            comparisons.append(
                FieldComparison(
                    frame_id,
                    "buttons",
                    name,
                    None,
                    observed_action,
                    "false_positive",
                    "LLM candidate marks this button as hidden or non-action.",
                )
            )
    for name, observed in observed_by_name.items():
        if name not in expected_names and observed.action_type is not None:
            comparisons.append(
                FieldComparison(
                    frame_id,
                    "buttons",
                    name,
                    None,
                    observed.action_type,
                    "false_positive",
                    "ROI/OCR produced a button outside the LLM candidate list.",
                )
            )
    return comparisons


def _compare_texts(
    frame_id: str,
    expected_items: Sequence[Mapping[str, object]],
    observed_by_name: Mapping[str, PokerLegendsRoiText],
) -> list[FieldComparison]:
    comparisons: list[FieldComparison] = []
    expected_names: set[str] = set()
    for item in expected_items:
        name = _roi_name(item.get("name"))
        if not name:
            continue
        expected_names.add(name)
        expected = _optional_int(item.get("normalized_number"))
        visible = bool(item.get("visible"))
        observed = observed_by_name.get(name)
        observed_value = None if observed is None else _best_text_number(observed, expected)
        if visible and expected is not None:
            status = _value_status(expected, observed_value)
            comparisons.append(
                FieldComparison(frame_id, "text_numbers", name, expected, observed_value, status)
            )
        elif observed is not None and observed.first_number is not None:
            comparisons.append(
                FieldComparison(
                    frame_id,
                    "text_numbers",
                    name,
                    None,
                    observed.first_number,
                    "false_positive",
                    "LLM candidate marks this text as hidden or non-numeric.",
                )
            )
    for name, observed in observed_by_name.items():
        if name not in expected_names and observed.first_number is not None:
            comparisons.append(
                FieldComparison(
                    frame_id,
                    "text_numbers",
                    name,
                    None,
                    observed.first_number,
                    "false_positive",
                    "ROI/OCR produced numeric text outside the LLM candidate list.",
                )
            )
    return comparisons


def _build_metrics(comparisons: Sequence[FieldComparison]) -> dict[str, _Metric]:
    metrics: dict[str, _Metric] = {
        "cards": _Metric(),
        "buttons": _Metric(),
        "text_numbers": _Metric(),
    }
    for comparison in comparisons:
        group = "cards" if comparison.group in {"board", "hero_hole_cards"} else comparison.group
        metrics.setdefault(group, _Metric()).record(comparison.status)
    metrics["overall"] = _Metric(
        correct=sum(metric.correct for name, metric in metrics.items() if name != "overall"),
        total=sum(metric.total for name, metric in metrics.items() if name != "overall"),
        missing=sum(metric.missing for name, metric in metrics.items() if name != "overall"),
        mismatch=sum(metric.mismatch for name, metric in metrics.items() if name != "overall"),
        false_positive=sum(
            metric.false_positive for name, metric in metrics.items() if name != "overall"
        ),
    )
    return metrics


def _write_comparison_report(
    path: Path,
    *,
    summary: Mapping[str, object],
    frame_rows: Sequence[Mapping[str, object]],
    comparisons: Sequence[FieldComparison],
) -> None:
    metrics = cast(Mapping[str, Mapping[str, object]], summary["metrics"])
    lines = [
        "# Poker Legends LLM vs ROI/OCR Comparison",
        "",
        "## Scope",
        f"- Frames: {summary['frames']}",
        f"- LLM candidates: {summary['llm_candidates']}",
        f"- LLM actionable frames: {summary['llm_actionable_frames']}",
        f"- LLM non-actionable/blocking frames: {summary['llm_non_actionable_frames']}",
        f"- LLM uncertain fields: {summary['llm_uncertain_fields']}",
        "- The numbers below are agreement with the LLM candidate annotations, "
        "not final ground-truth accuracy.",
        "",
        "## Agreement",
        "| Group | Correct | Compared | Agreement | Missing | Mismatch | False positives |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in ("cards", "buttons", "text_numbers", "overall"):
        metric = metrics[group]
        lines.append(
            f"| `{group}` | {metric['correct']} | {metric['total']} | "
            f"{_to_float(metric['agreement']):.3f} | {metric['missing']} | "
            f"{metric['mismatch']} | {metric['false_positive']} |"
        )
    lines.extend(
        [
            "",
            "## Frame Overview",
            "| Frame | LLM state | LLM cards | ROI/OCR cards | LLM buttons | ROI/OCR buttons | "
            "Conflicts | Uncertain |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in frame_rows:
        lines.append(
            f"| `{row['frame_id']}` | {row['state']} | {row['llm_cards']} | "
            f"{row['roi_cards']} | {row['llm_buttons']} | {row['roi_buttons']} | "
            f"{row['conflicts']} | {row['uncertain']} |"
        )

    conflicts = [
        comparison
        for comparison in comparisons
        if comparison.status not in {"match", "not_comparable"}
    ]
    lines.extend(["", "## Conflicts"])
    if conflicts:
        for comparison in conflicts:
            note = f" - {comparison.note}" if comparison.note else ""
            lines.append(
                f"- `{comparison.frame_id}` `{comparison.group}.{comparison.field}`: "
                f"{comparison.status}; LLM={comparison.expected!r}, "
                f"ROI/OCR={comparison.observed!r}{note}"
            )
    else:
        lines.append("No conflicts against the LLM candidate annotations.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _frame_report_row(
    frame_id: str,
    candidate: Mapping[str, object],
    roi: PokerLegendsRoiResult,
    comparisons: Sequence[FieldComparison],
) -> dict[str, object]:
    state = _mapping_or_empty(candidate.get("table_state"))
    blocking_reason = state.get("blocking_reason")
    state_label = "actionable" if bool(state.get("is_actionable")) else str(blocking_reason)
    uncertain = candidate.get("uncertain", [])
    return {
        "frame_id": frame_id,
        "state": state_label,
        "llm_cards": _format_candidate_cards(candidate),
        "roi_cards": _format_roi_cards((*roi.hero_hole_cards, *roi.board)),
        "llm_buttons": _format_candidate_buttons(candidate),
        "roi_buttons": _format_roi_buttons(roi.buttons),
        "conflicts": sum(
            1 for comparison in comparisons if comparison.status not in {"match", "not_comparable"}
        ),
        "uncertain": len(uncertain) if isinstance(uncertain, list) else 0,
    }


def _format_candidate_cards(candidate: Mapping[str, object]) -> str:
    cards = []
    for key in ("hero_hole_cards", "board"):
        for item in _candidate_items(candidate, key):
            card = item.get("card")
            if isinstance(card, str):
                cards.append(card)
    return " ".join(cards) if cards else "-"


def _format_roi_cards(cards: Sequence[PokerLegendsRoiCard]) -> str:
    values = [card.card for card in cards if card.card is not None]
    return " ".join(values) if values else "-"


def _format_candidate_buttons(candidate: Mapping[str, object]) -> str:
    actions = []
    for item in _candidate_items(candidate, "buttons"):
        action_type = item.get("action_type")
        if isinstance(action_type, str):
            actions.append(action_type)
    return " ".join(actions) if actions else "-"


def _format_roi_buttons(buttons: Sequence[PokerLegendsRoiButton]) -> str:
    actions = [button.action_type for button in buttons if button.action_type is not None]
    return " ".join(actions) if actions else "-"


def _summary_for_stdout(summary: Mapping[str, object]) -> dict[str, object]:
    metrics = cast(Mapping[str, Mapping[str, object]], summary["metrics"])
    return {
        "frames": summary["frames"],
        "llm_candidates": summary["llm_candidates"],
        "llm_uncertain_fields": summary["llm_uncertain_fields"],
        "overall_agreement": metrics["overall"]["agreement"],
        "report": "comparison_report.md",
        "comparison": "comparison.json",
    }


def _best_text_number(text: PokerLegendsRoiText, expected: int | None) -> int | None:
    if expected is not None:
        if text.first_number == expected:
            return text.first_number
        if expected in text.numbers:
            return expected
        if text.sum_number == expected:
            return text.sum_number
    return text.first_number


def _value_status(expected: object, observed: object | None) -> str:
    if observed is None:
        return "missing"
    if observed == expected:
        return "match"
    return "mismatch"


def _region_groups(annotation: Mapping[str, object]) -> dict[str, list[Mapping[str, object]]]:
    raw = annotation.get("regions", {})
    if not isinstance(raw, Mapping):
        return {}
    groups: dict[str, list[Mapping[str, object]]] = {}
    for group, regions in raw.items():
        if not isinstance(group, str) or not isinstance(regions, list):
            continue
        groups[group] = [region for region in regions if isinstance(region, Mapping)]
    return groups


def _rect_from_region(region: Mapping[str, object]) -> ScreenRect:
    rect = region.get("rect", {})
    if not isinstance(rect, Mapping):
        raise ValueError(f"region has no rect: {region}")
    return ScreenRect(
        x=_to_int(rect["x"]),
        y=_to_int(rect["y"]),
        width=_to_int(rect["width"]),
        height=_to_int(rect["height"]),
    )


def _read_json_object(path: str | Path) -> dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, object], data)


def _candidate_items(candidate: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    value = candidate.get(key, [])
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _mapping_or_empty(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def _mapping_list(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        raise ValueError(f"expected list of objects: {value!r}")
    return [item for item in value if isinstance(item, Mapping)]


def _roi_card_from_dict(data: Mapping[str, object]) -> PokerLegendsRoiCard:
    return PokerLegendsRoiCard(
        slot=str(data["slot"]),
        visible=bool(data["visible"]),
        card=_string_or_none(data["card"]),
        confidence=_to_float(data["confidence"]),
    )


def _roi_button_from_dict(data: Mapping[str, object]) -> PokerLegendsRoiButton:
    return PokerLegendsRoiButton(
        name=str(data["name"]),
        visible=bool(data["visible"]),
        label=_string_or_none(data["label"]),
        action_type=_string_or_none(data["action_type"]),
        confidence=_to_float(data["confidence"]),
    )


def _roi_text_from_dict(data: Mapping[str, object]) -> PokerLegendsRoiText:
    return PokerLegendsRoiText(
        name=str(data["name"]),
        visible=bool(data["visible"]),
        raw=str(data["raw"]),
        numbers=tuple(_to_int(item) for item in _sequence(data["numbers"])),
        first_number=_optional_int(data["first_number"]),
        sum_number=_optional_int(data["sum_number"]),
        confidence=_to_float(data["confidence"]),
    )


def _sequence(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"expected sequence: {value!r}")
    return value


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _card_or_none(value: object) -> str | None:
    card = _string_or_none(value)
    if card is None:
        return None
    return card.upper()


def _roi_name(value: object) -> str:
    if value is None:
        return ""
    return str(value).rsplit(".", maxsplit=1)[-1]


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return _to_int(value)


def _to_int(value: object) -> int:
    if type(value) is int:
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"expected int-compatible value: {value!r}")


def _to_float(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"expected float-compatible value: {value!r}")
