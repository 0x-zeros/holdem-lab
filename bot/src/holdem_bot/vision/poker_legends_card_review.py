"""Card-focused review candidate selection for Poker Legends videos."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from holdem_bot.screen_state import ScreenKind
from holdem_bot.vision.poker_legends_card_parts import (
    PokerLegendsCardPartPrediction,
    PokerLegendsCardPartTemplateRecognizer,
)
from holdem_bot.vision.poker_legends_cards import (
    PokerLegendsCardPrediction,
    PokerLegendsCardTemplateRecognizer,
)
from holdem_bot.vision.poker_legends_layout import (
    draw_layout_overlay,
    poker_legends_layout_regions,
)
from holdem_bot.vision.poker_legends_screen import detect_poker_legends_screen_state
from holdem_bot.vision.video import ExtractedFrame, VideoIngestManifest

RgbImage = NDArray[np.uint8]

DEFAULT_MAX_CANDIDATES = 40
DEFAULT_MAX_TARGET_PER_CARD = 6
DEFAULT_SPACED_SAMPLE_SECONDS = 90.0
DEFAULT_SCREEN_KINDS = (
    ScreenKind.ACTIONABLE_TABLE.value,
    ScreenKind.TABLE_OBSERVE.value,
)


@dataclass(frozen=True, slots=True)
class PokerLegendsCardReviewSource:
    name: str
    manifest: str | Path

    @classmethod
    def from_args(cls, args: Sequence[str]) -> Self:
        if len(args) != 2:
            raise ValueError("source requires NAME and MANIFEST")
        return cls(name=args[0], manifest=args[1])


@dataclass(frozen=True, slots=True)
class PokerLegendsExcludedTruth:
    source: str
    path: str | Path

    @classmethod
    def from_args(cls, args: Sequence[str]) -> Self:
        if len(args) != 2:
            raise ValueError("exclude truth requires SOURCE and PATH")
        return cls(source=args[0], path=args[1])


@dataclass(frozen=True, slots=True)
class PokerLegendsCardReviewCandidate:
    frame_id: str
    source: str
    original_frame_id: str
    image: str
    annotation: str
    source_image: str
    source_annotation: str
    frame_index: int
    timestamp_seconds: float
    screen_kind: str
    category: str
    note: str
    full_cards: tuple[str, ...]
    part_cards: tuple[str, ...]
    target_hits: tuple[str, ...]
    gap_slots: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "source": self.source,
            "original_frame_id": self.original_frame_id,
            "image": self.image,
            "annotation": self.annotation,
            "source_image": self.source_image,
            "source_annotation": self.source_annotation,
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "screen_kind": self.screen_kind,
            "category": self.category,
            "note": self.note,
            "full_cards": list(self.full_cards),
            "part_cards": list(self.part_cards),
            "target_hits": list(self.target_hits),
            "gap_slots": list(self.gap_slots),
        }


@dataclass(frozen=True, slots=True)
class PokerLegendsCardReviewSelection:
    schema_version: int
    sources: tuple[PokerLegendsCardReviewSource, ...]
    target_cards: tuple[str, ...]
    selected: tuple[PokerLegendsCardReviewCandidate, ...]
    scanned_frames: int
    eligible_frames: int
    excluded_frames: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sources": [
                {"name": source.name, "manifest": str(source.manifest)} for source in self.sources
            ],
            "target_cards": list(self.target_cards),
            "scanned_frames": self.scanned_frames,
            "eligible_frames": self.eligible_frames,
            "excluded_frames": self.excluded_frames,
            "frames": [candidate.to_dict() for candidate in self.selected],
        }


@dataclass(frozen=True, slots=True)
class _AnalyzedFrame:
    source: PokerLegendsCardReviewSource
    source_dir: Path
    source_frame: ExtractedFrame
    frame_id: str
    original_frame_id: str
    image_path: Path
    annotation: dict[str, object]
    screen_kind: str
    full_predictions: tuple[PokerLegendsCardPrediction, ...]
    part_predictions: tuple[PokerLegendsCardPartPrediction, ...]
    target_hits: tuple[str, ...]
    gap_slots: tuple[str, ...]

    @property
    def timestamp_seconds(self) -> float:
        return self.source_frame.timestamp_seconds

    @property
    def full_cards(self) -> tuple[str, ...]:
        return tuple(
            prediction.card
            for prediction in self.full_predictions
            if prediction.visible and prediction.card is not None
        )

    @property
    def part_cards(self) -> tuple[str, ...]:
        return tuple(
            prediction.card
            for prediction in self.part_predictions
            if prediction.visible and prediction.card is not None
        )

    @property
    def full_signature(self) -> tuple[str, ...]:
        items: list[str] = []
        for prediction in self.full_predictions:
            if prediction.visible and prediction.card is not None:
                items.append(f"{prediction.group}.{prediction.slot}={prediction.card}")
        return tuple(sorted(items))

    def target_confidence(self, card: str) -> float:
        return max(
            (
                prediction.confidence
                for prediction in self.part_predictions
                if prediction.visible and prediction.card == card
            ),
            default=0.0,
        )


@dataclass(slots=True)
class _SelectedFrame:
    row: _AnalyzedFrame
    category: str
    notes: list[str]


def select_poker_legends_card_review_candidates(
    sources: Sequence[PokerLegendsCardReviewSource],
    *,
    output_dir: str | Path,
    card_manifest: str | Path,
    card_part_manifest: str | Path,
    exclude_truths: Sequence[PokerLegendsExcludedTruth] = (),
    target_cards: Sequence[str] = (),
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_target_per_card: int = DEFAULT_MAX_TARGET_PER_CARD,
    spaced_sample_seconds: float = DEFAULT_SPACED_SAMPLE_SECONDS,
    screen_kinds: Sequence[str] = DEFAULT_SCREEN_KINDS,
) -> dict[str, object]:
    if not sources:
        raise ValueError("at least one source is required")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if max_target_per_card <= 0:
        raise ValueError("max_target_per_card must be positive")
    if spaced_sample_seconds <= 0:
        raise ValueError("spaced_sample_seconds must be positive")

    selected_screen_kinds = frozenset(str(kind) for kind in screen_kinds)
    normalized_targets: list[str] = []
    for raw_card in target_cards:
        card = _normalize_card_code(raw_card)
        if card is None:
            raise ValueError("target_cards must be 2-character card codes")
        normalized_targets.append(card)

    full_recognizer = PokerLegendsCardTemplateRecognizer.from_manifest(card_manifest)
    part_recognizer = PokerLegendsCardPartTemplateRecognizer.from_manifest(card_part_manifest)
    excluded = _load_excluded_frame_ids(exclude_truths)

    rows: list[_AnalyzedFrame] = []
    scanned_frames = 0
    excluded_frames = 0
    for source in sources:
        source_manifest = Path(source.manifest)
        source_dir = source_manifest.parent
        ingest = VideoIngestManifest.read_json(source_manifest)
        excluded_ids = excluded.get(source.name, frozenset())
        for frame in ingest.frames:
            scanned_frames += 1
            original_frame_id = Path(frame.image).stem
            if original_frame_id in excluded_ids:
                excluded_frames += 1
                continue
            image_path = source_dir / frame.image
            annotation = _layout_annotation(source_dir / frame.annotation)
            detection = detect_poker_legends_screen_state(
                image_path,
                layout_annotation=annotation,
            )
            screen_kind = detection.screen.kind.value
            if screen_kind not in selected_screen_kinds:
                continue
            frame_id = f"{_safe_source_name(source.name)}__{original_frame_id}"
            full_predictions = full_recognizer.recognize(
                image_path,
                annotation,
                frame_id=frame_id,
            )
            part_predictions = part_recognizer.recognize(
                image_path,
                annotation,
                frame_id=frame_id,
            )
            target_hits = _target_hits(part_predictions, tuple(normalized_targets))
            gap_slots = _gap_slots(full_predictions, part_predictions)
            rows.append(
                _AnalyzedFrame(
                    source=source,
                    source_dir=source_dir,
                    source_frame=frame,
                    frame_id=frame_id,
                    original_frame_id=original_frame_id,
                    image_path=image_path,
                    annotation=annotation,
                    screen_kind=screen_kind,
                    full_predictions=full_predictions,
                    part_predictions=part_predictions,
                    target_hits=target_hits,
                    gap_slots=gap_slots,
                )
            )

    selected = _select_candidate_rows(
        rows,
        target_cards=tuple(normalized_targets),
        max_candidates=max_candidates,
        max_target_per_card=max_target_per_card,
        spaced_sample_seconds=spaced_sample_seconds,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates = _materialize_selection(output, selected)
    selection = PokerLegendsCardReviewSelection(
        schema_version=1,
        sources=tuple(sources),
        target_cards=tuple(normalized_targets),
        selected=tuple(candidates),
        scanned_frames=scanned_frames,
        eligible_frames=len(rows),
        excluded_frames=excluded_frames,
    )
    manifest_path = output / "card_review_candidate_manifest.json"
    report_path = output / "card_review_candidate_report.md"
    manifest_path.write_text(
        json.dumps(selection.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(report_path, selection)
    contact_sheet = _write_contact_sheet(output, candidates)
    summary = {
        "schema_version": 1,
        "sources": len(sources),
        "scanned_frames": scanned_frames,
        "eligible_frames": len(rows),
        "excluded_frames": excluded_frames,
        "selected_frames": len(candidates),
        "target_cards": list(normalized_targets),
        "manifest": str(manifest_path),
        "report": str(report_path),
        "contact_sheet": None if contact_sheet is None else str(contact_sheet),
        "annotations_dir": str(output / "annotations"),
        "image_root": str(output),
    }
    (output / "card_review_candidate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select Poker Legends card-focused LLM review candidates."
    )
    parser.add_argument(
        "--source",
        nargs=2,
        action="append",
        metavar=("NAME", "INGEST_MANIFEST"),
        required=True,
        help="Repeatable video source.",
    )
    parser.add_argument("--card-manifest", required=True, help="Full-card template manifest.")
    parser.add_argument(
        "--card-part-manifest",
        required=True,
        help="Rank/suit part template manifest.",
    )
    parser.add_argument(
        "--exclude-truth",
        nargs=2,
        action="append",
        metavar=("SOURCE", "TRUTH_PATH_OR_DIR"),
        default=[],
        help="Repeatable source truth set to exclude from new review candidates.",
    )
    parser.add_argument(
        "--target-cards",
        default="",
        help="Comma-separated card codes to prioritize, e.g. 5S,7D,8S,QD.",
    )
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--max-target-per-card", type=int, default=DEFAULT_MAX_TARGET_PER_CARD)
    parser.add_argument(
        "--spaced-sample-seconds",
        type=float,
        default=DEFAULT_SPACED_SAMPLE_SECONDS,
    )
    parser.add_argument(
        "--screen-kinds",
        default=",".join(DEFAULT_SCREEN_KINDS),
        help="Comma-separated screen kinds to consider.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = select_poker_legends_card_review_candidates(
        [PokerLegendsCardReviewSource.from_args(item) for item in args.source],
        output_dir=args.out,
        card_manifest=args.card_manifest,
        card_part_manifest=args.card_part_manifest,
        exclude_truths=[PokerLegendsExcludedTruth.from_args(item) for item in args.exclude_truth],
        target_cards=_parse_csv(args.target_cards),
        max_candidates=args.max_candidates,
        max_target_per_card=args.max_target_per_card,
        spaced_sample_seconds=args.spaced_sample_seconds,
        screen_kinds=_parse_csv(args.screen_kinds),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _select_candidate_rows(
    rows: Sequence[_AnalyzedFrame],
    *,
    target_cards: Sequence[str],
    max_candidates: int,
    max_target_per_card: int,
    spaced_sample_seconds: float,
) -> list[_SelectedFrame]:
    selected: dict[str, _SelectedFrame] = {}

    def add(row: _AnalyzedFrame, category: str, note: str, *, near_seconds: float = 0.0) -> None:
        if row.frame_id in selected:
            item = selected[row.frame_id]
            if note not in item.notes:
                item.notes.append(note)
            return
        if len(selected) >= max_candidates:
            return
        if near_seconds > 0 and _has_near(list(selected.values()), row, seconds=near_seconds):
            return
        selected[row.frame_id] = _SelectedFrame(row=row, category=category, notes=[note])

    for card in target_cards:
        hits = sorted(
            (row for row in rows if card in row.target_hits),
            key=lambda row: (-row.target_confidence(card), row.timestamp_seconds),
        )
        added = 0
        for row in hits:
            if added >= max_target_per_card:
                break
            before = len(selected)
            add(
                row,
                "target_card_probe",
                f"rank/suit templates suspect missing target {card}",
                near_seconds=10.0,
            )
            if len(selected) > before:
                added += 1

    gap_rows = sorted(
        (row for row in rows if row.gap_slots),
        key=lambda row: (
            -len(row.gap_slots),
            -len(row.part_cards),
            row.timestamp_seconds,
        ),
    )
    for row in gap_rows:
        gap_note = (
            "part templates see cards where full-card template is missing: "
            f"{', '.join(row.gap_slots)}"
        )
        add(
            row,
            "card_template_gap",
            gap_note,
            near_seconds=20.0,
        )

    previous_signature_by_source: dict[str, tuple[str, ...]] = {}
    previous_selected_time_by_source: dict[str, float] = {}
    for row in sorted(rows, key=lambda item: (item.source.name, item.timestamp_seconds)):
        signature = row.full_signature
        if not signature:
            continue
        previous_signature = previous_signature_by_source.get(row.source.name)
        previous_signature_by_source[row.source.name] = signature
        if signature == previous_signature:
            continue
        previous_time = previous_selected_time_by_source.get(row.source.name)
        if previous_time is not None and row.timestamp_seconds - previous_time < 20.0:
            continue
        add(row, "new_card_signature", "full-card recognizer signature changed")
        previous_selected_time_by_source[row.source.name] = row.timestamp_seconds

    last_sample_by_source_kind: dict[tuple[str, str], float] = {}
    for row in sorted(rows, key=lambda item: (item.source.name, item.timestamp_seconds)):
        key = (row.source.name, row.screen_kind)
        last_time = last_sample_by_source_kind.get(key)
        if last_time is not None and row.timestamp_seconds - last_time < spaced_sample_seconds:
            continue
        add(row, "card_spaced_sample", f"{row.screen_kind} spaced card sample")
        last_sample_by_source_kind[key] = row.timestamp_seconds

    return sorted(
        selected.values(),
        key=lambda item: (item.row.source.name, item.row.timestamp_seconds, item.row.frame_id),
    )


def _materialize_selection(
    output: Path,
    selected: Sequence[_SelectedFrame],
) -> list[PokerLegendsCardReviewCandidate]:
    frames_dir = output / "frames"
    annotations_dir = output / "annotations"
    overlays_dir = output / "layout_overlays"
    frames_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[PokerLegendsCardReviewCandidate] = []
    for selected_frame in selected:
        row = selected_frame.row
        image_path = frames_dir / f"{row.frame_id}.png"
        annotation_path = annotations_dir / f"{row.frame_id}.json"
        shutil.copy2(row.image_path, image_path)
        annotation = dict(row.annotation)
        annotation["image"] = str(image_path.relative_to(output))
        annotation["source"] = "poker_legends_card_review_selection"
        annotation["source_frame"] = {
            "source": row.source.name,
            "original_frame_id": row.original_frame_id,
            "image": str(row.source_frame.image),
            "annotation": str(row.source_frame.annotation),
        }
        annotation_path.write_text(
            json.dumps(annotation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        draw_layout_overlay(
            output / str(annotation["image"]),
            annotation_path,
            overlays_dir / f"{row.frame_id}_layout.png",
        )
        note = "; ".join(selected_frame.notes)
        candidates.append(
            PokerLegendsCardReviewCandidate(
                frame_id=row.frame_id,
                source=row.source.name,
                original_frame_id=row.original_frame_id,
                image=str(image_path.relative_to(output)),
                annotation=str(annotation_path.relative_to(output)),
                source_image=str(row.source_frame.image),
                source_annotation=str(row.source_frame.annotation),
                frame_index=row.source_frame.frame_index,
                timestamp_seconds=row.timestamp_seconds,
                screen_kind=row.screen_kind,
                category=selected_frame.category,
                note=note,
                full_cards=row.full_cards,
                part_cards=row.part_cards,
                target_hits=row.target_hits,
                gap_slots=row.gap_slots,
            )
        )
    return candidates


def _layout_annotation(annotation_path: Path) -> dict[str, object]:
    annotation = _read_json_object(annotation_path)
    width = _to_int(annotation["width"])
    height = _to_int(annotation["height"])
    regions = annotation.get("regions")
    has_regions = isinstance(regions, Mapping) and any(
        _mapping_sequence(value) for value in regions.values()
    )
    if not has_regions:
        annotation = dict(annotation)
        annotation["regions"] = poker_legends_layout_regions(width, height)
    return annotation


def _load_excluded_frame_ids(
    exclude_truths: Sequence[PokerLegendsExcludedTruth],
) -> dict[str, frozenset[str]]:
    result: dict[str, set[str]] = {}
    for item in exclude_truths:
        source_ids = result.setdefault(item.source, set())
        for path in _resolve_json_paths(item.path):
            truth = _read_json_object(path)
            source = truth.get("source")
            if isinstance(source, Mapping) and source.get("name") == item.source:
                original = source.get("original_frame_id")
                if isinstance(original, str):
                    source_ids.add(original)
                    continue
            frame_id = str(truth.get("frame_id") or path.stem)
            prefix = f"{_safe_source_name(item.source)}__"
            source_ids.add(frame_id[len(prefix) :] if frame_id.startswith(prefix) else frame_id)
    return {source: frozenset(ids) for source, ids in result.items()}


def _target_hits(
    predictions: Sequence[PokerLegendsCardPartPrediction],
    target_cards: Sequence[str],
) -> tuple[str, ...]:
    targets = set(target_cards)
    hits = {
        prediction.card
        for prediction in predictions
        if prediction.visible and prediction.card is not None and prediction.card in targets
    }
    return tuple(sorted(hits))


def _gap_slots(
    full_predictions: Sequence[PokerLegendsCardPrediction],
    part_predictions: Sequence[PokerLegendsCardPartPrediction],
) -> tuple[str, ...]:
    full_by_slot = {
        (prediction.group, prediction.slot): prediction for prediction in full_predictions
    }
    slots: list[str] = []
    for prediction in part_predictions:
        if not prediction.visible or prediction.card is None:
            continue
        full = full_by_slot.get((prediction.group, prediction.slot))
        if full is None or not full.visible or full.card is None:
            slots.append(f"{prediction.group}.{prediction.slot}")
    return tuple(sorted(slots))


def _has_near(
    selected: Sequence[_SelectedFrame],
    row: _AnalyzedFrame,
    *,
    seconds: float,
) -> bool:
    return any(
        item.row.source.name == row.source.name
        and abs(item.row.timestamp_seconds - row.timestamp_seconds) < seconds
        for item in selected
    )


def _write_report(path: Path, selection: PokerLegendsCardReviewSelection) -> None:
    lines = [
        "# Poker Legends Card Review Candidates",
        "",
        "## Summary",
        f"- Sources: {len(selection.sources)}",
        f"- Scanned frames: {selection.scanned_frames}",
        f"- Excluded existing truth frames: {selection.excluded_frames}",
        f"- Eligible table frames: {selection.eligible_frames}",
        f"- Selected frames: {len(selection.selected)}",
        f"- Target cards: {', '.join(selection.target_cards) or '-'}",
        "",
        "## Selected Frames",
        "| Frame | Source | Time | Kind | Category | Full cards | Part cards | Hits | Note |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for frame in selection.selected:
        lines.append(
            f"| `{frame.frame_id}` | `{frame.source}` | {frame.timestamp_seconds:.1f} | "
            f"`{frame.screen_kind}` | `{frame.category}` | "
            f"{', '.join(frame.full_cards) or '-'} | {', '.join(frame.part_cards) or '-'} | "
            f"{', '.join(frame.target_hits) or '-'} | {frame.note.replace('|', '/')} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- Full-card predictions are conservative exact-card template matches.",
            "- Part-card predictions are rank/suit template probes; they are used only to "
            "select LLM review candidates, not as trusted truth.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_contact_sheet(
    output: Path,
    frames: Sequence[PokerLegendsCardReviewCandidate],
    *,
    columns: int = 4,
) -> Path | None:
    if not frames:
        return None
    thumb_width, thumb_height = 360, 221
    thumbnails: list[RgbImage] = []
    for frame in frames:
        image = cv2.imread(str(output / frame.image), cv2.IMREAD_COLOR)
        if image is None:
            continue
        thumb = cast(
            RgbImage,
            cv2.resize(image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA),
        )
        cv2.rectangle(thumb, (0, 0), (thumb_width, 64), (0, 0, 0), -1)
        cv2.putText(
            thumb,
            f"{frame.frame_id} {frame.timestamp_seconds:.1f}s",
            (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            thumb,
            frame.category[:44],
            (8, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (170, 255, 170),
            2,
            cv2.LINE_AA,
        )
        thumbnails.append(thumb)
    if not thumbnails:
        return None
    rows = (len(thumbnails) + columns - 1) // columns
    sheet = np.zeros((rows * thumb_height, columns * thumb_width, 3), dtype=np.uint8)
    for index, thumb in enumerate(thumbnails):
        row = index // columns
        column = index % columns
        y = row * thumb_height
        x = column * thumb_width
        sheet[y : y + thumb_height, x : x + thumb_width] = thumb
    path = output / "contact_sheet.jpg"
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"could not write contact sheet: {path}")
    return path


def _resolve_json_paths(path_or_dir: str | Path) -> tuple[Path, ...]:
    path = Path(path_or_dir)
    if path.is_dir():
        paths = tuple(sorted(path.glob("*.json")))
    elif path.is_file():
        paths = (path,)
    else:
        raise FileNotFoundError(f"truth path does not exist: {path}")
    if not paths:
        raise FileNotFoundError(f"no truth JSON files found: {path}")
    return paths


def _safe_source_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    if not safe:
        raise ValueError("source name cannot be empty")
    return safe


def _normalize_card_code(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    card = value.strip().upper()
    if len(card) != 2:
        return None
    if card[0] not in set("AKQJT98765432") or card[1] not in set("SHDC"):
        return None
    return card


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


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


def _read_json_object(path: str | Path) -> dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, object], data)
