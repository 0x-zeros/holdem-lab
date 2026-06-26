"""Dense ScreenState scans and LLM candidate selection for Poker Legends ingests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

from holdem_bot.vision.poker_legends_screen import (
    PokerLegendsRegionFeature,
    PokerLegendsScreenDetection,
    detect_poker_legends_screen_state,
)
from holdem_bot.vision.selection import KeyframeSelectionRequest, select_keyframes
from holdem_bot.vision.video import ExtractedFrame, VideoIngestManifest

DEFAULT_MAX_LLM_CANDIDATES = 60
DEFAULT_SPACED_SAMPLE_SECONDS = 90.0
CONTACT_SHEET_PAGE_SIZE = 24


@dataclass(frozen=True, slots=True)
class PokerLegendsScreenScanRow:
    frame_id: str
    image: str
    annotation: str
    frame_index: int
    timestamp_seconds: float
    screen_kind: str
    confidence: float
    blocking_reason: str | None
    hero_turn: bool | None
    active_primary_buttons: int
    overlay_signals: tuple[str, ...]
    button_features: tuple[PokerLegendsRegionFeature, ...]
    overlay_features: tuple[PokerLegendsRegionFeature, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "image": self.image,
            "annotation": self.annotation,
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "screen_kind": self.screen_kind,
            "confidence": self.confidence,
            "blocking_reason": self.blocking_reason,
            "hero_turn": self.hero_turn,
            "active_primary_buttons": self.active_primary_buttons,
            "overlay_signals": list(self.overlay_signals),
            "button_features": [feature.to_dict() for feature in self.button_features],
            "overlay_features": [feature.to_dict() for feature in self.overlay_features],
        }


@dataclass(frozen=True, slots=True)
class PokerLegendsScreenRun:
    screen_kind: str
    start_frame_id: str
    end_frame_id: str
    start_timestamp_seconds: float
    end_timestamp_seconds: float
    frame_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PokerLegendsLlmCandidate:
    frame_id: str
    timestamp_seconds: float
    category: str
    note: str
    screen_kind: str
    active_primary_buttons: int
    overlay_signals: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "timestamp_seconds": self.timestamp_seconds,
            "category": self.category,
            "note": self.note,
            "screen_kind": self.screen_kind,
            "active_primary_buttons": self.active_primary_buttons,
            "overlay_signals": list(self.overlay_signals),
            "select_spec": self.select_spec,
        }

    @property
    def select_spec(self) -> str:
        safe_note = self.note.replace("|", "/")
        return f"{self.frame_id}|{self.category}|{safe_note}"

    def to_selection_request(self) -> KeyframeSelectionRequest:
        return KeyframeSelectionRequest(
            frame_id=self.frame_id,
            category=self.category,
            note=self.note,
        )


@dataclass(frozen=True, slots=True)
class PokerLegendsScreenScan:
    schema_version: int
    source_manifest: str
    rows: tuple[PokerLegendsScreenScanRow, ...]
    runs: tuple[PokerLegendsScreenRun, ...]
    llm_candidates: tuple[PokerLegendsLlmCandidate, ...]
    screen_kind_counts: dict[str, int]
    active_primary_button_counts: dict[str, int]
    overlay_signal_counts: dict[str, int]
    contact_sheet_counts: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_manifest": self.source_manifest,
            "frames": len(self.rows),
            "runs": [run.to_dict() for run in self.runs],
            "llm_candidates": [candidate.to_dict() for candidate in self.llm_candidates],
            "screen_kind_counts": self.screen_kind_counts,
            "active_primary_button_counts": self.active_primary_button_counts,
            "overlay_signal_counts": self.overlay_signal_counts,
            "contact_sheet_counts": self.contact_sheet_counts,
            "rows": [row.to_dict() for row in self.rows],
        }


def build_poker_legends_screen_scan(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    max_llm_candidates: int = DEFAULT_MAX_LLM_CANDIDATES,
    spaced_sample_seconds: float = DEFAULT_SPACED_SAMPLE_SECONDS,
    selection_output_dir: str | Path | None = None,
) -> dict[str, object]:
    if max_llm_candidates <= 0:
        raise ValueError("max_llm_candidates must be positive")
    if spaced_sample_seconds <= 0:
        raise ValueError("spaced_sample_seconds must be positive")

    manifest_file = Path(manifest_path)
    scan = scan_poker_legends_ingest(
        manifest_file,
        max_llm_candidates=max_llm_candidates,
        spaced_sample_seconds=spaced_sample_seconds,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scan_path = output / "screen_state_scan.json"
    report_path = output / "screen_state_scan.md"
    scan_path.write_text(
        json.dumps(scan.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_scan_report(report_path, scan)

    selection_manifest = None
    if selection_output_dir is not None and scan.llm_candidates:
        select_keyframes(
            manifest_file,
            selection_output_dir,
            [candidate.to_selection_request() for candidate in scan.llm_candidates],
        )
        selection_manifest = str(Path(selection_output_dir) / "selected_manifest.json")

    summary = {
        "schema_version": 1,
        "frames": len(scan.rows),
        "runs": len(scan.runs),
        "llm_candidates": len(scan.llm_candidates),
        "screen_kind_counts": scan.screen_kind_counts,
        "scan": str(scan_path),
        "report": str(report_path),
        "selection_manifest": selection_manifest,
    }
    (output / "screen_state_scan_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def scan_poker_legends_ingest(
    manifest_path: str | Path,
    *,
    max_llm_candidates: int = DEFAULT_MAX_LLM_CANDIDATES,
    spaced_sample_seconds: float = DEFAULT_SPACED_SAMPLE_SECONDS,
) -> PokerLegendsScreenScan:
    manifest_file = Path(manifest_path)
    ingest = VideoIngestManifest.read_json(manifest_file)
    source_dir = manifest_file.parent
    rows: list[PokerLegendsScreenScanRow] = []
    for frame in ingest.frames:
        annotation_path = source_dir / frame.annotation
        annotation = _read_json_object(annotation_path)
        detection = detect_poker_legends_screen_state(
            source_dir / frame.image,
            layout_annotation=annotation,
        )
        rows.append(_row_from_detection(frame, detection))

    runs = _build_runs(rows)
    candidates = _select_llm_candidates(
        rows,
        max_candidates=max_llm_candidates,
        spaced_sample_seconds=spaced_sample_seconds,
    )
    return PokerLegendsScreenScan(
        schema_version=1,
        source_manifest=str(manifest_file),
        rows=tuple(rows),
        runs=tuple(runs),
        llm_candidates=tuple(candidates),
        screen_kind_counts=_counter_dict(row.screen_kind for row in rows),
        active_primary_button_counts=_counter_dict(str(row.active_primary_buttons) for row in rows),
        overlay_signal_counts=_counter_dict(
            signal for row in rows for signal in row.overlay_signals
        ),
        contact_sheet_counts=_contact_sheet_counts(rows),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan Poker Legends ingest keyframes and select LLM review candidates."
    )
    parser.add_argument("manifest", help="Input video ingest manifest.json path.")
    parser.add_argument("--out", required=True, help="Output directory for scan artifacts.")
    parser.add_argument(
        "--max-llm-candidates",
        type=int,
        default=DEFAULT_MAX_LLM_CANDIDATES,
        help="Maximum recommended LLM/review frames.",
    )
    parser.add_argument(
        "--spaced-sample-seconds",
        type=float,
        default=DEFAULT_SPACED_SAMPLE_SECONDS,
        help="Minimum spacing for routine per-kind samples.",
    )
    parser.add_argument(
        "--selection-out",
        help="Optional output directory for copied candidate frames via select_keyframes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = build_poker_legends_screen_scan(
        args.manifest,
        output_dir=args.out,
        max_llm_candidates=args.max_llm_candidates,
        spaced_sample_seconds=args.spaced_sample_seconds,
        selection_output_dir=args.selection_out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _row_from_detection(
    frame: ExtractedFrame,
    detection: PokerLegendsScreenDetection,
) -> PokerLegendsScreenScanRow:
    return PokerLegendsScreenScanRow(
        frame_id=Path(frame.image).stem,
        image=frame.image,
        annotation=frame.annotation,
        frame_index=frame.frame_index,
        timestamp_seconds=frame.timestamp_seconds,
        screen_kind=detection.screen.kind.value,
        confidence=detection.screen.confidence,
        blocking_reason=detection.screen.blocking_reason,
        hero_turn=detection.screen.hero_turn,
        active_primary_buttons=detection.active_primary_buttons,
        overlay_signals=tuple(detection.overlay_signals),
        button_features=tuple(detection.button_features),
        overlay_features=tuple(detection.overlay_features),
    )


def _build_runs(rows: Sequence[PokerLegendsScreenScanRow]) -> list[PokerLegendsScreenRun]:
    if not rows:
        return []
    runs: list[PokerLegendsScreenRun] = []
    start = rows[0]
    previous = rows[0]
    count = 1
    for row in rows[1:]:
        if row.screen_kind == previous.screen_kind:
            previous = row
            count += 1
            continue
        runs.append(_run_from_rows(start, previous, count))
        start = row
        previous = row
        count = 1
    runs.append(_run_from_rows(start, previous, count))
    return runs


def _run_from_rows(
    start: PokerLegendsScreenScanRow,
    end: PokerLegendsScreenScanRow,
    count: int,
) -> PokerLegendsScreenRun:
    return PokerLegendsScreenRun(
        screen_kind=start.screen_kind,
        start_frame_id=start.frame_id,
        end_frame_id=end.frame_id,
        start_timestamp_seconds=start.timestamp_seconds,
        end_timestamp_seconds=end.timestamp_seconds,
        frame_count=count,
    )


def _select_llm_candidates(
    rows: Sequence[PokerLegendsScreenScanRow],
    *,
    max_candidates: int,
    spaced_sample_seconds: float,
) -> list[PokerLegendsLlmCandidate]:
    candidates = _CandidateAccumulator(rows, max_candidates=max_candidates)
    if not rows:
        return []

    candidates.add(rows[0], "session_start", "first keyframe")

    for row in rows:
        if row.screen_kind == "blocked_overlay":
            note = _signal_note(row) or "blocked overlay detected"
            candidates.add(row, "blocked_overlay", note)
        if row.screen_kind == "actionable_table" and row.active_primary_buttons == 2:
            candidates.add(row, "actionable_edge_2_buttons", "two primary action buttons")

    last_sample_by_kind: dict[str, float] = {}
    for row in rows:
        previous_time = last_sample_by_kind.get(row.screen_kind)
        if previous_time is None or row.timestamp_seconds - previous_time >= spaced_sample_seconds:
            category = f"{row.screen_kind}_sample"
            candidates.add(row, category, f"{row.screen_kind} spaced sample")
            last_sample_by_kind[row.screen_kind] = row.timestamp_seconds

    for row in rows:
        if (
            row.screen_kind == "actionable_table"
            and row.active_primary_buttons >= 3
            and not candidates.has_near(row, seconds=spaced_sample_seconds / 2)
        ):
            candidates.add(row, "actionable_3_buttons", "three primary action buttons")

    transition_spacing = max(15.0, spaced_sample_seconds / 2)
    for index, row in enumerate(rows[1:], start=1):
        previous = rows[index - 1]
        if row.screen_kind == previous.screen_kind:
            continue
        if not candidates.has_near(previous, seconds=transition_spacing):
            candidates.add(previous, "screen_transition_before", f"before {row.screen_kind}")
        if not candidates.has_near(row, seconds=transition_spacing):
            candidates.add(row, "screen_transition_after", f"after {previous.screen_kind}")

    return candidates.items()


@dataclass(slots=True)
class _CandidateAccumulator:
    rows: Sequence[PokerLegendsScreenScanRow]
    max_candidates: int
    _by_id: dict[str, PokerLegendsScreenScanRow] = field(init=False)
    _categories: dict[str, str] = field(default_factory=dict)
    _notes: dict[str, list[str]] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._by_id = {row.frame_id: row for row in self.rows}

    def add(self, row: PokerLegendsScreenScanRow, category: str, note: str) -> None:
        if row.frame_id not in self._categories:
            if len(self._order) >= self.max_candidates:
                return
            self._categories[row.frame_id] = category
            self._notes[row.frame_id] = []
            self._order.append(row.frame_id)
        notes = self._notes[row.frame_id]
        if note and note not in notes:
            notes.append(note)

    def has_near(self, row: PokerLegendsScreenScanRow, *, seconds: float) -> bool:
        return any(
            abs(self._by_id[frame_id].timestamp_seconds - row.timestamp_seconds) < seconds
            for frame_id in self._order
        )

    def items(self) -> list[PokerLegendsLlmCandidate]:
        candidates: list[PokerLegendsLlmCandidate] = []
        for frame_id in sorted(
            self._order,
            key=lambda value: (self._by_id[value].timestamp_seconds, value),
        ):
            row = self._by_id[frame_id]
            candidates.append(
                PokerLegendsLlmCandidate(
                    frame_id=row.frame_id,
                    timestamp_seconds=row.timestamp_seconds,
                    category=self._categories[frame_id],
                    note="; ".join(self._notes[frame_id]),
                    screen_kind=row.screen_kind,
                    active_primary_buttons=row.active_primary_buttons,
                    overlay_signals=row.overlay_signals,
                )
            )
        return candidates


def _contact_sheet_counts(
    rows: Sequence[PokerLegendsScreenScanRow],
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for index, row in enumerate(rows):
        sheet = f"contact_sheet_{index // CONTACT_SHEET_PAGE_SIZE:03d}.jpg"
        counts.setdefault(sheet, Counter())[row.screen_kind] += 1
    return {sheet: dict(counter) for sheet, counter in sorted(counts.items())}


def _counter_dict(values: Sequence[str] | object) -> dict[str, int]:
    return dict(Counter(cast(Iterable[str], values)))


def _signal_note(row: PokerLegendsScreenScanRow) -> str:
    return ", ".join(row.overlay_signals)


def _write_scan_report(path: Path, scan: PokerLegendsScreenScan) -> None:
    lines = [
        "# Poker Legends ScreenState Scan",
        "",
        "## Source",
        f"- Manifest: `{scan.source_manifest}`",
        f"- Frames scanned: {len(scan.rows)}",
        f"- Runs: {len(scan.runs)}",
        f"- LLM candidates: {len(scan.llm_candidates)}",
        "",
        "## Screen Kind Counts",
    ]
    for kind, count in sorted(scan.screen_kind_counts.items()):
        lines.append(f"- `{kind}`: {count}")
    lines.extend(["", "## Active Primary Button Counts"])
    for buttons, count in sorted(
        scan.active_primary_button_counts.items(), key=lambda item: int(item[0])
    ):
        lines.append(f"- `{buttons}` active buttons: {count}")
    lines.extend(["", "## Overlay Signal Counts"])
    if scan.overlay_signal_counts:
        for signal, count in sorted(scan.overlay_signal_counts.items()):
            lines.append(f"- `{signal}`: {count}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Runs",
            "| Kind | Start | End | Start time | End time | Frames |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for run in scan.runs[:80]:
        lines.append(
            f"| `{run.screen_kind}` | `{run.start_frame_id}` | `{run.end_frame_id}` | "
            f"{run.start_timestamp_seconds:.1f} | {run.end_timestamp_seconds:.1f} | "
            f"{run.frame_count} |"
        )
    if len(scan.runs) > 80:
        lines.append(f"| ... | ... | ... | ... | ... | {len(scan.runs) - 80} more runs |")

    lines.extend(
        [
            "",
            "## Recommended LLM / Review Candidates",
            "| Frame | Time | Kind | Buttons | Category | Signals | Note |",
            "| --- | ---: | --- | ---: | --- | --- | --- |",
        ]
    )
    for candidate in scan.llm_candidates:
        signals = ", ".join(candidate.overlay_signals) or "-"
        note = candidate.note.replace("|", "/")
        lines.append(
            f"| `{candidate.frame_id}` | {candidate.timestamp_seconds:.1f} | "
            f"`{candidate.screen_kind}` | {candidate.active_primary_buttons} | "
            f"`{candidate.category}` | {signals} | {note} |"
        )

    lines.extend(
        [
            "",
            "## Selection Specs",
            "Use these with `holdem-bot-select-keyframes` if `--selection-out` was not used:",
            "",
            "```text",
        ]
    )
    for candidate in scan.llm_candidates:
        lines.append(f"--select {candidate.select_spec}")
    lines.extend(
        [
            "```",
            "",
            "## Notes",
            "- This is detector output, not truth. Use it to keep LLM packages small and focused.",
            "- Candidate frames intentionally include state transitions and blocked overlays, "
            "not only clean table states.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json_object(path: str | Path) -> dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, object], data)
