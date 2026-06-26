"""Representative keyframe selection for Poker Legends video ingests."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from holdem_bot.vision.video import VideoIngestManifest


@dataclass(frozen=True, slots=True)
class KeyframeSelectionRequest:
    frame_id: str
    category: str = "sample"
    note: str = ""

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("frame_id cannot be empty")
        if not self.category:
            raise ValueError("category cannot be empty")

    @classmethod
    def parse(cls, spec: str) -> Self:
        parts = spec.split("|", 2) if "|" in spec else spec.split("::", 2)
        frame_id = Path(parts[0].strip()).stem
        category = parts[1].strip() if len(parts) >= 2 and parts[1].strip() else "sample"
        note = parts[2].strip() if len(parts) >= 3 else ""
        return cls(frame_id=frame_id, category=category, note=note)


@dataclass(frozen=True, slots=True)
class SelectedKeyframe:
    frame_id: str
    image: str
    annotation: str
    source_image: str
    source_annotation: str
    frame_index: int
    timestamp_seconds: float
    reason: str
    category: str
    note: str


@dataclass(frozen=True, slots=True)
class KeyframeSelectionManifest:
    schema_version: int
    source_manifest: str
    contact_sheet: str | None
    selection_report: str
    frames: tuple[SelectedKeyframe, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported keyframe selection manifest schema version")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def read_json(cls, path: str | Path) -> Self:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            schema_version=int(data["schema_version"]),
            source_manifest=str(data["source_manifest"]),
            contact_sheet=None if data["contact_sheet"] is None else str(data["contact_sheet"]),
            selection_report=str(data["selection_report"]),
            frames=tuple(_selected_keyframe_from_dict(frame) for frame in data["frames"]),
        )


def select_keyframes(
    manifest_path: str | Path,
    output_dir: str | Path,
    selections: Sequence[KeyframeSelectionRequest],
) -> KeyframeSelectionManifest:
    if not selections:
        raise ValueError("at least one keyframe selection is required")

    source_manifest_path = Path(manifest_path)
    source_dir = source_manifest_path.parent
    output = Path(output_dir)
    frames_dir = output / "frames"
    annotations_dir = output / "annotations"
    frames_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    ingest = VideoIngestManifest.read_json(source_manifest_path)
    by_frame_id = {Path(frame.image).stem: frame for frame in ingest.frames}
    selected: list[SelectedKeyframe] = []
    seen: set[str] = set()

    for request in selections:
        if request.frame_id in seen:
            raise ValueError(f"duplicate selected keyframe: {request.frame_id}")
        seen.add(request.frame_id)

        frame = by_frame_id.get(request.frame_id)
        if frame is None:
            raise ValueError(f"unknown keyframe: {request.frame_id}")

        source_image = source_dir / frame.image
        source_annotation = source_dir / frame.annotation
        image_path = frames_dir / source_image.name
        annotation_path = annotations_dir / source_annotation.name
        shutil.copy2(source_image, image_path)
        shutil.copy2(source_annotation, annotation_path)

        selected.append(
            SelectedKeyframe(
                frame_id=request.frame_id,
                image=str(image_path.relative_to(output)),
                annotation=str(annotation_path.relative_to(output)),
                source_image=frame.image,
                source_annotation=frame.annotation,
                frame_index=frame.frame_index,
                timestamp_seconds=frame.timestamp_seconds,
                reason=frame.reason,
                category=request.category,
                note=request.note,
            )
        )

    contact_sheet = _write_selection_contact_sheet(output, selected)
    report_path = output / "selection_report.md"
    selection_manifest = KeyframeSelectionManifest(
        schema_version=1,
        source_manifest=str(source_manifest_path),
        contact_sheet=None if contact_sheet is None else str(contact_sheet.relative_to(output)),
        selection_report=str(report_path.relative_to(output)),
        frames=tuple(selected),
    )
    selection_manifest.write_json(output / "selected_manifest.json")
    _write_selection_report(report_path, selection_manifest)
    return selection_manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy representative keyframes from a video ingest manifest."
    )
    parser.add_argument("manifest", help="Input video ingest manifest.json path.")
    parser.add_argument("--out", required=True, help="Output directory for selected frames.")
    parser.add_argument(
        "--select",
        action="append",
        required=True,
        help=(
            "Selection spec: FRAME_ID[::CATEGORY[::NOTE]] or FRAME_ID[|CATEGORY[|NOTE]]. "
            "Repeat for multiple frames."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    selections = tuple(KeyframeSelectionRequest.parse(spec) for spec in args.select)
    manifest = select_keyframes(args.manifest, args.out, selections)
    print(
        json.dumps(
            {
                "frames": len(manifest.frames),
                "manifest": str(Path(args.out) / "selected_manifest.json"),
                "selection_report": str(Path(args.out) / manifest.selection_report),
                "source_manifest": manifest.source_manifest,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _write_selection_contact_sheet(
    output: Path,
    frames: Sequence[SelectedKeyframe],
    *,
    columns: int = 4,
) -> Path | None:
    if not frames:
        return None

    thumbnails: list[NDArray[np.uint8]] = []
    thumb_width, thumb_height = 360, 221
    for frame in frames:
        image = cv2.imread(str(output / frame.image), cv2.IMREAD_COLOR)
        if image is None:
            continue
        thumb = cast(
            NDArray[np.uint8],
            cv2.resize(image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA),
        )
        cv2.rectangle(thumb, (0, 0), (thumb_width, 54), (0, 0, 0), -1)
        cv2.putText(
            thumb,
            f"{frame.frame_id}  {frame.timestamp_seconds:.1f}s",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            thumb,
            frame.category[:42],
            (8, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 255, 180),
            2,
            cv2.LINE_AA,
        )
        thumbnails.append(thumb)

    if not thumbnails:
        return None

    rows = (len(thumbnails) + columns - 1) // columns
    sheet = np.zeros((rows * thumb_height, columns * thumb_width, 3), dtype=np.uint8)
    for index, thumbnail in enumerate(thumbnails):
        row = index // columns
        column = index % columns
        y = row * thumb_height
        x = column * thumb_width
        sheet[y : y + thumb_height, x : x + thumb_width] = thumbnail

    path = output / "contact_sheet.jpg"
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"could not write selection contact sheet: {path}")
    return path


def _write_selection_report(path: Path, manifest: KeyframeSelectionManifest) -> None:
    lines = [
        "# Poker Legends Keyframe Selection",
        "",
        "## Source",
        f"- Ingest manifest: `{manifest.source_manifest}`",
    ]
    if manifest.contact_sheet is not None:
        lines.append(f"- Contact sheet: `{manifest.contact_sheet}`")
    lines.extend(
        [
            "",
            "## Purpose",
            "This subset is the first manual annotation batch. It keeps core poker states "
            "and non-table overlays separate so recognition can learn when to act and when "
            "to stand down.",
            "",
            "## Selected Frames",
            "| Frame | Time | Category | Why selected | Draft annotation |",
            "| --- | ---: | --- | --- | --- |",
        ]
    )
    for frame in manifest.frames:
        note = frame.note.replace("|", "/") if frame.note else ""
        lines.append(
            f"| `{frame.frame_id}` | {frame.timestamp_seconds:.1f}s | "
            f"{frame.category} | {note} | `{frame.annotation}` |"
        )
    lines.extend(
        [
            "",
            "## Annotation Pass",
            "- Fill the copied draft JSON files under `annotations/` rather than editing "
            "the full 177-frame ingest set first.",
            "- For table frames, mark hero cards, board cards, pot, visible stacks, "
            "action buttons, and current-actor/my-turn cues.",
            "- For overlay frames, mark the blocking dialog/menu region and treat them "
            "as no-action states for the bot.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selected_keyframe_from_dict(data: object) -> SelectedKeyframe:
    raw = cast(dict[str, object], data)
    return SelectedKeyframe(
        frame_id=str(raw["frame_id"]),
        image=str(raw["image"]),
        annotation=str(raw["annotation"]),
        source_image=str(raw["source_image"]),
        source_annotation=str(raw["source_annotation"]),
        frame_index=_to_int(raw["frame_index"]),
        timestamp_seconds=_to_float(raw["timestamp_seconds"]),
        reason=str(raw["reason"]),
        category=str(raw["category"]),
        note=str(raw["note"]),
    )


def _to_int(value: object) -> int:
    if type(value) is int:
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"expected int or string: {value!r}")


def _to_float(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"expected number or string: {value!r}")
