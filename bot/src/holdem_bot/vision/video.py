"""Video ingestion for Poker Legends CV sample collection."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self, cast

import cv2
import numpy as np
from numpy.typing import NDArray

GrayFingerprint = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    source: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class ExtractedFrame:
    image: str
    annotation: str
    frame_index: int
    timestamp_seconds: float
    mean_abs_diff: float | None
    reason: str


@dataclass(frozen=True, slots=True)
class VideoIngestManifest:
    schema_version: int
    metadata: VideoMetadata
    sample_fps: float
    diff_threshold: float
    max_gap_seconds: float
    resize_width: int | None
    contact_sheet: str | None
    process_report: str
    frames: tuple[ExtractedFrame, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported video ingest manifest schema version")

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
            metadata=_metadata_from_dict(data["metadata"]),
            sample_fps=float(data["sample_fps"]),
            diff_threshold=float(data["diff_threshold"]),
            max_gap_seconds=float(data["max_gap_seconds"]),
            resize_width=None if data["resize_width"] is None else int(data["resize_width"]),
            contact_sheet=None if data["contact_sheet"] is None else str(data["contact_sheet"]),
            process_report=str(data["process_report"]),
            frames=tuple(_frame_from_dict(frame) for frame in data["frames"]),
        )


def ingest_video(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    sample_fps: float = 2.0,
    diff_threshold: float = 4.0,
    max_gap_seconds: float = 5.0,
    resize_width: int | None = 1600,
    max_frames: int | None = None,
) -> VideoIngestManifest:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if diff_threshold < 0:
        raise ValueError("diff_threshold cannot be negative")
    if max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be positive")
    if resize_width is not None and resize_width <= 0:
        raise ValueError("resize_width must be positive")

    video = Path(video_path)
    output = Path(output_dir)
    frames_dir = output / "frames"
    annotations_dir = output / "annotations"
    frames_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    try:
        metadata = _read_metadata(cap, video)
        sample_interval = max(1, int(round(metadata.fps / sample_fps))) if metadata.fps > 0 else 1
        extracted = _extract_keyframes(
            cap,
            output,
            frames_dir,
            annotations_dir,
            metadata,
            sample_interval=sample_interval,
            diff_threshold=diff_threshold,
            max_gap_seconds=max_gap_seconds,
            resize_width=resize_width,
            max_frames=max_frames,
        )
    finally:
        cap.release()

    contact_sheet = _write_contact_sheet(output, frames_dir, extracted)
    process_report = output / "process_report.md"
    manifest = VideoIngestManifest(
        schema_version=1,
        metadata=metadata,
        sample_fps=sample_fps,
        diff_threshold=diff_threshold,
        max_gap_seconds=max_gap_seconds,
        resize_width=resize_width,
        contact_sheet=None if contact_sheet is None else str(contact_sheet.relative_to(output)),
        process_report=str(process_report.relative_to(output)),
        frames=tuple(extracted),
    )
    manifest.write_json(output / "manifest.json")
    _write_process_report(process_report, manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract deduplicated keyframes from a poker video."
    )
    parser.add_argument("video", help="Input video path.")
    parser.add_argument("--out", required=True, help="Output directory for frames and manifest.")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Frame sampling rate.")
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=4.0,
        help="Mean absolute grayscale fingerprint delta required to keep a changed frame.",
    )
    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=5.0,
        help="Keep a frame after this much time even if it is visually similar.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=1600,
        help="Resize extracted frames to this width. Use 0 to keep original size.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    manifest = ingest_video(
        args.video,
        args.out,
        sample_fps=args.sample_fps,
        diff_threshold=args.diff_threshold,
        max_gap_seconds=args.max_gap_seconds,
        resize_width=None if args.resize_width == 0 else args.resize_width,
        max_frames=args.max_frames,
    )
    print(
        json.dumps(
            {
                "video": manifest.metadata.source,
                "duration_seconds": manifest.metadata.duration_seconds,
                "frames": len(manifest.frames),
                "manifest": str(Path(args.out) / "manifest.json"),
                "process_report": str(Path(args.out) / manifest.process_report),
            },
            indent=2,
            sort_keys=True,
        ),
    )


def _read_metadata(cap: cv2.VideoCapture, video: Path) -> VideoMetadata:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    return VideoMetadata(
        source=str(video),
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_seconds=duration,
    )


def _extract_keyframes(
    cap: cv2.VideoCapture,
    output: Path,
    frames_dir: Path,
    annotations_dir: Path,
    metadata: VideoMetadata,
    *,
    sample_interval: int,
    diff_threshold: float,
    max_gap_seconds: float,
    resize_width: int | None,
    max_frames: int | None,
) -> list[ExtractedFrame]:
    extracted: list[ExtractedFrame] = []
    last_fingerprint: GrayFingerprint | None = None
    last_timestamp: float | None = None
    frame_index = 0

    while True:
        if max_frames is not None and len(extracted) >= max_frames:
            break

        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame = cast(NDArray[np.uint8], frame)
        if frame_index % sample_interval != 0:
            frame_index += 1
            continue

        frame = _resize_frame(frame, resize_width)
        timestamp = frame_index / metadata.fps if metadata.fps > 0 else float(frame_index)
        fingerprint = _fingerprint(frame)
        diff = None if last_fingerprint is None else _mean_abs_diff(fingerprint, last_fingerprint)
        reason = _keep_reason(
            diff=diff,
            timestamp=timestamp,
            last_timestamp=last_timestamp,
            diff_threshold=diff_threshold,
            max_gap_seconds=max_gap_seconds,
        )
        if reason is None:
            frame_index += 1
            continue

        frame_name = f"keyframe_{len(extracted):06d}.png"
        annotation_name = f"keyframe_{len(extracted):06d}.json"
        frame_path = frames_dir / frame_name
        annotation_path = annotations_dir / annotation_name
        if not cv2.imwrite(str(frame_path), frame):
            raise RuntimeError(f"could not write frame: {frame_path}")

        image_rel = frame_path.relative_to(output)
        annotation_rel = annotation_path.relative_to(output)
        _write_draft_annotation(
            annotation_path,
            metadata,
            image=str(image_rel),
            image_width=int(frame.shape[1]),
            image_height=int(frame.shape[0]),
            frame_index=frame_index,
            timestamp_seconds=timestamp,
        )
        extracted.append(
            ExtractedFrame(
                image=str(image_rel),
                annotation=str(annotation_rel),
                frame_index=frame_index,
                timestamp_seconds=timestamp,
                mean_abs_diff=diff,
                reason=reason,
            ),
        )
        last_fingerprint = fingerprint
        last_timestamp = timestamp
        frame_index += 1

    return extracted


def _fingerprint(frame: NDArray[np.uint8]) -> GrayFingerprint:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    return cast(GrayFingerprint, resized)


def _resize_frame(frame: NDArray[np.uint8], resize_width: int | None) -> NDArray[np.uint8]:
    if resize_width is None or frame.shape[1] <= resize_width:
        return frame
    scale = resize_width / frame.shape[1]
    height = max(1, int(round(frame.shape[0] * scale)))
    return cast(
        NDArray[np.uint8],
        cv2.resize(frame, (resize_width, height), interpolation=cv2.INTER_AREA),
    )


def _mean_abs_diff(current: GrayFingerprint, previous: GrayFingerprint) -> float:
    return float(np.mean(np.abs(current.astype(np.int16) - previous.astype(np.int16))))


def _keep_reason(
    *,
    diff: float | None,
    timestamp: float,
    last_timestamp: float | None,
    diff_threshold: float,
    max_gap_seconds: float,
) -> str | None:
    if diff is None or last_timestamp is None:
        return "first"
    if diff >= diff_threshold:
        return "changed"
    if timestamp - last_timestamp >= max_gap_seconds:
        return "max_gap"
    return None


def _write_draft_annotation(
    path: Path,
    metadata: VideoMetadata,
    *,
    image: str,
    image_width: int,
    image_height: int,
    frame_index: int,
    timestamp_seconds: float,
) -> None:
    draft = {
        "schema_version": 1,
        "source": "poker_legends_video",
        "image": image,
        "video": metadata.source,
        "width": image_width,
        "height": image_height,
        "source_width": metadata.width,
        "source_height": metadata.height,
        "frame_index": frame_index,
        "timestamp_seconds": timestamp_seconds,
        "todo": [
            "current_seat",
            "hero_hole_cards",
            "board",
            "pot",
            "seat_stacks",
            "seat_committed",
            "buttons",
        ],
        "regions": {
            "board": [],
            "seats": [],
            "buttons": [],
            "texts": [],
        },
    }
    path.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_contact_sheet(
    output: Path,
    frames_dir: Path,
    frames: list[ExtractedFrame],
    *,
    columns: int = 4,
    max_images: int = 24,
) -> Path | None:
    if not frames:
        return None

    selected = frames[:max_images]
    thumbnails: list[NDArray[np.uint8]] = []
    thumb_width, thumb_height = 320, 196
    for frame in selected:
        image = cv2.imread(str(output / frame.image), cv2.IMREAD_COLOR)
        if image is None:
            continue
        thumb = cast(
            NDArray[np.uint8],
            cv2.resize(image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA),
        )
        cv2.putText(
            thumb,
            f"{frame.timestamp_seconds:.1f}s",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
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

    path = frames_dir.parent / "contact_sheet.jpg"
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"could not write contact sheet: {path}")
    return path


def _write_process_report(path: Path, manifest: VideoIngestManifest) -> None:
    metadata = manifest.metadata
    lines = [
        "# Poker Legends Video Ingest Report",
        "",
        "## Source",
        f"- Video: `{metadata.source}`",
        f"- Size: {metadata.width}x{metadata.height}",
        f"- FPS: {metadata.fps:.3f}",
        f"- Frames: {metadata.frame_count}",
        f"- Duration: {metadata.duration_seconds:.2f}s",
        "",
        "## Extraction",
        f"- Sample FPS: {manifest.sample_fps}",
        f"- Difference threshold: {manifest.diff_threshold}",
        f"- Max gap seconds: {manifest.max_gap_seconds}",
        f"- Resize width: {manifest.resize_width or 'original'}",
        f"- Keyframes kept: {len(manifest.frames)}",
    ]
    if manifest.contact_sheet is not None:
        lines.append(f"- Contact sheet: `{manifest.contact_sheet}`")
    lines.extend(
        [
            "",
            "## Outputs",
            "- `manifest.json`: machine-readable list of kept keyframes.",
            "- `frames/`: extracted PNG keyframes.",
            "- `annotations/`: draft JSON files to fill with Poker Legends regions and values.",
            "- `process_report.md`: this human-readable processing summary.",
            "",
            "## Next Annotation Pass",
            "For the first pass, fill regions for a small representative subset:",
            "- hero hole cards",
            "- board cards",
            "- pot text",
            "- seat stack/commitment text",
            "- action buttons",
            "- current actor / my-turn indicator",
        ],
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metadata_from_dict(data: object) -> VideoMetadata:
    raw = cast(dict[str, object], data)
    return VideoMetadata(
        source=str(raw["source"]),
        fps=_to_float(raw["fps"]),
        frame_count=_to_int(raw["frame_count"]),
        width=_to_int(raw["width"]),
        height=_to_int(raw["height"]),
        duration_seconds=_to_float(raw["duration_seconds"]),
    )


def _frame_from_dict(data: object) -> ExtractedFrame:
    raw = cast(dict[str, object], data)
    raw_diff = raw["mean_abs_diff"]
    return ExtractedFrame(
        image=str(raw["image"]),
        annotation=str(raw["annotation"]),
        frame_index=_to_int(raw["frame_index"]),
        timestamp_seconds=_to_float(raw["timestamp_seconds"]),
        mean_abs_diff=None if raw_diff is None else _to_float(raw_diff),
        reason=str(raw["reason"]),
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
