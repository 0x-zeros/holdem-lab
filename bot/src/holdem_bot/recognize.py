"""Recognition abstractions for translating capture output to GameState."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from holdem_common import GameState

from holdem_bot.capture import CapturedFrame
from holdem_bot.screen_state import ScreenState


class RecognitionMode(StrEnum):
    IMAGE_ONLY_LIVE = "image_only_live"
    IMAGE_ONLY_REPLAY = "image_only_replay"
    TRUTH_ASSISTED_REPLAY = "truth_assisted_replay"
    SYNTHETIC_TEST = "synthetic_test"


IMAGE_ONLY_RECOGNITION_MODES = frozenset(
    {RecognitionMode.IMAGE_ONLY_LIVE, RecognitionMode.IMAGE_ONLY_REPLAY}
)
REVIEWED_TRUTH_SOURCE = "reviewed_truth"
TRUTH_ASSISTED_SOURCE_BLOCK_REASON = "truth_assisted_source_in_image_only_mode"


@dataclass(frozen=True, slots=True)
class FrameEvidence:
    session_id: str | None
    frame_id: str
    frame_seq: int | None = None
    wall_timestamp: str | None = None
    monotonic_timestamp_ms: int | None = None
    image_hash: str | None = None
    image_size: tuple[int, int] | None = None
    capture_backend: str | None = None
    window_id: str | None = None
    window_title_hash: str | None = None
    dpi_scale: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "frame_id": self.frame_id,
            "frame_seq": self.frame_seq,
            "wall_timestamp": self.wall_timestamp,
            "monotonic_timestamp_ms": self.monotonic_timestamp_ms,
            "image_hash": self.image_hash,
            "image_size": list(self.image_size) if self.image_size is not None else None,
            "capture_backend": self.capture_backend,
            "window_id": self.window_id,
            "window_title_hash": self.window_title_hash,
            "dpi_scale": self.dpi_scale,
        }


@dataclass(frozen=True, slots=True)
class RoiEvidence:
    roi_id: str
    roi_rect_screen: tuple[int, int, int, int] | None = None
    roi_rect_canonical: tuple[float, float, float, float] | None = None
    crop_hash: str | None = None
    crop_path: str | None = None
    layout_version: str | None = None
    layout_transform_version: str | None = None
    crop_quality: float | None = None
    occlusion_score: float | None = None
    blur_score: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "roi_id": self.roi_id,
            "roi_rect_screen": list(self.roi_rect_screen)
            if self.roi_rect_screen is not None
            else None,
            "roi_rect_canonical": list(self.roi_rect_canonical)
            if self.roi_rect_canonical is not None
            else None,
            "crop_hash": self.crop_hash,
            "crop_path": self.crop_path,
            "layout_version": self.layout_version,
            "layout_transform_version": self.layout_transform_version,
            "crop_quality": self.crop_quality,
            "occlusion_score": self.occlusion_score,
            "blur_score": self.blur_score,
        }


@dataclass(frozen=True, slots=True)
class AcceptedCriticalField:
    field_path: str
    source: str
    value: object | None = None
    evidence_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "field_path": self.field_path,
            "source": self.source,
            "value": self.value,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True, slots=True)
class SourcePolicyViolation:
    field_path: str
    source: str
    recognition_mode: RecognitionMode
    reason: str = TRUTH_ASSISTED_SOURCE_BLOCK_REASON

    def to_dict(self) -> dict[str, object]:
        return {
            "field_path": self.field_path,
            "source": self.source,
            "recognition_mode": self.recognition_mode.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class AcceptedCriticalFieldMismatch:
    field_path: str
    expected: object
    accepted: object | None
    source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "field_path": self.field_path,
            "expected": self.expected,
            "accepted": self.accepted,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class AcceptedCriticalFieldEvaluation:
    authorization_events: int
    source_policy_violations: tuple[SourcePolicyViolation, ...]
    accepted_critical_wrong_cases: tuple[AcceptedCriticalFieldMismatch, ...]

    @property
    def unsafe_authorization_events(self) -> int:
        if self.source_policy_violations or self.accepted_critical_wrong_cases:
            return self.authorization_events
        return 0

    def to_dict(self) -> dict[str, object]:
        return {
            "authorization_events": self.authorization_events,
            "unsafe_authorization_events": self.unsafe_authorization_events,
            "source_policy_violations": [
                violation.to_dict() for violation in self.source_policy_violations
            ],
            "accepted_critical_wrong_cases": [
                mismatch.to_dict() for mismatch in self.accepted_critical_wrong_cases
            ],
        }


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    state: GameState | None
    confidence: float = 1.0
    metadata: Mapping[str, object] = field(default_factory=dict)
    screen: ScreenState = field(default_factory=ScreenState.actionable_table)
    recognition_mode: RecognitionMode = RecognitionMode.SYNTHETIC_TEST
    frame_evidence: FrameEvidence | None = None
    accepted_critical_fields: tuple[AcceptedCriticalField, ...] = ()
    source_policy_violations: tuple[SourcePolicyViolation, ...] = ()


class Recognizer(Protocol):
    def recognize(self, frame: CapturedFrame) -> RecognitionResult:
        """Translate one capture output into the canonical GameState."""
        ...


def coerce_recognition_mode(value: object) -> RecognitionMode | None:
    if isinstance(value, RecognitionMode):
        return value
    if isinstance(value, str):
        try:
            return RecognitionMode(value)
        except ValueError:
            return None
    return None


def recognition_mode_from_frame(
    frame: CapturedFrame,
    *,
    has_annotation: bool,
) -> RecognitionMode:
    explicit = coerce_recognition_mode(frame.metadata.get("recognition_mode"))
    if explicit is not None:
        return explicit
    if has_annotation:
        return RecognitionMode.TRUTH_ASSISTED_REPLAY
    if _looks_live_source(frame.source, frame.metadata):
        return RecognitionMode.IMAGE_ONLY_LIVE
    return RecognitionMode.IMAGE_ONLY_REPLAY


def frame_evidence_from_frame(
    frame: CapturedFrame,
    *,
    frame_id: str,
    image_path: str | Path | None = None,
) -> FrameEvidence:
    metadata = frame.metadata
    return FrameEvidence(
        session_id=_optional_str(metadata.get("session_id")),
        frame_id=frame_id,
        frame_seq=_optional_int(metadata.get("frame_seq")),
        wall_timestamp=_optional_str(metadata.get("wall_timestamp")),
        monotonic_timestamp_ms=_optional_int(metadata.get("monotonic_timestamp_ms")),
        image_hash=_image_hash(image_path),
        image_size=_image_size(metadata.get("image_size")),
        capture_backend=_optional_str(metadata.get("capture_backend") or metadata.get("source")),
        window_id=_optional_str(metadata.get("window_id")),
        window_title_hash=_optional_str(metadata.get("window_title_hash")),
        dpi_scale=_optional_float(metadata.get("dpi_scale")),
    )


def source_policy_violations_for_fields(
    recognition_mode: RecognitionMode,
    accepted_fields: tuple[AcceptedCriticalField, ...],
) -> tuple[SourcePolicyViolation, ...]:
    if recognition_mode not in IMAGE_ONLY_RECOGNITION_MODES:
        return ()
    return tuple(
        SourcePolicyViolation(
            field_path=field.field_path,
            source=field.source,
            recognition_mode=recognition_mode,
        )
        for field in accepted_fields
        if field.source == REVIEWED_TRUTH_SOURCE
    )


def evaluate_accepted_critical_fields(
    result: RecognitionResult,
    *,
    expected_values: Mapping[str, object] | None = None,
) -> AcceptedCriticalFieldEvaluation:
    expected = expected_values or {}
    wrong_cases = tuple(
        AcceptedCriticalFieldMismatch(
            field_path=field.field_path,
            expected=expected[field.field_path],
            accepted=field.value,
            source=field.source,
        )
        for field in result.accepted_critical_fields
        if field.field_path in expected and field.value != expected[field.field_path]
    )
    return AcceptedCriticalFieldEvaluation(
        authorization_events=1 if result.state is not None else 0,
        source_policy_violations=result.source_policy_violations,
        accepted_critical_wrong_cases=wrong_cases,
    )


def _looks_live_source(source: str, metadata: Mapping[str, object]) -> bool:
    lowered = source.lower()
    if "live" in lowered:
        return True
    return "capture_command" in metadata


def _image_hash(image_path: str | Path | None) -> str | None:
    if image_path is None:
        return None
    path = Path(image_path)
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _image_size(value: object) -> tuple[int, int] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    width = _optional_int(value[0])
    height = _optional_int(value[1])
    if width is None or height is None:
        return None
    return (width, height)


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, int):
        return str(value)
    return None


def _optional_int(value: object) -> int | None:
    if type(value) is int:
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None
