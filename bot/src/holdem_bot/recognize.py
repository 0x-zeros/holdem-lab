"""Recognition abstractions for translating capture output to GameState."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from holdem_common import GameState

from holdem_bot.capture import CapturedFrame
from holdem_bot.screen_state import ScreenKind, ScreenState


class RecognitionMode(StrEnum):
    IMAGE_ONLY_LIVE = "image_only_live"
    IMAGE_ONLY_REPLAY = "image_only_replay"
    TRUTH_ASSISTED_REPLAY = "truth_assisted_replay"
    SYNTHETIC_TEST = "synthetic_test"


class AssemblyStatus(StrEnum):
    BLOCKED_SCREEN = "blocked_screen"
    NO_STATE = "no_state"
    SINGLE_FRAME_VALID = "single_frame_valid"
    TEMPORALLY_UNSTABLE = "temporally_unstable"
    TEMPORALLY_STABLE_VALID = "temporally_stable_valid"
    INVALID = "invalid"
    UNSAFE_TRANSITION = "unsafe_transition"


class ValidityScope(StrEnum):
    NONE = "none"
    SINGLE_FRAME = "single_frame"
    TEMPORAL_WINDOW = "temporal_window"
    HAND_LOCKED = "hand_locked"


class ContractLevel(StrEnum):
    OBSERVE_ONLY = "observe_only"
    GAME_STATE = "game_state"
    POLICY_DECISION = "policy_decision"
    FOLD_CHECK_ONLY = "fold_check_only"
    CALL_DECISION = "call_decision"
    SIZING_DECISION = "sizing_decision"
    CLICK_PLAN = "click_plan"


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
class Candidate:
    value: object
    confidence: float
    source: str
    raw: object | None = None
    evidence: tuple[RoiEvidence, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "raw": self.raw,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class LayoutObservation:
    profile_id: str | None
    layout_version: str | None
    transform_type: str | None
    transform_residual_px: float | None
    anchor_scores: Mapping[str, float] = field(default_factory=dict)
    roi_generation: str | None = None
    confidence: float = 1.0
    image_size: tuple[int, int] | None = None
    source: str = "unknown"
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "layout_version": self.layout_version,
            "transform_type": self.transform_type,
            "transform_residual_px": self.transform_residual_px,
            "anchor_scores": dict(self.anchor_scores),
            "roi_generation": self.roi_generation,
            "confidence": self.confidence,
            "image_size": list(self.image_size) if self.image_size is not None else None,
            "source": self.source,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class CardSlotObservation:
    group: str
    slot: str
    occupancy: str
    rank_candidates: tuple[Candidate, ...] = ()
    suit_candidates: tuple[Candidate, ...] = ()
    card_candidates: tuple[Candidate, ...] = ()
    accepted_card: str | None = None
    locked_card: str | None = None
    accepted_by_single_frame: bool = False
    locked_by_tracker: bool = False
    confidence: float = 0.0
    consensus_components: Mapping[str, object] = field(default_factory=dict)
    evidence: RoiEvidence | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "group": self.group,
            "slot": self.slot,
            "occupancy": self.occupancy,
            "rank_candidates": [candidate.to_dict() for candidate in self.rank_candidates],
            "suit_candidates": [candidate.to_dict() for candidate in self.suit_candidates],
            "card_candidates": [candidate.to_dict() for candidate in self.card_candidates],
            "accepted_card": self.accepted_card,
            "locked_card": self.locked_card,
            "accepted_by_single_frame": self.accepted_by_single_frame,
            "locked_by_tracker": self.locked_by_tracker,
            "confidence": self.confidence,
            "consensus_components": dict(self.consensus_components),
            "evidence": self.evidence.to_dict() if self.evidence is not None else None,
        }


@dataclass(frozen=True, slots=True)
class ButtonObservation:
    slot: str
    visible: bool
    enabled: bool | None
    action_candidates: tuple[Candidate, ...] = ()
    accepted_action: str | None = None
    amount_candidates: tuple[Candidate, ...] = ()
    confidence: float = 0.0
    evidence: RoiEvidence | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "slot": self.slot,
            "visible": self.visible,
            "enabled": self.enabled,
            "action_candidates": [candidate.to_dict() for candidate in self.action_candidates],
            "accepted_action": self.accepted_action,
            "amount_candidates": [candidate.to_dict() for candidate in self.amount_candidates],
            "confidence": self.confidence,
            "evidence": self.evidence.to_dict() if self.evidence is not None else None,
        }


@dataclass(frozen=True, slots=True)
class ActionPanelObservation:
    panel_kind: str
    visible: bool
    enabled: bool | None
    hero_turn_indicator: bool | None
    row_bbox: tuple[int, int, int, int] | None
    buttons: tuple[ButtonObservation, ...] = ()
    confidence: float = 0.0
    ambiguity_flags: tuple[str, ...] = ()
    evidence: RoiEvidence | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "panel_kind": self.panel_kind,
            "visible": self.visible,
            "enabled": self.enabled,
            "hero_turn_indicator": self.hero_turn_indicator,
            "row_bbox": list(self.row_bbox) if self.row_bbox is not None else None,
            "buttons": [button.to_dict() for button in self.buttons],
            "confidence": self.confidence,
            "ambiguity_flags": list(self.ambiguity_flags),
            "evidence": self.evidence.to_dict() if self.evidence is not None else None,
        }


@dataclass(frozen=True, slots=True)
class NumericObservation:
    role: str
    group: str
    name: str
    visible: bool
    raw_text: str
    normalized_text: str | None
    unit: str | None
    scale: str | None
    candidates: tuple[Candidate, ...] = ()
    accepted_value: int | None = None
    ocr_confidence: float = 0.0
    parse_confidence: float = 0.0
    value_confidence: float = 0.0
    parser_version: str = "unknown"
    format_flags: tuple[str, ...] = ()
    evidence: RoiEvidence | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "group": self.group,
            "name": self.name,
            "visible": self.visible,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "unit": self.unit,
            "scale": self.scale,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "accepted_value": self.accepted_value,
            "ocr_confidence": self.ocr_confidence,
            "parse_confidence": self.parse_confidence,
            "value_confidence": self.value_confidence,
            "parser_version": self.parser_version,
            "format_flags": list(self.format_flags),
            "evidence": self.evidence.to_dict() if self.evidence is not None else None,
        }


@dataclass(frozen=True, slots=True)
class SeatObservation:
    seat: int
    occupied: bool | None
    in_hand: bool | None
    has_hole_cards: bool | None
    folded: bool | None
    all_in: bool | None
    sitting_out: bool | None
    current_actor: bool | None
    hero_seat: bool
    dealer_button_nearby: bool | None
    small_blind_marker: bool | None
    big_blind_marker: bool | None
    stack_candidates: tuple[Candidate, ...] = ()
    accepted_stack: int | None = None
    committed_current_street: int | None = None
    committed_total_hand: int | None = None
    showing_cards: tuple[CardSlotObservation, ...] = ()
    confidence: float = 0.0
    evidence: RoiEvidence | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "seat": self.seat,
            "occupied": self.occupied,
            "in_hand": self.in_hand,
            "has_hole_cards": self.has_hole_cards,
            "folded": self.folded,
            "all_in": self.all_in,
            "sitting_out": self.sitting_out,
            "current_actor": self.current_actor,
            "hero_seat": self.hero_seat,
            "dealer_button_nearby": self.dealer_button_nearby,
            "small_blind_marker": self.small_blind_marker,
            "big_blind_marker": self.big_blind_marker,
            "stack_candidates": [candidate.to_dict() for candidate in self.stack_candidates],
            "accepted_stack": self.accepted_stack,
            "committed_current_street": self.committed_current_street,
            "committed_total_hand": self.committed_total_hand,
            "showing_cards": [card.to_dict() for card in self.showing_cards],
            "confidence": self.confidence,
            "evidence": self.evidence.to_dict() if self.evidence is not None else None,
        }


@dataclass(frozen=True, slots=True)
class VisualObservation:
    frame: FrameEvidence
    recognition_mode: RecognitionMode
    screen: ScreenState
    layout: LayoutObservation
    cards: tuple[CardSlotObservation, ...] = ()
    action_panels: tuple[ActionPanelObservation, ...] = ()
    numbers: tuple[NumericObservation, ...] = ()
    seats: tuple[SeatObservation, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "frame": self.frame.to_dict(),
            "recognition_mode": self.recognition_mode.value,
            "screen": _screen_state_to_dict(self.screen),
            "layout": self.layout.to_dict(),
            "cards": [card.to_dict() for card in self.cards],
            "action_panels": [panel.to_dict() for panel in self.action_panels],
            "numbers": [number.to_dict() for number in self.numbers],
            "seats": [seat.to_dict() for seat in self.seats],
            "warnings": list(self.warnings),
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
class RecognitionSafetySummary:
    total_frames: int
    mode_counts: Mapping[str, int]
    screen_kind_counts: Mapping[str, int]
    assembly_status_counts: Mapping[str, int]
    contract_counts: Mapping[str, int]
    blocking_issue_counts: Mapping[str, int]
    expected_actionable_frames: int
    expected_non_actionable_frames: int
    false_actionable_count: int
    authorization_events: int
    unsafe_authorization_events: int
    stale_authorization_events: int
    truth_assisted_authorization_events: int
    source_policy_violation_count: int
    accepted_critical_wrong_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "total_frames": self.total_frames,
            "mode_counts": dict(self.mode_counts),
            "screen_kind_counts": dict(self.screen_kind_counts),
            "assembly_status_counts": dict(self.assembly_status_counts),
            "contract_counts": dict(self.contract_counts),
            "blocking_issue_counts": dict(self.blocking_issue_counts),
            "expected_actionable_frames": self.expected_actionable_frames,
            "expected_non_actionable_frames": self.expected_non_actionable_frames,
            "false_actionable_count": self.false_actionable_count,
            "authorization_events": self.authorization_events,
            "unsafe_authorization_events": self.unsafe_authorization_events,
            "stale_authorization_events": self.stale_authorization_events,
            "truth_assisted_authorization_events": self.truth_assisted_authorization_events,
            "source_policy_violation_count": self.source_policy_violation_count,
            "accepted_critical_wrong_count": self.accepted_critical_wrong_count,
        }


@dataclass(frozen=True, slots=True)
class Freshness:
    source_frame_id: str
    current_frame_revalidated: bool
    critical_fields_fresh: bool
    action_row_fresh: bool
    state_age_ms: int | None = None
    stable_frame_count: int = 0
    stable_duration_ms: int = 0
    tracker_hand_id: str | None = None
    tracker_generation: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_frame_id": self.source_frame_id,
            "current_frame_revalidated": self.current_frame_revalidated,
            "critical_fields_fresh": self.critical_fields_fresh,
            "action_row_fresh": self.action_row_fresh,
            "state_age_ms": self.state_age_ms,
            "stable_frame_count": self.stable_frame_count,
            "stable_duration_ms": self.stable_duration_ms,
            "tracker_hand_id": self.tracker_hand_id,
            "tracker_generation": self.tracker_generation,
        }


@dataclass(frozen=True, slots=True)
class ContractRequirement:
    field_path: str
    required_for: tuple[ContractLevel, ...]
    risk_tier: str
    min_confidence: float
    freshness_required: bool
    allowed_sources: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "field_path": self.field_path,
            "required_for": [level.value for level in self.required_for],
            "risk_tier": self.risk_tier,
            "min_confidence": self.min_confidence,
            "freshness_required": self.freshness_required,
            "allowed_sources": list(self.allowed_sources),
        }


@dataclass(frozen=True, slots=True)
class AssemblyIssue:
    issue_type: str
    reason_code: str
    field_path: str
    rule_name: str
    severity: str
    blocking: bool
    required_by_contract: tuple[ContractLevel, ...] = ()
    observed_value: object | None = None
    candidate_values: tuple[object, ...] = ()
    source: str | None = None
    evidence_refs: tuple[str, ...] = ()
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "issue_type": self.issue_type,
            "reason_code": self.reason_code,
            "field_path": self.field_path,
            "rule_name": self.rule_name,
            "severity": self.severity,
            "blocking": self.blocking,
            "required_by_contract": [level.value for level in self.required_by_contract],
            "observed_value": self.observed_value,
            "candidate_values": list(self.candidate_values),
            "source": self.source,
            "evidence_refs": list(self.evidence_refs),
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class GameStateAssemblyResult:
    status: AssemblyStatus
    validity_scope: ValidityScope
    state: GameState | None
    contract_level: ContractLevel
    contract_status: str
    valid_for: tuple[ContractLevel, ...]
    issues: tuple[AssemblyIssue, ...]
    freshness: Freshness
    field_confidences: Mapping[str, float] = field(default_factory=dict)
    critical_min_confidence: float | None = None
    layout_confidence: float = 0.0
    screen_confidence: float = 0.0
    rule_consistency: str = "not_evaluated"
    observation_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "validity_scope": self.validity_scope.value,
            "state_present": self.state is not None,
            "contract_level": self.contract_level.value,
            "contract_status": self.contract_status,
            "valid_for": [level.value for level in self.valid_for],
            "issues": [issue.to_dict() for issue in self.issues],
            "freshness": self.freshness.to_dict(),
            "field_confidences": dict(self.field_confidences),
            "critical_min_confidence": self.critical_min_confidence,
            "layout_confidence": self.layout_confidence,
            "screen_confidence": self.screen_confidence,
            "rule_consistency": self.rule_consistency,
            "observation_id": self.observation_id,
        }


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    state: GameState | None
    confidence: float = 1.0
    metadata: Mapping[str, object] = field(default_factory=dict)
    screen: ScreenState = field(default_factory=ScreenState.actionable_table)
    visual_observation: VisualObservation | None = None
    assembly_result: GameStateAssemblyResult | None = None
    recognition_mode: RecognitionMode = RecognitionMode.SYNTHETIC_TEST
    safety_contract: ContractLevel = ContractLevel.OBSERVE_ONLY
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


def summarize_recognition_safety(
    results: Iterable[RecognitionResult],
    *,
    expected_screen_kind_by_frame: Mapping[str, str] | None = None,
    expected_values_by_frame: Mapping[str, Mapping[str, object]] | None = None,
) -> RecognitionSafetySummary:
    expected_screen = expected_screen_kind_by_frame or {}
    expected_by_frame = expected_values_by_frame or {}
    total = 0
    mode_counts: dict[str, int] = {}
    screen_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    contract_counts: dict[str, int] = {}
    issue_counts: dict[str, int] = {}
    expected_actionable_frames = 0
    expected_non_actionable_frames = 0
    false_actionable_count = 0
    authorization_events = 0
    unsafe_authorization_events = 0
    stale_authorization_events = 0
    truth_assisted_authorization_events = 0
    source_policy_violation_count = 0
    accepted_critical_wrong_count = 0

    for result in results:
        total += 1
        _increment(mode_counts, result.recognition_mode.value)
        _increment(screen_counts, result.screen.kind.value)
        _increment(contract_counts, result.safety_contract.value)

        assembly = result.assembly_result
        if assembly is not None:
            _increment(status_counts, assembly.status.value)
            for issue in assembly.issues:
                if issue.blocking:
                    _increment(issue_counts, issue.reason_code)
            if (
                result.state is not None
                and not assembly.freshness.current_frame_revalidated
            ):
                stale_authorization_events += 1
        else:
            _increment(status_counts, "missing_assembly_result")

        frame_id = result.frame_evidence.frame_id if result.frame_evidence is not None else ""
        expected_kind = expected_screen.get(frame_id)
        false_authorization = False
        if expected_kind is not None:
            if expected_kind == ScreenKind.ACTIONABLE_TABLE.value:
                expected_actionable_frames += 1
            else:
                expected_non_actionable_frames += 1
                if result.screen.kind is ScreenKind.ACTIONABLE_TABLE:
                    false_actionable_count += 1
                if result.state is not None:
                    false_authorization = True
        evaluation = evaluate_accepted_critical_fields(
            result,
            expected_values=expected_by_frame.get(frame_id, {}),
        )
        authorization_events += evaluation.authorization_events
        if (
            evaluation.authorization_events
            and (evaluation.unsafe_authorization_events or false_authorization)
        ):
            unsafe_authorization_events += 1
        source_policy_violation_count += len(evaluation.source_policy_violations)
        accepted_critical_wrong_count += len(evaluation.accepted_critical_wrong_cases)
        if (
            result.state is not None
            and result.recognition_mode is RecognitionMode.TRUTH_ASSISTED_REPLAY
        ):
            truth_assisted_authorization_events += 1

    return RecognitionSafetySummary(
        total_frames=total,
        mode_counts=mode_counts,
        screen_kind_counts=screen_counts,
        assembly_status_counts=status_counts,
        contract_counts=contract_counts,
        blocking_issue_counts=issue_counts,
        expected_actionable_frames=expected_actionable_frames,
        expected_non_actionable_frames=expected_non_actionable_frames,
        false_actionable_count=false_actionable_count,
        authorization_events=authorization_events,
        unsafe_authorization_events=unsafe_authorization_events,
        stale_authorization_events=stale_authorization_events,
        truth_assisted_authorization_events=truth_assisted_authorization_events,
        source_policy_violation_count=source_policy_violation_count,
        accepted_critical_wrong_count=accepted_critical_wrong_count,
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


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _screen_state_to_dict(screen: ScreenState) -> dict[str, object]:
    return {
        "kind": screen.kind.value,
        "confidence": screen.confidence,
        "reason": screen.reason,
        "blocking_reason": screen.blocking_reason,
        "hero_turn": screen.hero_turn,
        "metadata": dict(screen.metadata),
    }
