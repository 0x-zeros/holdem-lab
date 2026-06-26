"""Screen-state classification and bot safety gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from holdem_common import GameState


class ScreenKind(StrEnum):
    ACTIONABLE_TABLE = "actionable_table"
    TABLE_OBSERVE = "table_observe"
    BLOCKED_OVERLAY = "blocked_overlay"
    NON_TABLE_UI = "non_table_ui"
    UNKNOWN_OR_TRANSITION = "unknown_or_transition"


@dataclass(frozen=True, slots=True)
class ScreenState:
    kind: ScreenKind
    confidence: float = 1.0
    reason: str = ""
    blocking_reason: str | None = None
    hero_turn: bool | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("screen confidence must be between 0 and 1")

    @classmethod
    def actionable_table(
        cls,
        *,
        confidence: float = 1.0,
        hero_turn: bool | None = None,
        reason: str = "actionable table",
    ) -> ScreenState:
        return cls(
            kind=ScreenKind.ACTIONABLE_TABLE,
            confidence=confidence,
            reason=reason,
            hero_turn=hero_turn,
        )

    @classmethod
    def table_observe(
        cls,
        *,
        confidence: float = 1.0,
        reason: str = "table observe",
    ) -> ScreenState:
        return cls(kind=ScreenKind.TABLE_OBSERVE, confidence=confidence, reason=reason)

    @classmethod
    def blocked_overlay(
        cls,
        *,
        blocking_reason: str,
        confidence: float = 1.0,
        reason: str = "blocked overlay",
    ) -> ScreenState:
        return cls(
            kind=ScreenKind.BLOCKED_OVERLAY,
            confidence=confidence,
            reason=reason,
            blocking_reason=blocking_reason,
        )

    @classmethod
    def non_table_ui(
        cls,
        *,
        confidence: float = 1.0,
        reason: str = "non-table UI",
    ) -> ScreenState:
        return cls(kind=ScreenKind.NON_TABLE_UI, confidence=confidence, reason=reason)

    @classmethod
    def unknown_or_transition(
        cls,
        *,
        confidence: float = 0.0,
        reason: str = "unknown or transition",
    ) -> ScreenState:
        return cls(kind=ScreenKind.UNKNOWN_OR_TRANSITION, confidence=confidence, reason=reason)


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    allowed: bool
    reason: str
    screen: ScreenState
    state: GameState | None
    confidence: float


def evaluate_safety(
    *,
    screen: ScreenState,
    state: GameState | None,
    recognition_confidence: float,
    controlled_seat: int,
    min_confidence: float,
) -> SafetyDecision:
    if controlled_seat < 0:
        raise ValueError("controlled seat cannot be negative")
    if min_confidence < 0.0 or min_confidence > 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    if recognition_confidence < 0.0 or recognition_confidence > 1.0:
        raise ValueError("recognition confidence must be between 0 and 1")

    confidence = min(recognition_confidence, screen.confidence)
    if recognition_confidence < min_confidence:
        return _blocked("low_confidence", screen, state, confidence)
    if screen.confidence < min_confidence:
        return _blocked("screen_low_confidence", screen, state, confidence)
    if screen.kind is ScreenKind.BLOCKED_OVERLAY:
        return _blocked("blocked_overlay", screen, state, confidence)
    if screen.kind is ScreenKind.NON_TABLE_UI:
        return _blocked("non_table_ui", screen, state, confidence)
    if screen.kind is ScreenKind.UNKNOWN_OR_TRANSITION:
        return _blocked("unknown_or_transition", screen, state, confidence)
    if screen.kind is ScreenKind.TABLE_OBSERVE:
        return _blocked("table_observe", screen, state, confidence)

    if state is None:
        return _blocked("no_game_state", screen, state, confidence)
    if state.current_seat is None:
        return _blocked("terminal", screen, state, confidence)
    if screen.hero_turn is False:
        return _blocked("waiting", screen, state, confidence)
    if state.current_seat != controlled_seat:
        return _blocked("waiting", screen, state, confidence)
    if not state.legal_actions:
        return _blocked("no_legal_actions", screen, state, confidence)

    return SafetyDecision(
        allowed=True,
        reason="safe_to_act",
        screen=screen,
        state=state,
        confidence=confidence,
    )


def _blocked(
    reason: str,
    screen: ScreenState,
    state: GameState | None,
    confidence: float,
) -> SafetyDecision:
    return SafetyDecision(
        allowed=False,
        reason=reason,
        screen=screen,
        state=state,
        confidence=confidence,
    )
