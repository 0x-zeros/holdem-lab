"""Bot orchestration: capture, recognize, decide, automate."""

from __future__ import annotations

from dataclasses import dataclass

from holdem_ai import decide
from holdem_common import Action, GameState

from holdem_bot.automate import Automator
from holdem_bot.capture import Capture
from holdem_bot.recognize import Recognizer


@dataclass(frozen=True, slots=True)
class BotStepResult:
    acted: bool
    reason: str
    state: GameState | None = None
    action: Action | None = None
    confidence: float = 0.0


class BotOrchestrator:
    def __init__(
        self,
        *,
        capture: Capture,
        recognizer: Recognizer,
        automator: Automator,
        seat: int,
        min_confidence: float = 0.80,
    ) -> None:
        if seat < 0:
            raise ValueError("seat cannot be negative")
        if min_confidence < 0.0 or min_confidence > 1.0:
            raise ValueError("min_confidence must be between 0 and 1")

        self.capture = capture
        self.recognizer = recognizer
        self.automator = automator
        self.seat = seat
        self.min_confidence = min_confidence

    def run_once(self) -> BotStepResult:
        frame = self.capture.capture()
        recognition = self.recognizer.recognize(frame)
        state = recognition.state

        if recognition.confidence < self.min_confidence:
            return BotStepResult(
                acted=False,
                reason="low_confidence",
                state=state,
                confidence=recognition.confidence,
            )

        if state.current_seat is None:
            return BotStepResult(
                acted=False,
                reason="terminal",
                state=state,
                confidence=recognition.confidence,
            )

        if state.current_seat != self.seat:
            return BotStepResult(
                acted=False,
                reason="waiting",
                state=state,
                confidence=recognition.confidence,
            )

        action = decide(state)
        self.automator.perform(action, state)
        return BotStepResult(
            acted=True,
            reason="acted",
            state=state,
            action=action,
            confidence=recognition.confidence,
        )
