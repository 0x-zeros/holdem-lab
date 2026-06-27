"""Bot orchestration: capture, recognize, decide, automate."""

from __future__ import annotations

from dataclasses import dataclass

from holdem_ai import PolicyDecision, explain_decision
from holdem_common import Action, GameState

from holdem_bot.automate import Automator
from holdem_bot.capture import Capture
from holdem_bot.recognize import Recognizer
from holdem_bot.screen_state import ScreenState, evaluate_safety


@dataclass(frozen=True, slots=True)
class BotStepResult:
    acted: bool
    reason: str
    state: GameState | None = None
    action: Action | None = None
    policy_decision: PolicyDecision | None = None
    confidence: float = 0.0
    screen: ScreenState | None = None


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
        decision = evaluate_safety(
            screen=recognition.screen,
            state=recognition.state,
            recognition_confidence=recognition.confidence,
            controlled_seat=self.seat,
            min_confidence=self.min_confidence,
        )

        if not decision.allowed:
            return BotStepResult(
                acted=False,
                reason=decision.reason,
                state=decision.state,
                confidence=decision.confidence,
                screen=decision.screen,
            )

        state = decision.state
        if state is None:
            raise RuntimeError("safety gate allowed an action without a GameState")
        policy_decision = explain_decision(state)
        action = policy_decision.action
        self.automator.perform(action, state)
        return BotStepResult(
            acted=True,
            reason="acted",
            state=state,
            action=action,
            policy_decision=policy_decision,
            confidence=decision.confidence,
            screen=decision.screen,
        )
