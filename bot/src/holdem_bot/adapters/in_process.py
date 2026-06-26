"""In-process adapter for testing against local GameState sources."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from holdem_common import Action, GameState

from holdem_bot.automate import Automator
from holdem_bot.capture import Capture, CapturedFrame
from holdem_bot.recognize import RecognitionResult, Recognizer


class StateCapture(Capture):
    def __init__(
        self,
        get_state: Callable[[], GameState],
        *,
        source: str = "in_process",
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._get_state = get_state
        self._source = source
        self._metadata = metadata or {}

    def capture(self) -> CapturedFrame:
        return CapturedFrame(
            payload=self._get_state(),
            source=self._source,
            metadata=self._metadata,
        )


class StateRecognizer(Recognizer):
    def recognize(self, frame: CapturedFrame) -> RecognitionResult:
        if not isinstance(frame.payload, GameState):
            raise TypeError("StateRecognizer expects a GameState payload")
        return RecognitionResult(
            state=frame.payload,
            confidence=1.0,
            metadata=frame.metadata,
        )


class ActionCallbackAutomator(Automator):
    def __init__(self, apply_action: Callable[[Action], None]) -> None:
        self._apply_action = apply_action

    def perform(self, action: Action, _state: GameState) -> None:
        self._apply_action(action)
