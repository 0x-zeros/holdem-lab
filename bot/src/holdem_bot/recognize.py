"""Recognition abstractions for translating capture output to GameState."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from holdem_common import GameState

from holdem_bot.capture import CapturedFrame


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    state: GameState
    confidence: float = 1.0
    metadata: Mapping[str, object] = field(default_factory=dict)


class Recognizer(Protocol):
    def recognize(self, frame: CapturedFrame) -> RecognitionResult:
        """Translate one capture output into the canonical GameState."""
        ...
