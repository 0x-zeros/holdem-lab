"""Poker Legends recognition adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from holdem_bot.capture import CapturedFrame
from holdem_bot.recognize import RecognitionResult, Recognizer
from holdem_bot.screen_state import ScreenState
from holdem_bot.vision.poker_legends_screen import detect_poker_legends_screen_state
from holdem_bot.vision.poker_legends_truth import screen_state_from_poker_legends_annotation


class PokerLegendsScreenStateRecognizer(Recognizer):
    """Classify Poker Legends frames into ScreenState before GameState extraction exists."""

    def recognize(self, frame: CapturedFrame) -> RecognitionResult:
        annotation = self._annotation_from_frame(frame)
        if annotation is None:
            image_path = self._image_path_from_frame(frame)
            if image_path is not None:
                detection = detect_poker_legends_screen_state(
                    image_path,
                    layout_annotation=self._layout_annotation_from_frame(frame),
                )
                return RecognitionResult(
                    state=None,
                    confidence=detection.screen.confidence,
                    metadata={
                        "source": frame.source,
                        "image": str(image_path),
                        "screen_kind": detection.screen.kind.value,
                        "active_primary_buttons": detection.active_primary_buttons,
                        "overlay_signals": detection.overlay_signals,
                    },
                    screen=detection.screen,
                )
            screen = ScreenState.unknown_or_transition(
                confidence=0.0,
                reason="unsupported Poker Legends frame payload",
            )
            return RecognitionResult(
                state=None,
                confidence=0.0,
                metadata={"source": frame.source, "screen_kind": screen.kind.value},
                screen=screen,
            )

        screen = screen_state_from_poker_legends_annotation(annotation)
        metadata = {
            "source": frame.source,
            "frame_id": annotation.get("frame_id"),
            "screen_kind": screen.kind.value,
            "blocking_reason": screen.blocking_reason,
            "hero_turn": screen.hero_turn,
        }
        return RecognitionResult(
            state=None,
            confidence=screen.confidence,
            metadata=metadata,
            screen=screen,
        )

    def _annotation_from_frame(self, frame: CapturedFrame) -> Mapping[str, object] | None:
        if isinstance(frame.payload, Mapping):
            return cast(Mapping[str, object], frame.payload)
        if isinstance(frame.payload, str | Path):
            path = Path(frame.payload)
            if path.suffix.lower() == ".json" and path.exists():
                return _read_json_object(path)
        metadata_annotation = frame.metadata.get("poker_legends_annotation")
        if isinstance(metadata_annotation, Mapping):
            return cast(Mapping[str, object], metadata_annotation)
        metadata_path = frame.metadata.get("poker_legends_annotation_path")
        if isinstance(metadata_path, str):
            path = Path(metadata_path)
            if path.suffix.lower() == ".json" and path.exists():
                return _read_json_object(path)
        return None

    def _image_path_from_frame(self, frame: CapturedFrame) -> Path | None:
        if isinstance(frame.payload, str | Path):
            path = Path(frame.payload)
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"} and path.exists():
                return path
        metadata_path = frame.metadata.get("poker_legends_image_path")
        if isinstance(metadata_path, str):
            path = Path(metadata_path)
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"} and path.exists():
                return path
        return None

    def _layout_annotation_from_frame(
        self,
        frame: CapturedFrame,
    ) -> Mapping[str, object] | None:
        annotation = frame.metadata.get("poker_legends_layout_annotation")
        if isinstance(annotation, Mapping):
            return cast(Mapping[str, object], annotation)
        annotation_path = frame.metadata.get("poker_legends_layout_annotation_path")
        if isinstance(annotation_path, str):
            path = Path(annotation_path)
            if path.suffix.lower() == ".json" and path.exists():
                return _read_json_object(path)
        return None


def _read_json_object(path: str | Path) -> dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, object], data)
