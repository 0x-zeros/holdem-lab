"""Poker Legends recognition adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, cast

from holdem_common import Action, ActionType, Card, GameState, PlayerState, Pot, Street

from holdem_bot.capture import CapturedFrame
from holdem_bot.recognize import (
    IMAGE_ONLY_RECOGNITION_MODES,
    REVIEWED_TRUTH_SOURCE,
    TRUTH_ASSISTED_SOURCE_BLOCK_REASON,
    AcceptedCriticalField,
    ActionPanelObservation,
    AssemblyIssue,
    AssemblyStatus,
    ButtonObservation,
    Candidate,
    CardSlotObservation,
    ContractLevel,
    FrameEvidence,
    Freshness,
    GameStateAssemblyResult,
    LayoutObservation,
    NumericObservation,
    RecognitionMode,
    RecognitionResult,
    Recognizer,
    RoiEvidence,
    SeatObservation,
    SourcePolicyViolation,
    ValidityScope,
    VisualObservation,
    frame_evidence_from_frame,
    recognition_mode_from_frame,
    source_policy_violations_for_fields,
)
from holdem_bot.screen_state import ScreenKind, ScreenState
from holdem_bot.vision.poker_legends_buttons import (
    PokerLegendsButtonPrediction,
    PokerLegendsButtonRecognizer,
)
from holdem_bot.vision.poker_legends_card_consensus import (
    PokerLegendsCardConsensusPrediction,
    PokerLegendsCardConsensusRecognizer,
)
from holdem_bot.vision.poker_legends_numbers import (
    PokerLegendsNumberPrediction,
    PokerLegendsNumberRecognizer,
    parse_poker_legends_chip_amount,
)
from holdem_bot.vision.poker_legends_screen import detect_poker_legends_screen_state
from holdem_bot.vision.poker_legends_truth import screen_state_from_poker_legends_annotation
from holdem_bot.vision.recognition import (
    RecognizedButton,
    RecognizedCard,
    RecognizedSeat,
    RecognizedTable,
)

_ACTION_MAP = {
    "fold": ActionType.FOLD,
    "check": ActionType.CHECK,
    "call": ActionType.CALL,
    "bet": ActionType.BET,
    "raise": ActionType.RAISE,
    "all_in": ActionType.ALL_IN,
}
_STREET_MAP = {
    "preflop": Street.PREFLOP,
    "flop": Street.FLOP,
    "turn": Street.TURN,
    "river": Street.RIVER,
    "showdown": Street.SHOWDOWN,
}


class _LlmStubRecognizer:
    """No-op CV recognizer for LLM-mode tables (state comes from the LLM, not pixels)."""

    def recognize(self, *args: object, **kwargs: object) -> tuple[object, ...]:
        return ()


class _CardConsensusRecognizer(Protocol):
    def recognize(
        self,
        image_path: str | Path,
        annotation: Mapping[str, object],
        *,
        frame_id: str,
        exclude_frame_id: str | None = None,
        exclude_card: str | None = None,
    ) -> tuple[PokerLegendsCardConsensusPrediction, ...]: ...


class _ButtonRecognizer(Protocol):
    def recognize(
        self,
        image_path: str | Path,
        annotation: Mapping[str, object],
        *,
        frame_id: str,
        exclude_frame_id: str | None = None,
    ) -> tuple[PokerLegendsButtonPrediction, ...]: ...


class _NumberRecognizer(Protocol):
    def recognize(
        self,
        image_path: str | Path,
        annotation: Mapping[str, object],
        *,
        text_names: tuple[str, ...] = ("pot", "hero_stack", "right_top_stack"),
        button_names: tuple[str, ...] = ("primary_left",),
    ) -> tuple[PokerLegendsNumberPrediction, ...]: ...


class PokerLegendsScreenStateRecognizer(Recognizer):
    """Classify Poker Legends frames into ScreenState before GameState extraction exists."""

    def recognize(self, frame: CapturedFrame) -> RecognitionResult:
        annotation = self._annotation_from_frame(frame)
        image_path = self._image_path_from_frame(frame)
        mode = recognition_mode_from_frame(frame, has_annotation=annotation is not None)
        frame_id = _frame_id_from_inputs(annotation, image_path)
        evidence = frame_evidence_from_frame(frame, frame_id=frame_id, image_path=image_path)
        if annotation is None:
            if image_path is not None:
                detection = detect_poker_legends_screen_state(
                    image_path,
                    layout_annotation=self._layout_annotation_from_frame(frame),
                )
                metadata = _slice0_metadata(
                    {
                        "source": frame.source,
                        "image": str(image_path),
                        "screen_kind": detection.screen.kind.value,
                        "active_primary_buttons": detection.active_primary_buttons,
                        "overlay_signals": detection.overlay_signals,
                    },
                    recognition_mode=mode,
                    frame_evidence=evidence,
                )
                return RecognitionResult(
                    state=None,
                    confidence=detection.screen.confidence,
                    metadata=metadata,
                    screen=detection.screen,
                    recognition_mode=mode,
                    frame_evidence=evidence,
                )
            screen = ScreenState.unknown_or_transition(
                confidence=0.0,
                reason="unsupported Poker Legends frame payload",
            )
            metadata = _slice0_metadata(
                {"source": frame.source, "screen_kind": screen.kind.value},
                recognition_mode=mode,
                frame_evidence=evidence,
            )
            return RecognitionResult(
                state=None,
                confidence=0.0,
                metadata=metadata,
                screen=screen,
                recognition_mode=mode,
                frame_evidence=evidence,
            )

        if mode in IMAGE_ONLY_RECOGNITION_MODES:
            if image_path is not None:
                detection = detect_poker_legends_screen_state(
                    image_path,
                    layout_annotation=self._layout_annotation_from_frame(frame),
                )
                screen = detection.screen
                metadata = _slice0_metadata(
                    {
                        "source": frame.source,
                        "image": str(image_path),
                        "screen_kind": screen.kind.value,
                        "active_primary_buttons": detection.active_primary_buttons,
                        "overlay_signals": detection.overlay_signals,
                    },
                    recognition_mode=mode,
                    frame_evidence=evidence,
                )
                return RecognitionResult(
                    state=None,
                    confidence=screen.confidence,
                    metadata=metadata,
                    screen=screen,
                    recognition_mode=mode,
                    frame_evidence=evidence,
                )
            screen = ScreenState.unknown_or_transition(
                confidence=0.0,
                reason="image-only mode requires image-based screen detection",
            )
            metadata = _slice0_metadata(
                {"source": frame.source, "screen_kind": screen.kind.value},
                recognition_mode=mode,
                frame_evidence=evidence,
            )
            return RecognitionResult(
                state=None,
                confidence=0.0,
                metadata=metadata,
                screen=screen,
                recognition_mode=mode,
                frame_evidence=evidence,
            )

        screen = screen_state_from_poker_legends_annotation(annotation)
        metadata = _slice0_metadata(
            {
                "source": frame.source,
                "frame_id": annotation.get("frame_id"),
                "screen_kind": screen.kind.value,
                "blocking_reason": screen.blocking_reason,
                "hero_turn": screen.hero_turn,
            },
            recognition_mode=mode,
            frame_evidence=evidence,
        )
        return RecognitionResult(
            state=None,
            confidence=screen.confidence,
            metadata=metadata,
            screen=screen,
            recognition_mode=mode,
            frame_evidence=evidence,
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


class PokerLegendsTableRecognizer(PokerLegendsScreenStateRecognizer):
    """Recognize actionable Poker Legends table frames into a fail-closed GameState."""

    def __init__(
        self,
        *,
        card_recognizer: _CardConsensusRecognizer,
        button_recognizer: _ButtonRecognizer,
        number_recognizer: _NumberRecognizer | None = None,
        controlled_seat: int = 0,
        small_blind: int = 5,
        big_blind: int = 10,
        min_card_confidence: float = 0.20,
        min_button_confidence: float = 0.20,
        min_number_confidence: float = 0.70,
    ) -> None:
        if controlled_seat < 0:
            raise ValueError("controlled seat cannot be negative")
        if small_blind < 0 or big_blind <= 0:
            raise ValueError("blind amounts must be non-negative with a positive big blind")
        if not 0.0 <= min_number_confidence <= 1.0:
            raise ValueError("minimum number confidence must be between 0 and 1")
        self.card_recognizer = card_recognizer
        self.button_recognizer = button_recognizer
        self.number_recognizer = number_recognizer
        self.controlled_seat = controlled_seat
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.min_card_confidence = min_card_confidence
        self.min_button_confidence = min_button_confidence
        self.min_number_confidence = min_number_confidence

    @classmethod
    def from_manifests(
        cls,
        *,
        card_part_manifest: str | Path,
        card_classifier_manifest: str | Path,
        button_manifest: str | Path,
        card_template_manifest: str | Path | None = None,
        controlled_seat: int = 0,
    ) -> PokerLegendsTableRecognizer:
        return cls(
            card_recognizer=PokerLegendsCardConsensusRecognizer.from_manifests(
                card_template_manifest=card_template_manifest,
                card_part_manifest=card_part_manifest,
                card_classifier_manifest=card_classifier_manifest,
            ),
            button_recognizer=PokerLegendsButtonRecognizer.from_manifest(button_manifest),
            number_recognizer=PokerLegendsNumberRecognizer(),
            controlled_seat=controlled_seat,
        )

    @classmethod
    def for_llm(
        cls,
        *,
        controlled_seat: int = 0,
        small_blind: int = 5,
        big_blind: int = 10,
    ) -> PokerLegendsTableRecognizer:
        """Build a recognizer that assembles state from an LLM annotation (no CV manifests)."""
        stub = cast(Any, _LlmStubRecognizer())
        return cls(
            card_recognizer=stub,
            button_recognizer=stub,
            number_recognizer=None,
            controlled_seat=controlled_seat,
            small_blind=small_blind,
            big_blind=big_blind,
        )

    def recognize_from_llm_annotation(
        self,
        annotation: Mapping[str, object],
        *,
        image: str = "",
        frame_id: str | None = None,
    ) -> RecognitionResult:
        """Assemble a fail-closed GameState from an LLM-produced annotation.

        The annotation follows ``annotation_output_schema`` (cards/seats/pot/buttons/blinds
        all read by the LLM); no pixel CV runs. The same ``_state_from_table`` guards apply.
        """
        fid = frame_id or str(annotation.get("frame_id") or "llm")
        mode = RecognitionMode.TRUTH_ASSISTED_REPLAY
        frame_evidence = FrameEvidence(session_id=None, frame_id=fid)
        raw_ts = annotation.get("table_state")
        table_state: Mapping[str, object] = raw_ts if isinstance(raw_ts, Mapping) else {}
        hero_cards = _llm_cards(annotation.get("hero_hole_cards"))
        board = _llm_cards(annotation.get("board"))
        hero_turn = _llm_hero_current(annotation, controlled_name="hero")
        seats = _recognized_seats_from_annotation(
            annotation,
            controlled_seat=self.controlled_seat,
            hero_hole_cards=hero_cards,
            number_predictions=(),
            hero_current=hero_turn,
        )
        buttons = _recognized_buttons_from_predictions(
            annotation, button_predictions=(), number_predictions=()
        )
        pot = _pot_from_annotation(annotation)
        if not pot:
            committed_total = sum(seat.committed for seat in seats)
            if committed_total > 0:
                pot = committed_total
        street = _street_name_from_annotation(annotation, board)
        confidence = _to_float(table_state.get("confidence"), default=0.0)
        screen = _llm_screen_state(table_state, hero_turn=hero_turn, confidence=confidence)
        table = RecognizedTable(
            source="poker_legends_llm",
            image=image,
            hand_id=fid,
            street=street,
            current_seat=self.controlled_seat if hero_turn else _current_seat(seats),
            pot=pot,
            board=board,
            seats=seats,
            buttons=buttons,
            confidence=confidence if confidence > 0 else screen.confidence,
        )
        metadata = _slice0_metadata(
            {
                "source": "poker_legends_llm",
                "screen_kind": screen.kind.value,
                "recognized_table": _recognized_table_to_dict(table),
            },
            recognition_mode=mode,
            frame_evidence=frame_evidence,
        )
        visual = _visual_observation_from_parts(
            frame_evidence=frame_evidence,
            recognition_mode=mode,
            screen=screen,
            layout_annotation=None,
            table=table,
            card_predictions=(),
            button_predictions=(),
            number_predictions=(),
            controlled_seat=self.controlled_seat,
        )
        if screen.kind is not ScreenKind.ACTIONABLE_TABLE:
            assembly = _assembly_result_from_outcome(
                state=None,
                block_reason="screen_not_actionable",
                screen=screen,
                observation=visual,
                table_confidence=table.confidence,
                accepted_fields=(),
                source_policy_violations=(),
            )
            metadata = _slice0_metadata(
                metadata,
                recognition_mode=mode,
                frame_evidence=frame_evidence,
                visual_observation=visual,
                assembly_result=assembly,
            )
            return RecognitionResult(
                state=None,
                confidence=screen.confidence,
                metadata=metadata,
                screen=screen,
                visual_observation=visual,
                assembly_result=assembly,
                recognition_mode=mode,
                safety_contract=assembly.contract_level,
                frame_evidence=frame_evidence,
            )
        state, block_reason = self._state_from_table(
            table,
            annotation=annotation,
            screen=screen,
            card_predictions=(),
            number_predictions=(),
        )
        accepted_fields = _accepted_critical_fields_for_state(
            state=state,
            table=table,
            annotation=annotation,
            screen_source="vlm_annotation",
            annotation_source="vlm_annotation",
            card_predictions=(),
            button_predictions=(),
            number_predictions=(),
            controlled_seat=self.controlled_seat,
        )
        metadata = _slice0_metadata(
            metadata,
            recognition_mode=mode,
            frame_evidence=frame_evidence,
            accepted_critical_fields=accepted_fields,
        )
        if block_reason is not None:
            metadata["state_block_reason"] = block_reason
        assembly = _assembly_result_from_outcome(
            state=state,
            block_reason=block_reason,
            screen=screen,
            observation=visual,
            table_confidence=table.confidence,
            accepted_fields=accepted_fields,
            source_policy_violations=(),
        )
        metadata = _slice0_metadata(
            metadata,
            recognition_mode=mode,
            frame_evidence=frame_evidence,
            accepted_critical_fields=accepted_fields,
            visual_observation=visual,
            assembly_result=assembly,
        )
        return RecognitionResult(
            state=state,
            confidence=table.confidence,
            metadata=metadata,
            screen=screen,
            visual_observation=visual,
            assembly_result=assembly,
            recognition_mode=mode,
            safety_contract=assembly.contract_level,
            frame_evidence=frame_evidence,
            accepted_critical_fields=accepted_fields,
        )

    def recognize(self, frame: CapturedFrame) -> RecognitionResult:
        annotation = self._annotation_from_frame(frame)
        image_path = self._image_path_from_frame(frame)
        layout_annotation = self._layout_annotation_from_frame(frame)
        mode = recognition_mode_from_frame(frame, has_annotation=annotation is not None)
        frame_id = _frame_id_from_inputs(annotation, image_path)
        frame_evidence = frame_evidence_from_frame(
            frame,
            frame_id=frame_id,
            image_path=image_path,
        )
        context = _FrameContext(
            annotation=annotation,
            image_path=image_path,
            layout_annotation=layout_annotation,
            recognition_mode=mode,
            frame_evidence=frame_evidence,
        )
        if context.recognition_mode in IMAGE_ONLY_RECOGNITION_MODES and annotation is not None:
            violation = SourcePolicyViolation(
                field_path="poker_legends_annotation",
                source=REVIEWED_TRUTH_SOURCE,
                recognition_mode=context.recognition_mode,
            )
            screen = ScreenState.unknown_or_transition(
                confidence=0.0,
                reason=TRUTH_ASSISTED_SOURCE_BLOCK_REASON,
            )
            visual = _visual_observation_from_context(context, screen=screen)
            assembly = _assembly_result_from_outcome(
                state=None,
                block_reason=TRUTH_ASSISTED_SOURCE_BLOCK_REASON,
                screen=screen,
                observation=visual,
                table_confidence=0.0,
                accepted_fields=(),
                source_policy_violations=(violation,),
            )
            metadata = _slice0_metadata(
                {
                    "source": frame.source,
                    "screen_kind": screen.kind.value,
                    "state_block_reason": TRUTH_ASSISTED_SOURCE_BLOCK_REASON,
                },
                recognition_mode=context.recognition_mode,
                frame_evidence=context.frame_evidence,
                source_policy_violations=(violation,),
                visual_observation=visual,
                assembly_result=assembly,
            )
            if context.image_path is not None:
                metadata["image"] = str(context.image_path)
            metadata["frame_id"] = context.frame_id
            return RecognitionResult(
                state=None,
                confidence=0.0,
                metadata=metadata,
                screen=screen,
                visual_observation=visual,
                assembly_result=assembly,
                recognition_mode=context.recognition_mode,
                safety_contract=assembly.contract_level,
                frame_evidence=context.frame_evidence,
                source_policy_violations=(violation,),
            )
        screen = self._screen_state(frame, context)
        visual = _visual_observation_from_context(context, screen=screen)
        metadata = _slice0_metadata(
            {
                "source": frame.source,
                "screen_kind": screen.kind.value,
            },
            recognition_mode=context.recognition_mode,
            frame_evidence=context.frame_evidence,
            visual_observation=visual,
        )
        if context.annotation is not None:
            metadata["frame_id"] = context.frame_id
        if context.image_path is not None:
            metadata["image"] = str(context.image_path)

        if screen.kind is not ScreenKind.ACTIONABLE_TABLE:
            assembly = _assembly_result_from_outcome(
                state=None,
                block_reason="screen_not_actionable",
                screen=screen,
                observation=visual,
                table_confidence=screen.confidence,
                accepted_fields=(),
                source_policy_violations=(),
            )
            metadata = _slice0_metadata(
                metadata,
                recognition_mode=context.recognition_mode,
                frame_evidence=context.frame_evidence,
                visual_observation=visual,
                assembly_result=assembly,
            )
            return RecognitionResult(
                state=None,
                confidence=screen.confidence,
                metadata=metadata,
                screen=screen,
                visual_observation=visual,
                assembly_result=assembly,
                recognition_mode=context.recognition_mode,
                safety_contract=assembly.contract_level,
                frame_evidence=context.frame_evidence,
            )
        if context.image_path is None:
            metadata["state_block_reason"] = "missing_image_path"
            assembly = _assembly_result_from_outcome(
                state=None,
                block_reason="missing_image_path",
                screen=screen,
                observation=visual,
                table_confidence=0.0,
                accepted_fields=(),
                source_policy_violations=(),
            )
            metadata = _slice0_metadata(
                metadata,
                recognition_mode=context.recognition_mode,
                frame_evidence=context.frame_evidence,
                visual_observation=visual,
                assembly_result=assembly,
            )
            return RecognitionResult(
                state=None,
                confidence=0.0,
                metadata=metadata,
                screen=screen,
                visual_observation=visual,
                assembly_result=assembly,
                recognition_mode=context.recognition_mode,
                safety_contract=assembly.contract_level,
                frame_evidence=context.frame_evidence,
            )
        if context.layout_annotation is None:
            metadata["state_block_reason"] = "missing_layout_annotation"
            assembly = _assembly_result_from_outcome(
                state=None,
                block_reason="missing_layout_annotation",
                screen=screen,
                observation=visual,
                table_confidence=0.0,
                accepted_fields=(),
                source_policy_violations=(),
            )
            metadata = _slice0_metadata(
                metadata,
                recognition_mode=context.recognition_mode,
                frame_evidence=context.frame_evidence,
                visual_observation=visual,
                assembly_result=assembly,
            )
            return RecognitionResult(
                state=None,
                confidence=0.0,
                metadata=metadata,
                screen=screen,
                visual_observation=visual,
                assembly_result=assembly,
                recognition_mode=context.recognition_mode,
                safety_contract=assembly.contract_level,
                frame_evidence=context.frame_evidence,
            )

        card_predictions = self.card_recognizer.recognize(
            context.image_path,
            context.layout_annotation,
            frame_id=context.frame_id,
        )
        button_predictions = self.button_recognizer.recognize(
            context.image_path,
            context.layout_annotation,
            frame_id=context.frame_id,
        )
        number_predictions: tuple[PokerLegendsNumberPrediction, ...] = ()
        if self.number_recognizer is not None:
            text_names, button_names = _number_roi_names_for_fallbacks(context.annotation)
            try:
                if text_names or button_names:
                    number_predictions = self.number_recognizer.recognize(
                        context.image_path,
                        context.layout_annotation,
                        text_names=text_names,
                        button_names=button_names,
                    )
            except Exception as exc:
                metadata["number_recognizer_error"] = f"{type(exc).__name__}: {exc}"
        accepted_number_predictions = tuple(
            prediction
            for prediction in number_predictions
            if prediction.normalized_number is not None
            and prediction.confidence >= self.min_number_confidence
        )
        table = _recognized_table_from_predictions(
            source=frame.source,
            image=str(context.image_path),
            frame_id=context.frame_id,
            screen=screen,
            annotation=context.annotation,
            card_predictions=card_predictions,
            button_predictions=button_predictions,
            number_predictions=accepted_number_predictions,
            controlled_seat=self.controlled_seat,
        )
        visual = _visual_observation_from_context(
            context,
            screen=screen,
            table=table,
            card_predictions=card_predictions,
            button_predictions=button_predictions,
            number_predictions=accepted_number_predictions,
            controlled_seat=self.controlled_seat,
        )
        state, block_reason = self._state_from_table(
            table,
            annotation=context.annotation,
            screen=screen,
            card_predictions=card_predictions,
            number_predictions=accepted_number_predictions,
        )
        metadata["recognized_table"] = _recognized_table_to_dict(table)
        if number_predictions:
            metadata["number_predictions"] = [
                prediction.to_dict() for prediction in number_predictions
            ]
            metadata["accepted_number_predictions"] = [
                prediction.to_dict() for prediction in accepted_number_predictions
            ]
        if block_reason is not None:
            metadata["state_block_reason"] = block_reason
        accepted_fields = _accepted_critical_fields_for_state(
            state=state,
            table=table,
            annotation=context.annotation,
            screen_source=_screen_source(context),
            annotation_source=REVIEWED_TRUTH_SOURCE,
            card_predictions=card_predictions,
            button_predictions=button_predictions,
            number_predictions=accepted_number_predictions,
            controlled_seat=self.controlled_seat,
        )
        policy_violations = source_policy_violations_for_fields(
            context.recognition_mode,
            accepted_fields,
        )
        if policy_violations:
            state = None
            metadata["state_block_reason"] = TRUTH_ASSISTED_SOURCE_BLOCK_REASON
        raw_block_reason = metadata.get("state_block_reason")
        assembly_block_reason = raw_block_reason if isinstance(raw_block_reason, str) else None
        assembly = _assembly_result_from_outcome(
            state=state,
            block_reason=assembly_block_reason,
            screen=screen,
            observation=visual,
            table_confidence=table.confidence,
            accepted_fields=accepted_fields,
            source_policy_violations=policy_violations,
        )
        metadata = _slice0_metadata(
            metadata,
            recognition_mode=context.recognition_mode,
            frame_evidence=context.frame_evidence,
            accepted_critical_fields=accepted_fields,
            source_policy_violations=policy_violations,
            visual_observation=visual,
            assembly_result=assembly,
        )
        return RecognitionResult(
            state=state,
            confidence=table.confidence,
            metadata=metadata,
            screen=screen,
            visual_observation=visual,
            assembly_result=assembly,
            recognition_mode=context.recognition_mode,
            safety_contract=assembly.contract_level,
            frame_evidence=context.frame_evidence,
            accepted_critical_fields=accepted_fields,
            source_policy_violations=policy_violations,
        )

    def _screen_state(self, frame: CapturedFrame, context: _FrameContext) -> ScreenState:
        if context.annotation is not None:
            return screen_state_from_poker_legends_annotation(context.annotation)
        if context.image_path is not None:
            detection = detect_poker_legends_screen_state(
                context.image_path,
                layout_annotation=context.layout_annotation,
            )
            return detection.screen
        return ScreenState.unknown_or_transition(
            confidence=0.0,
            reason=f"unsupported Poker Legends frame payload from {frame.source}",
        )

    def _state_from_table(
        self,
        table: RecognizedTable,
        *,
        annotation: Mapping[str, object] | None,
        screen: ScreenState,
        card_predictions: tuple[PokerLegendsCardConsensusPrediction, ...],
        number_predictions: tuple[PokerLegendsNumberPrediction, ...],
    ) -> tuple[GameState | None, str | None]:
        if annotation is None:
            return None, "missing_table_metadata"
        if screen.hero_turn is not True:
            return None, "hero_not_current"
        if table.pot is None:
            return None, "missing_pot"
        if not table.seats:
            return None, "missing_seats"
        hero = _seat_by_number(table.seats, self.controlled_seat)
        if hero is None:
            return None, "missing_hero_seat"
        if hero.stack < 0:
            return None, "missing_hero_stack"
        if len(tuple(card for card in hero.hole_cards if card.visible and card.card)) != 2:
            return None, "missing_hero_hole_cards"
        if any(
            prediction.visible
            and prediction.card is not None
            and prediction.confidence < self.min_card_confidence
            for prediction in card_predictions
        ):
            return None, "low_card_confidence"
        small_blind, big_blind = _blinds_from_annotation(
            annotation, default_small=self.small_blind, default_big=self.big_blind
        )
        if _has_visible_primary_preselect_or_shortcut_button(annotation):
            return None, "preselect_ambiguous"
        legal_actions, to_call, action_block_reason = self._legal_actions(
            table.buttons,
            annotation=annotation,
            number_predictions=number_predictions,
            hero_stack=hero.stack,
            hero_committed=hero.committed,
            big_blind=big_blind,
        )
        if action_block_reason is not None:
            return None, action_block_reason
        if not legal_actions:
            return None, "missing_legal_actions"

        street = _street_from_table(table)
        if street is None:
            return None, "missing_street"
        board_cards = _cards_from_recognized_cards(table.board)
        expected_board_count = _expected_board_count(street)
        if expected_board_count is not None and len(board_cards) != expected_board_count:
            return None, "board_count_mismatch"
        # An unknown (negative-sentinel) stack on ANY seat would raise out of
        # PlayerState; fail closed instead of crashing (hero is already guarded).
        if any(seat.stack < 0 for seat in table.seats):
            return None, "missing_seat_stack"
        players = tuple(
            PlayerState(
                seat=seat.seat,
                stack=seat.stack,
                committed=seat.committed,
                hole_cards=_cards_from_recognized_cards(seat.hole_cards)
                if seat.seat == self.controlled_seat
                else (),
                active=seat.active,
            )
            for seat in table.seats
        )
        if len(players) < 2:
            return None, "not_enough_players"
        button_seat, button_seat_source = _resolve_button_seat(table.seats, self.controlled_seat)
        return (
            GameState(
                hand_id=table.hand_id,
                street=street,
                players=players,
                board=board_cards,
                pots=(
                    Pot(
                        amount=table.pot,
                        eligible_seats=frozenset(seat.seat for seat in players),
                    ),
                ),
                current_seat=self.controlled_seat,
                button_seat=button_seat,
                small_blind=small_blind,
                big_blind=big_blind,
                min_raise=big_blind,
                to_call=to_call,
                legal_actions=legal_actions,
                metadata={
                    "source": "poker_legends_prototype",
                    "recognizer": "PokerLegendsTableRecognizer",
                    "screen_kind": screen.kind.value,
                    "button_seat": button_seat,
                    "button_seat_source": button_seat_source,
                    "small_blind": small_blind,
                    "big_blind": big_blind,
                },
            ),
            None,
        )

    def _legal_actions(
        self,
        buttons: tuple[RecognizedButton, ...],
        *,
        annotation: Mapping[str, object],
        number_predictions: tuple[PokerLegendsNumberPrediction, ...],
        hero_stack: int,
        hero_committed: int,
        big_blind: int,
    ) -> tuple[tuple[Action, ...], int, str | None]:
        # First pass: gate confidence and resolve the amount to call, so the raise
        # floor is sized correctly regardless of where the call button appears.
        if not buttons:
            return (), 0, "missing_current_action_row"
        resolved: list[ActionType] = []
        to_call = 0
        for button in buttons:
            if button.action_type is None:
                continue
            if button.confidence < self.min_button_confidence:
                return (), 0, "low_button_confidence"
            action_type = _ACTION_MAP.get(button.action_type)
            if action_type is None:
                continue
            if action_type is ActionType.CALL:
                label = _truth_button_label(annotation, button.command)
                if label is not None and _is_preselect_or_shortcut_label(label):
                    return (), 0, "preselect_ambiguous"
                amount = _button_amount(
                    annotation,
                    button.command,
                    number_predictions,
                    hero_committed=hero_committed,
                )
                if amount is None:
                    return (), 0, "missing_call_amount"
                to_call = amount
            resolved.append(action_type)

        # Second pass: a legal min raise must first match the current bet
        # (hero_committed + to_call) and then add at least one big blind, capped at
        # the all-in stack. The old floor ignored to_call and could fall below the
        # call amount, i.e. propose an illegal raise.
        max_raise_to = hero_stack + hero_committed
        min_raise_to = min(max_raise_to, hero_committed + to_call + big_blind)
        # A raise is only legal if hero can put in more than the call (otherwise the
        # only options are call / all-in). Drop a RAISE button in that underwater
        # corner rather than emit a raise-to at or below the call amount.
        can_raise = max_raise_to > hero_committed + to_call
        actions: list[Action] = []
        for action_type in resolved:
            if action_type is ActionType.CALL:
                actions.append(Action(ActionType.CALL, amount=to_call))
            elif action_type is ActionType.RAISE:
                if can_raise:
                    actions.append(
                        Action(
                            ActionType.RAISE,
                            amount=min_raise_to,
                            min_amount=min_raise_to,
                            max_amount=max_raise_to,
                        )
                    )
            else:
                actions.append(Action(action_type))
        action_types = {action.action_type for action in actions}
        if to_call == 0 and not action_types.intersection({ActionType.CHECK, ActionType.BET}):
            return (), 0, "missing_passive_action"
        if to_call > 0 and ActionType.CALL not in action_types:
            return (), 0, "missing_call_action"
        return tuple(actions), to_call, None


class _FrameContext:
    def __init__(
        self,
        *,
        annotation: Mapping[str, object] | None,
        image_path: Path | None,
        layout_annotation: Mapping[str, object] | None,
        recognition_mode: RecognitionMode,
        frame_evidence: FrameEvidence,
    ) -> None:
        self.annotation = annotation
        self.image_path = image_path
        self.layout_annotation = layout_annotation
        self.recognition_mode = recognition_mode
        self.frame_evidence = frame_evidence

    @property
    def frame_id(self) -> str:
        return self.frame_evidence.frame_id


def _slice0_metadata(
    metadata: Mapping[str, object],
    *,
    recognition_mode: RecognitionMode,
    frame_evidence: FrameEvidence,
    accepted_critical_fields: tuple[AcceptedCriticalField, ...] = (),
    source_policy_violations: tuple[SourcePolicyViolation, ...] = (),
    visual_observation: VisualObservation | None = None,
    assembly_result: GameStateAssemblyResult | None = None,
) -> dict[str, object]:
    merged = dict(metadata)
    merged["recognition_mode"] = recognition_mode.value
    merged["frame_evidence"] = frame_evidence.to_dict()
    if accepted_critical_fields:
        merged["accepted_critical_fields"] = [
            field.to_dict() for field in accepted_critical_fields
        ]
    if source_policy_violations:
        merged["source_policy_violations"] = [
            violation.to_dict() for violation in source_policy_violations
        ]
    if visual_observation is not None:
        merged["visual_observation"] = visual_observation.to_dict()
    if assembly_result is not None:
        merged["assembly_result"] = assembly_result.to_dict()
        for issue in assembly_result.issues:
            if issue.blocking:
                merged.setdefault("state_block_reason", issue.reason_code.lower())
                break
    return merged


def _visual_observation_from_context(
    context: _FrameContext,
    *,
    screen: ScreenState,
    table: RecognizedTable | None = None,
    card_predictions: tuple[PokerLegendsCardConsensusPrediction, ...] = (),
    button_predictions: tuple[PokerLegendsButtonPrediction, ...] = (),
    number_predictions: tuple[PokerLegendsNumberPrediction, ...] = (),
    controlled_seat: int = 0,
) -> VisualObservation:
    return _visual_observation_from_parts(
        frame_evidence=context.frame_evidence,
        recognition_mode=context.recognition_mode,
        screen=screen,
        layout_annotation=context.layout_annotation,
        table=table,
        card_predictions=card_predictions,
        button_predictions=button_predictions,
        number_predictions=number_predictions,
        controlled_seat=controlled_seat,
    )


def _visual_observation_from_parts(
    *,
    frame_evidence: FrameEvidence,
    recognition_mode: RecognitionMode,
    screen: ScreenState,
    layout_annotation: Mapping[str, object] | None,
    table: RecognizedTable | None,
    card_predictions: tuple[PokerLegendsCardConsensusPrediction, ...],
    button_predictions: tuple[PokerLegendsButtonPrediction, ...],
    number_predictions: tuple[PokerLegendsNumberPrediction, ...],
    controlled_seat: int,
) -> VisualObservation:
    return VisualObservation(
        frame=frame_evidence,
        recognition_mode=recognition_mode,
        screen=screen,
        layout=_layout_observation(layout_annotation, frame_evidence=frame_evidence),
        cards=_card_slot_observations(
            table,
            card_predictions,
            controlled_seat=controlled_seat,
        ),
        action_panels=_action_panel_observations(
            screen,
            button_predictions,
            number_predictions,
            table=table,
        ),
        numbers=_numeric_observations(number_predictions),
        seats=_seat_observations(table, controlled_seat=controlled_seat),
        warnings=(),
    )


def _layout_observation(
    annotation: Mapping[str, object] | None,
    *,
    frame_evidence: FrameEvidence,
) -> LayoutObservation:
    if annotation is None:
        return LayoutObservation(
            profile_id=None,
            layout_version=None,
            transform_type=None,
            transform_residual_px=None,
            confidence=0.0,
            image_size=frame_evidence.image_size,
            source="missing_layout_annotation",
            warnings=("missing_layout_annotation",),
        )
    profile = annotation.get("layout_profile") or annotation.get("profile_id")
    layout_version = annotation.get("layout_version") or annotation.get("layout")
    return LayoutObservation(
        profile_id=str(profile) if isinstance(profile, str) and profile else None,
        layout_version=str(layout_version)
        if isinstance(layout_version, str) and layout_version
        else None,
        transform_type="static_roi",
        transform_residual_px=None,
        anchor_scores={},
        roi_generation=str(annotation.get("roi_generation"))
        if isinstance(annotation.get("roi_generation"), str)
        else None,
        confidence=1.0,
        image_size=_layout_image_size(annotation) or frame_evidence.image_size,
        source="layout_annotation",
        warnings=(),
    )


def _layout_image_size(annotation: Mapping[str, object]) -> tuple[int, int] | None:
    width = _optional_int(annotation.get("width"))
    height = _optional_int(annotation.get("height"))
    if width is None or height is None:
        return None
    return (width, height)


def _card_slot_observations(
    table: RecognizedTable | None,
    predictions: tuple[PokerLegendsCardConsensusPrediction, ...],
    *,
    controlled_seat: int,
) -> tuple[CardSlotObservation, ...]:
    if predictions:
        return tuple(_card_slot_from_prediction(prediction) for prediction in predictions)
    if table is None:
        return ()
    cards: list[CardSlotObservation] = []
    for card in table.board:
        cards.append(_card_slot_from_recognized_card("board", card, source=table.source))
    for seat in table.seats:
        for card in seat.hole_cards:
            group = (
                "hero_hole_cards"
                if seat.seat == controlled_seat
                else f"seat_{seat.seat}_hole_cards"
            )
            cards.append(_card_slot_from_recognized_card(group, card, source=table.source))
    return tuple(cards)


def _card_slot_from_prediction(
    prediction: PokerLegendsCardConsensusPrediction,
) -> CardSlotObservation:
    evidence = RoiEvidence(roi_id=f"card:{prediction.group}:{prediction.slot}")
    candidate = (
        Candidate(
            value=prediction.card,
            confidence=prediction.confidence,
            source="image_card_consensus",
            raw=prediction.to_dict(),
            evidence=(evidence,),
        ),
    ) if prediction.card is not None else ()
    return CardSlotObservation(
        group=prediction.group,
        slot=prediction.slot,
        occupancy=_card_occupancy(prediction.visible, prediction.card),
        card_candidates=candidate,
        accepted_card=prediction.card if prediction.visible else None,
        accepted_by_single_frame=prediction.visible and prediction.card is not None,
        confidence=prediction.confidence,
        consensus_components={
            "method": prediction.method,
            "full_card": prediction.full_card,
            "part_card": prediction.part_card,
            "classifier_card": prediction.classifier_card,
            "full_confidence": prediction.full_confidence,
            "part_confidence": prediction.part_confidence,
            "classifier_confidence": prediction.classifier_confidence,
        },
        evidence=evidence,
    )


def _card_slot_from_recognized_card(
    group: str,
    card: RecognizedCard,
    *,
    source: str,
) -> CardSlotObservation:
    evidence = RoiEvidence(roi_id=f"card:{group}:{card.slot}")
    candidate = (
        Candidate(
            value=card.card,
            confidence=card.confidence,
            source=source,
            evidence=(evidence,),
        ),
    ) if card.card is not None else ()
    return CardSlotObservation(
        group=group,
        slot=card.slot,
        occupancy=_card_occupancy(card.visible, card.card),
        card_candidates=candidate,
        accepted_card=card.card if card.visible else None,
        accepted_by_single_frame=card.visible and card.card is not None,
        confidence=card.confidence,
        evidence=evidence,
    )


def _card_occupancy(visible: bool, card: str | None) -> str:
    if visible and card is not None:
        return "face_up"
    if visible:
        return "unknown"
    return "empty"


def _action_panel_observations(
    screen: ScreenState,
    button_predictions: tuple[PokerLegendsButtonPrediction, ...],
    number_predictions: tuple[PokerLegendsNumberPrediction, ...],
    *,
    table: RecognizedTable | None,
) -> tuple[ActionPanelObservation, ...]:
    button_observations = _accepted_button_observations(table) or tuple(
        _button_observation(prediction, number_predictions) for prediction in button_predictions
    )
    preselect_buttons = _preselect_button_observations(table)
    if screen.kind is ScreenKind.ACTIONABLE_TABLE:
        ambiguity_flags: list[str] = []
        if not button_observations:
            ambiguity_flags.append("missing_current_action_row")
        if preselect_buttons:
            ambiguity_flags.append("preselect_shortcut_label")
        if table is not None and table.buttons:
            if not _has_passive_or_start_action(table.buttons):
                ambiguity_flags.append("missing_passive_action")
            if _has_button_label_action_mismatch(table.buttons):
                ambiguity_flags.append("button_label_action_mismatch")
        panels = [
            ActionPanelObservation(
                panel_kind="current_action_row",
                visible=bool(button_observations),
                enabled=True if button_observations else None,
                hero_turn_indicator=screen.hero_turn,
                row_bbox=None,
                buttons=button_observations,
                confidence=screen.confidence,
                ambiguity_flags=tuple(ambiguity_flags),
                evidence=RoiEvidence(roi_id="action_panel:current_action_row"),
            ),
        ]
        if preselect_buttons:
            panels.append(
                ActionPanelObservation(
                    panel_kind="preselect_strip",
                    visible=True,
                    enabled=True,
                    hero_turn_indicator=screen.hero_turn,
                    row_bbox=None,
                    buttons=preselect_buttons,
                    confidence=min(button.confidence for button in preselect_buttons),
                    ambiguity_flags=("preselect_shortcut_label",),
                    evidence=RoiEvidence(roi_id="action_panel:preselect_strip"),
                )
            )
        return tuple(panels)
    return (
        ActionPanelObservation(
            panel_kind="unknown",
            visible=False,
            enabled=None,
            hero_turn_indicator=screen.hero_turn,
            row_bbox=None,
            buttons=button_observations,
            confidence=screen.confidence,
            ambiguity_flags=(screen.kind.value,),
            evidence=RoiEvidence(roi_id="action_panel:unknown"),
        ),
    )


def _preselect_button_observations(
    table: RecognizedTable | None,
) -> tuple[ButtonObservation, ...]:
    if table is None:
        return ()
    return tuple(
        _button_observation_from_recognized_button(button, source=table.source)
        for button in table.buttons
        if _is_preselect_or_shortcut_label(button.label)
    )


def _accepted_button_observations(
    table: RecognizedTable | None,
) -> tuple[ButtonObservation, ...]:
    if table is None:
        return ()
    return tuple(
        _button_observation_from_recognized_button(button, source=table.source)
        for button in table.buttons
        if not _is_preselect_or_shortcut_label(button.label)
    )


def _has_passive_or_start_action(buttons: tuple[RecognizedButton, ...]) -> bool:
    return any(button.action_type in {"check", "call", "bet"} for button in buttons)


def _has_button_label_action_mismatch(buttons: tuple[RecognizedButton, ...]) -> bool:
    return any(_button_label_action_mismatch(button) for button in buttons)


def _button_label_action_mismatch(button: RecognizedButton) -> bool:
    action_type = _action_type_from_button_label(button.label)
    return action_type is not None and button.action_type != action_type


def _action_type_from_button_label(label: str) -> str | None:
    normalized = " ".join(label.lower().replace("-", " ").split())
    if not normalized or _is_preselect_or_shortcut_label(normalized):
        return None
    if "check" in normalized:
        return "check"
    if "call" in normalized:
        return "call"
    if "fold" in normalized:
        return "fold"
    if "raise" in normalized:
        return "raise"
    if "bet" in normalized:
        return "bet"
    if "all in" in normalized:
        return "all_in"
    return None


def _button_observation(
    prediction: PokerLegendsButtonPrediction,
    number_predictions: tuple[PokerLegendsNumberPrediction, ...],
) -> ButtonObservation:
    evidence = RoiEvidence(roi_id=f"button:{prediction.slot}")
    action_candidates = (
        Candidate(
            value=prediction.action_type,
            confidence=prediction.confidence,
            source="image_button_recognizer",
            raw=prediction.to_dict(),
            evidence=(evidence,),
        ),
    ) if prediction.action_type is not None else ()
    amount = _number_prediction(number_predictions, "buttons", prediction.slot)
    amount_candidates = (
        Candidate(
            value=amount.normalized_number,
            confidence=amount.confidence,
            source="image_ocr",
            raw=amount.to_dict(),
            evidence=(evidence,),
        ),
    ) if amount is not None else ()
    return ButtonObservation(
        slot=prediction.slot,
        visible=prediction.visible,
        enabled=True if prediction.visible else None,
        action_candidates=action_candidates,
        accepted_action=prediction.action_type if prediction.visible else None,
        amount_candidates=amount_candidates,
        confidence=prediction.confidence,
        evidence=evidence,
    )


def _button_observation_from_recognized_button(
    button: RecognizedButton,
    *,
    source: str,
) -> ButtonObservation:
    evidence = RoiEvidence(roi_id=f"button:{button.command}")
    action_candidates = (
        Candidate(
            value=button.action_type,
            confidence=button.confidence,
            source=source,
            raw=asdict(button),
            evidence=(evidence,),
        ),
    ) if button.action_type is not None else ()
    return ButtonObservation(
        slot=button.command,
        visible=True,
        enabled=True,
        action_candidates=action_candidates,
        accepted_action=button.action_type,
        confidence=button.confidence,
        evidence=evidence,
    )


def _numeric_observations(
    predictions: tuple[PokerLegendsNumberPrediction, ...],
) -> tuple[NumericObservation, ...]:
    return tuple(_numeric_observation(prediction) for prediction in predictions)


def _numeric_observation(prediction: PokerLegendsNumberPrediction) -> NumericObservation:
    evidence = RoiEvidence(roi_id=f"number:{prediction.group}:{prediction.name}")
    candidate = (
        Candidate(
            value=prediction.normalized_number,
            confidence=prediction.confidence,
            source="image_ocr",
            raw=prediction.to_dict(),
            evidence=(evidence,),
        ),
    ) if prediction.normalized_number is not None else ()
    normalized = (
        str(prediction.normalized_number)
        if prediction.normalized_number is not None
        else None
    )
    return NumericObservation(
        role=_numeric_role(prediction.group, prediction.name),
        group=prediction.group,
        name=prediction.name,
        visible=prediction.visible,
        raw_text=prediction.raw,
        normalized_text=normalized,
        unit=None,
        scale=None,
        candidates=candidate,
        accepted_value=prediction.normalized_number,
        ocr_confidence=prediction.confidence,
        parse_confidence=prediction.confidence if prediction.normalized_number is not None else 0.0,
        value_confidence=prediction.confidence if prediction.normalized_number is not None else 0.0,
        parser_version="poker_legends_numbers_v1",
        format_flags=(),
        evidence=evidence,
    )


def _numeric_role(group: str, name: str) -> str:
    if group == "buttons":
        return "call_amount" if name == "primary_left" else "button_amount"
    if name in {"pot", "pot_size"}:
        return "pot"
    if name == "hero_stack":
        return "hero_stack"
    if name.endswith("_stack") or name == "opponent_stack":
        return "seat_stack"
    return name


def _seat_observations(
    table: RecognizedTable | None,
    *,
    controlled_seat: int,
) -> tuple[SeatObservation, ...]:
    if table is None:
        return ()
    return tuple(_seat_observation(seat, controlled_seat=controlled_seat) for seat in table.seats)


def _seat_observation(
    seat: RecognizedSeat,
    *,
    controlled_seat: int,
) -> SeatObservation:
    evidence = RoiEvidence(roi_id=f"seat:{seat.seat}")
    stack_candidate = (
        Candidate(
            value=seat.stack,
            confidence=seat.confidence,
            source="recognized_table",
            evidence=(evidence,),
        ),
    ) if seat.stack >= 0 else ()
    return SeatObservation(
        seat=seat.seat,
        occupied=True,
        in_hand=seat.active,
        has_hole_cards=bool(seat.hole_cards),
        folded=False if seat.active else None,
        all_in=True if seat.stack == 0 else False,
        sitting_out=False if seat.active else None,
        current_actor=seat.current,
        hero_seat=seat.seat == controlled_seat,
        dealer_button_nearby=seat.position in _BUTTON_LABELS if seat.position is not None else None,
        small_blind_marker=seat.position == "sb" if seat.position is not None else None,
        big_blind_marker=seat.position == "bb" if seat.position is not None else None,
        stack_candidates=stack_candidate,
        accepted_stack=seat.stack if seat.stack >= 0 else None,
        committed_current_street=seat.committed,
        committed_total_hand=seat.committed,
        showing_cards=tuple(
            _card_slot_from_recognized_card(
                "hero_hole_cards" if seat.seat == controlled_seat else f"seat_{seat.seat}_cards",
                card,
                source="recognized_table",
            )
            for card in seat.hole_cards
        ),
        confidence=seat.confidence,
        evidence=evidence,
    )


def _assembly_result_from_outcome(
    *,
    state: GameState | None,
    block_reason: str | None,
    screen: ScreenState,
    observation: VisualObservation,
    table_confidence: float,
    accepted_fields: tuple[AcceptedCriticalField, ...],
    source_policy_violations: tuple[SourcePolicyViolation, ...],
) -> GameStateAssemblyResult:
    issues = tuple(
        [
            *(
                _issue_from_source_policy_violation(violation)
                for violation in source_policy_violations
            ),
            *(
                ()
                if block_reason is None
                or (
                    source_policy_violations
                    and block_reason == TRUTH_ASSISTED_SOURCE_BLOCK_REASON
                )
                else (_issue_from_block_reason(block_reason),)
            ),
        ]
    )
    status = _assembly_status(
        state=state,
        screen=screen,
        block_reason=block_reason,
        source_policy_violations=source_policy_violations,
    )
    valid_for = _valid_contracts_for_state(state)
    contract_level = (
        ContractLevel.POLICY_DECISION if state is not None else ContractLevel.OBSERVE_ONLY
    )
    contract_status = "satisfied" if state is not None else "blocked"
    return GameStateAssemblyResult(
        status=status,
        validity_scope=ValidityScope.SINGLE_FRAME if state is not None else ValidityScope.NONE,
        state=state,
        contract_level=contract_level,
        contract_status=contract_status,
        valid_for=valid_for,
        issues=issues,
        freshness=Freshness(
            source_frame_id=observation.frame.frame_id,
            current_frame_revalidated=True,
            critical_fields_fresh=state is not None,
            action_row_fresh=state is not None,
            stable_frame_count=1 if state is not None else 0,
        ),
        field_confidences=_field_confidences(accepted_fields, table_confidence),
        critical_min_confidence=table_confidence if state is not None else None,
        layout_confidence=observation.layout.confidence,
        screen_confidence=screen.confidence,
        rule_consistency="consistent" if state is not None else "blocked",
        observation_id=observation.frame.frame_id,
    )


def _assembly_status(
    *,
    state: GameState | None,
    screen: ScreenState,
    block_reason: str | None,
    source_policy_violations: tuple[SourcePolicyViolation, ...],
) -> AssemblyStatus:
    if source_policy_violations:
        return AssemblyStatus.INVALID
    if screen.kind is not ScreenKind.ACTIONABLE_TABLE:
        return AssemblyStatus.BLOCKED_SCREEN
    if state is not None:
        return AssemblyStatus.SINGLE_FRAME_VALID
    if block_reason in {"board_count_mismatch", "low_card_confidence"}:
        return AssemblyStatus.INVALID
    return AssemblyStatus.NO_STATE


def _valid_contracts_for_state(state: GameState | None) -> tuple[ContractLevel, ...]:
    if state is None:
        return (ContractLevel.OBSERVE_ONLY,)
    valid = [ContractLevel.OBSERVE_ONLY, ContractLevel.GAME_STATE, ContractLevel.POLICY_DECISION]
    action_types = {action.action_type for action in state.legal_actions}
    if action_types <= {ActionType.FOLD, ActionType.CHECK}:
        valid.append(ContractLevel.FOLD_CHECK_ONLY)
    if ActionType.CALL in action_types:
        valid.append(ContractLevel.CALL_DECISION)
    if ActionType.BET in action_types or ActionType.RAISE in action_types:
        valid.append(ContractLevel.SIZING_DECISION)
    return tuple(valid)


def _field_confidences(
    accepted_fields: tuple[AcceptedCriticalField, ...],
    table_confidence: float,
) -> dict[str, float]:
    return {field.field_path: table_confidence for field in accepted_fields}


def _issue_from_source_policy_violation(violation: SourcePolicyViolation) -> AssemblyIssue:
    return AssemblyIssue(
        issue_type="source_policy",
        reason_code="TRUTH_ASSISTED_FIELD_IN_IMAGE_ONLY_MODE",
        field_path=violation.field_path,
        rule_name="image_only_source_policy",
        severity="hard",
        blocking=True,
        required_by_contract=(ContractLevel.POLICY_DECISION,),
        source=violation.source,
        message=violation.reason,
    )


def _issue_from_block_reason(reason: str) -> AssemblyIssue:
    reason_code, field_path, issue_type = _reason_code_field_and_type(reason)
    return AssemblyIssue(
        issue_type=issue_type,
        reason_code=reason_code,
        field_path=field_path,
        rule_name="single_frame_contract",
        severity="hard",
        blocking=True,
        required_by_contract=(ContractLevel.POLICY_DECISION,),
        message=reason,
    )


def _reason_code_field_and_type(reason: str) -> tuple[str, str, str]:
    mapping = {
        "screen_not_actionable": ("SCREEN_NOT_ACTIONABLE", "screen.kind", "missing"),
        "missing_image_path": ("MISSING_IMAGE", "frame.image", "missing"),
        "missing_layout_annotation": ("LAYOUT_LOW_CONFIDENCE", "layout", "missing"),
        "missing_table_metadata": ("MISSING_TABLE_METADATA", "table", "missing"),
        "hero_not_current": ("HERO_TURN_NOT_CONFIRMED", "screen.hero_turn", "missing"),
        "missing_pot": ("POT_REQUIRED_BY_POLICY", "numbers.pot", "missing"),
        "missing_seats": ("MISSING_SEATS", "seats", "missing"),
        "missing_hero_seat": ("MISSING_HERO_SEAT", "seats.hero", "missing"),
        "missing_hero_stack": ("MISSING_HERO_STACK", "numbers.hero_stack", "missing"),
        "missing_hero_hole_cards": ("MISSING_HERO_CARDS", "cards.hero", "missing"),
        "low_card_confidence": ("CARD_LOW_CONFIDENCE", "cards", "low_confidence"),
        "missing_call_amount": ("CALL_AMOUNT_UNTRUSTED", "numbers.call_amount", "missing"),
        "missing_legal_actions": ("ACTION_ROW_UNSTABLE", "actions", "missing"),
        "missing_current_action_row": ("ACTION_ROW_UNSTABLE", "actions.current_row", "missing"),
        "missing_passive_action": ("ACTION_ROW_UNSTABLE", "actions.passive_action", "missing"),
        "missing_call_action": ("ACTION_ROW_UNSTABLE", "actions.call", "missing"),
        "preselect_ambiguous": ("PRESELECT_AMBIGUOUS", "actions", "ambiguous"),
        "missing_street": ("MISSING_STREET", "street", "missing"),
        "board_count_mismatch": ("MISSING_BOARD_CARD", "cards.board", "contradiction"),
        "missing_seat_stack": ("MISSING_SEAT_STACK", "numbers.seat_stack", "missing"),
        "not_enough_players": ("NOT_ENOUGH_PLAYERS", "seats", "missing"),
        "low_button_confidence": ("ACTION_ROW_UNSTABLE", "actions.buttons", "low_confidence"),
        TRUTH_ASSISTED_SOURCE_BLOCK_REASON: (
            "TRUTH_ASSISTED_FIELD_IN_IMAGE_ONLY_MODE",
            "source_policy",
            "source_policy",
        ),
    }
    return mapping.get(reason, (reason.upper(), "unknown", "missing"))


def _frame_id_from_inputs(
    annotation: Mapping[str, object] | None,
    image_path: Path | None,
) -> str:
    if annotation is None:
        return image_path.stem if image_path is not None else "unknown"
    return str(annotation.get("frame_id") or "unknown")


def _screen_source(context: _FrameContext) -> str:
    if context.annotation is not None:
        return REVIEWED_TRUTH_SOURCE
    return "image_screen_state"


def _accepted_critical_fields_for_state(
    *,
    state: GameState | None,
    table: RecognizedTable,
    annotation: Mapping[str, object] | None,
    screen_source: str,
    annotation_source: str,
    card_predictions: tuple[PokerLegendsCardConsensusPrediction, ...],
    button_predictions: tuple[PokerLegendsButtonPrediction, ...],
    number_predictions: tuple[PokerLegendsNumberPrediction, ...],
    controlled_seat: int,
) -> tuple[AcceptedCriticalField, ...]:
    if state is None:
        return ()
    hero = state.player(controlled_seat)
    fields = [
        AcceptedCriticalField("screen.actionability", screen_source, table.current_seat),
        AcceptedCriticalField(
            "cards.hero",
            _card_source(card_predictions, table.source),
            tuple(card.code for card in hero.hole_cards),
        ),
        AcceptedCriticalField(
            "cards.board",
            _card_source(card_predictions, table.source),
            tuple(card.code for card in state.board),
        ),
        AcceptedCriticalField(
            "numbers.pot",
            _pot_source(annotation, number_predictions, annotation_source=annotation_source),
            state.pots[0].amount if state.pots else None,
        ),
        AcceptedCriticalField(
            "numbers.hero_stack",
            _hero_stack_source(
                annotation,
                number_predictions,
                annotation_source=annotation_source,
            ),
            hero.stack,
        ),
        AcceptedCriticalField(
            "actions.legal_labels",
            _button_source(annotation, button_predictions, annotation_source=annotation_source),
            tuple(action.action_type.value for action in state.legal_actions),
        ),
    ]
    if state.to_call > 0 or any(
        action.action_type is ActionType.CALL for action in state.legal_actions
    ):
        fields.append(
            AcceptedCriticalField(
                "numbers.call_amount",
                _call_amount_source(
                    annotation,
                    number_predictions,
                    annotation_source=annotation_source,
                    hero_committed=hero.committed,
                ),
                state.to_call,
            )
        )
    return tuple(fields)


def _card_source(
    card_predictions: tuple[PokerLegendsCardConsensusPrediction, ...],
    table_source: str,
) -> str:
    if card_predictions:
        return "image_card_consensus"
    if table_source == "poker_legends_llm":
        return "vlm_annotation"
    return "unknown"


def _pot_source(
    annotation: Mapping[str, object] | None,
    number_predictions: tuple[PokerLegendsNumberPrediction, ...],
    *,
    annotation_source: str,
) -> str:
    if _pot_from_annotation(annotation) is not None:
        return annotation_source
    if _number_prediction_value(number_predictions, "texts", "pot") is not None:
        return "image_ocr"
    if _pot_from_trusted_committed(annotation) is not None:
        return "rule_inferred_committed"
    return "rule_or_default"


def _hero_stack_source(
    annotation: Mapping[str, object] | None,
    number_predictions: tuple[PokerLegendsNumberPrediction, ...],
    *,
    annotation_source: str,
) -> str:
    if (
        _annotation_hero_stack(annotation) is not None
        or _hero_stack_from_texts(annotation) is not None
    ):
        return annotation_source
    if _number_prediction_value(number_predictions, "texts", "hero_stack") is not None:
        return "image_ocr"
    return "unknown"


def _button_source(
    annotation: Mapping[str, object] | None,
    button_predictions: tuple[PokerLegendsButtonPrediction, ...],
    *,
    annotation_source: str,
) -> str:
    if annotation is not None and _truth_action_buttons(annotation):
        return annotation_source
    if button_predictions:
        return "image_button_recognizer"
    return "unknown"


def _call_amount_source(
    annotation: Mapping[str, object] | None,
    number_predictions: tuple[PokerLegendsNumberPrediction, ...],
    *,
    annotation_source: str,
    hero_committed: int,
) -> str:
    if annotation is not None and _truth_call_amount(annotation) is not None:
        return annotation_source
    if _has_button_number_prediction(number_predictions):
        return "image_ocr"
    if _call_amount_from_trusted_committed(annotation, hero_committed=hero_committed) is not None:
        return "rule_inferred_committed"
    return _button_source(annotation, (), annotation_source=annotation_source)


def _read_json_object(path: str | Path) -> dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, object], data)


def _llm_cards(raw: object) -> tuple[RecognizedCard, ...]:
    cards: list[RecognizedCard] = []
    if isinstance(raw, list):
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                continue
            code = item.get("card")
            card = code if isinstance(code, str) and code else None
            cards.append(
                RecognizedCard(
                    slot=str(item.get("slot") or f"card_{index}"),
                    card=card,
                    visible=bool(item.get("visible", card is not None)),
                    confidence=_to_float(item.get("confidence"), default=1.0),
                )
            )
    return tuple(cards)


def _llm_hero_current(annotation: Mapping[str, object], *, controlled_name: str) -> bool:
    for item in _mapping_sequence(annotation.get("seats")):
        if str(item.get("name") or "").lower() == controlled_name:
            return _optional_bool(item.get("current")) is True
    return False


def _llm_screen_state(
    table_state: Mapping[str, object], *, hero_turn: bool, confidence: float
) -> ScreenState:
    if not bool(table_state.get("is_table")):
        return ScreenState.non_table_ui(confidence=confidence, reason="llm_non_table")
    blocking = table_state.get("blocking_reason")
    if isinstance(blocking, str) and blocking not in {"", "none"}:
        return ScreenState.blocked_overlay(
            blocking_reason=blocking, confidence=confidence, reason="llm_overlay"
        )
    if bool(table_state.get("is_actionable")):
        return ScreenState.actionable_table(
            confidence=confidence, hero_turn=hero_turn, reason="llm_actionable"
        )
    return ScreenState.table_observe(confidence=confidence, reason="llm_observe")


def _recognized_table_from_predictions(
    *,
    source: str,
    image: str,
    frame_id: str,
    screen: ScreenState,
    annotation: Mapping[str, object] | None,
    card_predictions: tuple[PokerLegendsCardConsensusPrediction, ...],
    button_predictions: tuple[PokerLegendsButtonPrediction, ...],
    number_predictions: tuple[PokerLegendsNumberPrediction, ...],
    controlled_seat: int,
) -> RecognizedTable:
    board = tuple(
        RecognizedCard(
            slot=prediction.slot,
            card=prediction.card,
            visible=prediction.visible,
            confidence=prediction.confidence,
        )
        for prediction in card_predictions
        if prediction.group == "board"
    )
    hero_hole_cards = tuple(
        RecognizedCard(
            slot=prediction.slot,
            card=prediction.card,
            visible=prediction.visible,
            confidence=prediction.confidence,
        )
        for prediction in card_predictions
        if prediction.group == "hero_hole_cards"
    )
    buttons = _recognized_buttons_from_predictions(
        annotation,
        button_predictions=button_predictions,
        number_predictions=number_predictions,
    )
    seats = _recognized_seats_from_annotation(
        annotation,
        controlled_seat=controlled_seat,
        hero_hole_cards=hero_hole_cards,
        number_predictions=number_predictions,
        hero_current=screen.hero_turn is True,
    )
    confidences = [
        screen.confidence,
        *(card.confidence for card in board if card.visible),
        *(card.confidence for card in hero_hole_cards if card.visible),
        *(button.confidence for button in buttons),
        *(
            prediction.confidence
            for prediction in number_predictions
            if prediction.visible and prediction.normalized_number is not None
        ),
    ]
    return RecognizedTable(
        source=source,
        image=image,
        hand_id=frame_id,
        street=_street_name_from_annotation(annotation, board),
        current_seat=controlled_seat if screen.hero_turn is True else _current_seat(seats),
        pot=_first_number(
            _pot_from_annotation(annotation),
            _number_prediction_value(number_predictions, "texts", "pot"),
            _pot_from_trusted_committed(annotation),
        ),
        board=board,
        seats=seats,
        buttons=buttons,
        confidence=min(confidences) if confidences else screen.confidence,
    )


def _recognized_seats_from_annotation(
    annotation: Mapping[str, object] | None,
    *,
    controlled_seat: int,
    hero_hole_cards: tuple[RecognizedCard, ...],
    number_predictions: tuple[PokerLegendsNumberPrediction, ...],
    hero_current: bool,
) -> tuple[RecognizedSeat, ...]:
    raw_seats = _mapping_sequence(None if annotation is None else annotation.get("seats"))
    seat_numbers = _seat_numbers(raw_seats, controlled_seat=controlled_seat)
    seats: list[RecognizedSeat] = []
    for item in raw_seats:
        name = str(item.get("name") or "")
        seat_number = seat_numbers.get(name)
        if seat_number is None:
            continue
        stack = _optional_int(item.get("stack"))
        if stack is None and seat_number == controlled_seat:
            stack = _number_prediction_value(number_predictions, "texts", "hero_stack")
        active = _optional_bool(item.get("active"))
        current = _optional_bool(item.get("current"))
        seats.append(
            RecognizedSeat(
                seat=seat_number,
                stack=-1 if stack is None else stack,
                committed=_optional_int(item.get("committed")) or 0,
                active=True if active is None else active,
                current=False if current is None else current,
                hole_cards=hero_hole_cards if seat_number == controlled_seat else (),
                confidence=_to_float(item.get("confidence"), default=0.0),
                position=_normalize_position(item.get("position")),
            )
        )
    if _seat_by_number(tuple(seats), controlled_seat) is None and hero_hole_cards:
        stack = _first_number(
            _hero_stack_from_texts(annotation),
            _number_prediction_value(number_predictions, "texts", "hero_stack"),
        )
        if stack is not None:
            seats.append(
                RecognizedSeat(
                    seat=controlled_seat,
                    stack=stack,
                    committed=0,
                    active=True,
                    current=hero_current,
                    hole_cards=hero_hole_cards,
                    confidence=1.0,
                )
            )
    if len(seats) == 1:
        opponent_stack = _first_number(
            _opponent_stack_from_texts(annotation),
            _number_prediction_value(number_predictions, "texts", "right_top_stack"),
            _number_prediction_value(number_predictions, "texts", "opponent_stack"),
        )
        if opponent_stack is not None:
            seats.append(
                RecognizedSeat(
                    seat=_next_available_seat(seats, controlled_seat=controlled_seat),
                    stack=opponent_stack,
                    committed=0,
                    active=True,
                    current=False,
                    hole_cards=(),
                    confidence=0.75,
                )
            )
    return tuple(sorted(seats, key=lambda seat: seat.seat))


def _seat_numbers(
    seats: list[Mapping[str, object]],
    *,
    controlled_seat: int,
) -> dict[str, int]:
    numbers: dict[str, int] = {}
    next_seat = 0
    for item in seats:
        name = str(item.get("name") or "")
        if not name:
            continue
        if name.lower() == "hero":
            numbers[name] = controlled_seat
            next_seat = max(next_seat, controlled_seat + 1)
            continue
        while next_seat == controlled_seat:
            next_seat += 1
        numbers[name] = next_seat
        next_seat += 1
    return numbers


def _next_available_seat(
    seats: list[RecognizedSeat],
    *,
    controlled_seat: int,
) -> int:
    used = {seat.seat for seat in seats}
    seat = 0
    while seat in used or seat == controlled_seat:
        seat += 1
    return seat


def _recognized_table_to_dict(table: RecognizedTable) -> dict[str, object]:
    return {
        "source": table.source,
        "image": table.image,
        "hand_id": table.hand_id,
        "street": table.street,
        "current_seat": table.current_seat,
        "pot": table.pot,
        "board": [asdict(card) for card in table.board],
        "seats": [asdict(seat) for seat in table.seats],
        "buttons": [asdict(button) for button in table.buttons],
        "confidence": table.confidence,
    }


def _recognized_buttons_from_predictions(
    annotation: Mapping[str, object] | None,
    *,
    button_predictions: tuple[PokerLegendsButtonPrediction, ...],
    number_predictions: tuple[PokerLegendsNumberPrediction, ...],
) -> tuple[RecognizedButton, ...]:
    truth_buttons = _truth_action_buttons(annotation)
    if truth_buttons and not _has_visible_primary_preselect_or_shortcut_button(annotation):
        return truth_buttons
    buttons: list[RecognizedButton] = []
    for prediction in button_predictions:
        if not prediction.visible or prediction.action_type is None:
            continue
        truth_visible = _truth_button_visible(annotation, prediction.slot)
        if truth_visible is False and prediction.slot != "primary_left":
            continue
        label = (
            _truth_button_label(annotation, prediction.slot)
            or _number_prediction_raw(number_predictions, "buttons", prediction.slot)
            or prediction.action_type
        )
        action_type = _action_type_from_button_label(label) or prediction.action_type
        buttons.append(
            RecognizedButton(
                label=label,
                action_type=action_type,
                command=prediction.slot,
                confidence=prediction.confidence,
            )
        )
    if buttons:
        return tuple(buttons)
    return ()


def _street_name_from_annotation(
    annotation: Mapping[str, object] | None,
    board: tuple[RecognizedCard, ...],
) -> str:
    visible_board_count = len(tuple(card for card in board if card.visible and card.card))
    table_state = annotation.get("table_state") if annotation is not None else None
    if isinstance(table_state, Mapping):
        raw = table_state.get("street")
        if isinstance(raw, str) and raw in _STREET_MAP:
            expected_count = _expected_board_count_for_street_name(raw)
            if expected_count is None or expected_count == visible_board_count:
                return raw
            inferred = _street_name_from_board_count(visible_board_count)
            if inferred is not None:
                return inferred
            return raw
    return _street_name_from_board_count(visible_board_count) or "unknown"


def _street_name_from_board_count(visible_board_count: int) -> str | None:
    inferred_streets: dict[int, str] = {
        0: Street.PREFLOP.value,
        3: Street.FLOP.value,
        4: Street.TURN.value,
        5: Street.RIVER.value,
    }
    return inferred_streets.get(visible_board_count)


def _expected_board_count_for_street_name(street: str) -> int | None:
    parsed = _STREET_MAP.get(street)
    if parsed is None:
        return None
    return _expected_board_count(parsed)


def _street_from_table(table: RecognizedTable) -> Street | None:
    return _STREET_MAP.get(table.street)


def _expected_board_count(street: Street) -> int | None:
    if street is Street.PREFLOP:
        return 0
    if street is Street.FLOP:
        return 3
    if street is Street.TURN:
        return 4
    if street is Street.RIVER:
        return 5
    return None


def _cards_from_recognized_cards(cards: tuple[RecognizedCard, ...]) -> tuple[Card, ...]:
    return tuple(Card.from_code(card.card) for card in cards if card.visible and card.card)


def _seat_by_number(
    seats: tuple[RecognizedSeat, ...],
    seat_number: int,
) -> RecognizedSeat | None:
    for seat in seats:
        if seat.seat == seat_number:
            return seat
    return None


def _current_seat(seats: tuple[RecognizedSeat, ...]) -> int | None:
    for seat in seats:
        if seat.current:
            return seat.seat
    return None


#: Position labels that mark the dealer button (avoid ambiguous single letters
#: like "b", which collides with the big blind).
_BUTTON_LABELS = frozenset({"button", "btn", "bu", "dealer", "dlr", "d"})


def _normalize_position(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    return text or None


def _resolve_button_seat(
    seats: tuple[RecognizedSeat, ...],
    controlled_seat: int,
) -> tuple[int, str]:
    """Find the dealer-button seat, or fall back so hero is treated as out of position.

    Only "is hero on the button" changes the heuristic's position handling, so when
    the dealer button is not recognized we deliberately place it on another seat:
    that makes the bot play the tighter, safer out-of-position ranges instead of
    over-opening as if it were always on the button (the old hardcoded behaviour).
    """
    for seat in seats:
        if seat.position in _BUTTON_LABELS:
            return seat.seat, "recognized"
    for seat in seats:
        if seat.seat != controlled_seat and seat.active:
            return seat.seat, "default_oop"
    for seat in seats:
        if seat.seat != controlled_seat:
            return seat.seat, "default_oop"
    return controlled_seat, "default_hero"


def _blinds_from_annotation(
    annotation: Mapping[str, object],
    *,
    default_small: int,
    default_big: int,
) -> tuple[int, int]:
    """Read blinds from the annotation when present, else fall back to the config."""
    small, big = default_small, default_big
    table_state = annotation.get("table_state")
    # table_state is the authoritative table read, so apply it LAST (it wins over a
    # top-level blind field).
    sources: tuple[Mapping[str, object], ...] = (annotation,) + (
        (cast(Mapping[str, object], table_state),) if isinstance(table_state, Mapping) else ()
    )
    for source in sources:
        small_value = _optional_int(source.get("small_blind"))
        big_value = _optional_int(source.get("big_blind"))
        if small_value is not None and small_value >= 0:
            small = small_value
        if big_value is not None and big_value > 0:
            big = big_value
    return small, big


def _pot_from_annotation(annotation: Mapping[str, object] | None) -> int | None:
    for text in _mapping_sequence(None if annotation is None else annotation.get("texts")):
        if str(text.get("name") or "") in {"pot", "pot_size"} and bool(text.get("visible", True)):
            value = _optional_int(text.get("normalized_number"))
            if value is not None:
                return value
    return None


def _pot_from_trusted_committed(annotation: Mapping[str, object] | None) -> int | None:
    active_committed: list[int] = []
    for seat in _mapping_sequence(None if annotation is None else annotation.get("seats")):
        if _optional_bool(seat.get("active")) is False:
            continue
        committed = _optional_int(seat.get("committed"))
        if committed is None or committed < 0:
            return None
        active_committed.append(committed)
    if len(active_committed) < 2:
        return None
    total = sum(active_committed)
    return total if total > 0 else None


def _hero_stack_from_texts(annotation: Mapping[str, object] | None) -> int | None:
    for text in _mapping_sequence(None if annotation is None else annotation.get("texts")):
        if str(text.get("name") or "") == "hero_stack" and bool(text.get("visible", True)):
            value = _optional_int(text.get("normalized_number"))
            if value is not None:
                return value
    return None


def _opponent_stack_from_texts(annotation: Mapping[str, object] | None) -> int | None:
    for text in _mapping_sequence(None if annotation is None else annotation.get("texts")):
        if str(text.get("name") or "") in {"right_top_stack", "opponent_stack"} and bool(
            text.get("visible", True)
        ):
            value = _optional_int(text.get("normalized_number"))
            if value is not None:
                return value
    return None


def _truth_button(
    annotation: Mapping[str, object] | None,
    slot: str,
) -> Mapping[str, object] | None:
    for button in _mapping_sequence(None if annotation is None else annotation.get("buttons")):
        if str(button.get("name") or "") == slot:
            return button
    return None


def _truth_button_label(annotation: Mapping[str, object] | None, slot: str) -> str | None:
    button = _truth_button(annotation, slot)
    if button is None:
        return None
    label = button.get("label")
    if isinstance(label, str) and label.strip() and bool(button.get("visible", True)):
        return label
    return None


def _truth_button_visible(annotation: Mapping[str, object] | None, slot: str) -> bool | None:
    button = _truth_button(annotation, slot)
    if button is None:
        return None
    return bool(button.get("visible", True))


def _truth_action_buttons(annotation: Mapping[str, object] | None) -> tuple[RecognizedButton, ...]:
    buttons: list[RecognizedButton] = []
    for button in _mapping_sequence(None if annotation is None else annotation.get("buttons")):
        action_type = str(button.get("action_type") or "")
        if action_type not in _ACTION_MAP:
            continue
        if not bool(button.get("visible", True)):
            continue
        name = str(button.get("name") or "")
        label = button.get("label")
        if not isinstance(label, str) or not label.strip():
            label = action_type
        if not _is_direct_truth_action_button(name, label):
            continue
        action_type = _action_type_from_button_label(label) or action_type
        buttons.append(
            RecognizedButton(
                label=label,
                action_type=action_type,
                command=name or action_type,
                confidence=_to_float(button.get("confidence"), default=0.75),
            )
        )
    return tuple(buttons)


def _is_direct_truth_action_button(name: str, label: str) -> bool:
    normalized_name = name.lower()
    normalized_label = label.lower()
    if "any" in normalized_label or "check/fold" in normalized_label:
        return False
    if normalized_name.startswith("raise_shortcut"):
        return False
    if normalized_name in {
        "primary_left",
        "primary_middle",
        "primary_right",
        "check",
        "call",
        "raise",
        "fold",
    }:
        return True
    if normalized_name.endswith("_button"):
        prefix = normalized_name.removesuffix("_button")
        return prefix in {"check", "call", "raise", "fold"}
    return False


def _button_amount(
    annotation: Mapping[str, object],
    slot: str,
    number_predictions: tuple[PokerLegendsNumberPrediction, ...],
    *,
    hero_committed: int,
) -> int | None:
    label = _truth_button_label(annotation, slot)
    if label is not None:
        amount = _button_amount_from_label(label)
        if amount is not None:
            return amount
        if "any" in label.lower():
            return None
    amount = _truth_call_amount(annotation)
    if amount is not None:
        return amount
    amount = _number_prediction_value(number_predictions, "buttons", slot)
    if amount is not None:
        return amount
    return _call_amount_from_trusted_committed(annotation, hero_committed=hero_committed)


def _button_amount_from_label(label: str) -> int | None:
    if _is_preselect_or_shortcut_label(label):
        return None
    normalized = label.lower()
    if "check" in normalized:
        return 0
    return parse_poker_legends_chip_amount(label)


def _is_preselect_or_shortcut_label(label: str) -> bool:
    normalized = " ".join(label.lower().replace("/", " / ").split())
    return (
        "any" in normalized
        or "check / fold" in normalized
        or "check/fold" in normalized
        or "fold to" in normalized
    )


def _has_visible_primary_preselect_or_shortcut_button(
    annotation: Mapping[str, object] | None,
) -> bool:
    for button in _mapping_sequence(None if annotation is None else annotation.get("buttons")):
        if not bool(button.get("visible", True)):
            continue
        name = str(button.get("name") or "").lower()
        if name not in {"primary_left", "primary_middle", "primary_right"}:
            continue
        label = button.get("label")
        if isinstance(label, str) and _is_preselect_or_shortcut_label(label):
            return True
    return False


def _truth_call_amount(annotation: Mapping[str, object]) -> int | None:
    for button in _mapping_sequence(annotation.get("buttons")):
        if not bool(button.get("visible", True)):
            continue
        label = button.get("label")
        if not isinstance(label, str):
            continue
        if "call" not in label.lower() or "any" in label.lower():
            continue
        amount = _button_amount_from_label(label)
        if amount is not None:
            return amount
    for text in _mapping_sequence(annotation.get("texts")):
        if str(text.get("name") or "") != "top_action_banner" or not bool(
            text.get("visible", True)
        ):
            continue
        value = text.get("value")
        if not isinstance(value, str) or "call" not in value.lower():
            continue
        amount = _button_amount_from_label(value)
        if amount is not None:
            return amount
    return None


def _number_roi_names_for_fallbacks(
    annotation: Mapping[str, object] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    text_names: list[str] = []
    if _pot_from_annotation(annotation) is None:
        text_names.append("pot")
    if _annotation_hero_stack(annotation) is None and _hero_stack_from_texts(annotation) is None:
        text_names.append("hero_stack")
    if _truth_active_seat_count(annotation) < 2 and _opponent_stack_from_texts(annotation) is None:
        text_names.append("right_top_stack")

    button_names: list[str] = []
    for button in _mapping_sequence(None if annotation is None else annotation.get("buttons")):
        if not bool(button.get("visible", True)):
            continue
        if str(button.get("action_type") or "") != "call":
            continue
        label = button.get("label")
        if isinstance(label, str) and "any" in label.lower():
            continue
        if not isinstance(label, str) or _button_amount_from_label(label) is None:
            name = str(button.get("name") or "")
            if name:
                button_names.append(name)
    if annotation is None:
        button_names.append("primary_left")
    elif not button_names and not _has_truth_buttons(annotation):
        button_names.append("primary_left")
    return tuple(dict.fromkeys(text_names)), tuple(dict.fromkeys(button_names))


def _truth_active_seat_count(annotation: Mapping[str, object] | None) -> int:
    count = 0
    for seat in _mapping_sequence(None if annotation is None else annotation.get("seats")):
        if _optional_bool(seat.get("active")) is not False:
            count += 1
    return count


def _annotation_hero_stack(annotation: Mapping[str, object] | None) -> int | None:
    for seat in _mapping_sequence(None if annotation is None else annotation.get("seats")):
        if str(seat.get("name") or "").lower() == "hero":
            return _optional_int(seat.get("stack"))
    return None


def _has_truth_buttons(annotation: Mapping[str, object]) -> bool:
    return any(
        bool(button.get("visible", True)) for button in _mapping_sequence(annotation.get("buttons"))
    )


def _number_prediction_value(
    predictions: tuple[PokerLegendsNumberPrediction, ...],
    group: str,
    name: str,
) -> int | None:
    prediction = _number_prediction(predictions, group, name)
    if prediction is None:
        return None
    return prediction.normalized_number


def _has_button_number_prediction(
    predictions: tuple[PokerLegendsNumberPrediction, ...],
) -> bool:
    return any(
        prediction.group == "buttons" and prediction.normalized_number is not None
        for prediction in predictions
    )


def _number_prediction_raw(
    predictions: tuple[PokerLegendsNumberPrediction, ...],
    group: str,
    name: str,
) -> str | None:
    prediction = _number_prediction(predictions, group, name)
    if prediction is None or not prediction.raw.strip():
        return None
    return prediction.raw


def _number_prediction(
    predictions: tuple[PokerLegendsNumberPrediction, ...],
    group: str,
    name: str,
) -> PokerLegendsNumberPrediction | None:
    for prediction in predictions:
        if (
            prediction.group == group
            and prediction.name == name
            and prediction.visible
            and prediction.normalized_number is not None
        ):
            return prediction
    return None


def _first_number(*values: int | None) -> int | None:
    for value in values:
        if value is not None:
            return value
    return None


def _call_amount_from_trusted_committed(
    annotation: Mapping[str, object] | None,
    *,
    hero_committed: int,
) -> int | None:
    committed_values: list[int] = []
    for seat in _mapping_sequence(None if annotation is None else annotation.get("seats")):
        if _optional_bool(seat.get("active")) is False:
            continue
        committed = _optional_int(seat.get("committed"))
        if committed is None or committed < 0:
            return None
        committed_values.append(committed)
    if len(committed_values) < 2:
        return None
    amount = max(committed_values) - hero_committed
    return amount if amount > 0 else None


def _mapping_sequence(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [cast(Mapping[str, object], item) for item in value if isinstance(item, Mapping)]


def _optional_int(value: object) -> int | None:
    if type(value) is int:
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip():
        digits = "".join(char for char in value if char.isdigit())
        return int(digits) if digits else None
    return None


def _optional_bool(value: object) -> bool | None:
    if type(value) is bool:
        return value
    return None


def _to_float(value: object, *, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        return float(value)
    return default
