"""Poker Legends recognition adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, cast

from holdem_common import Action, ActionType, Card, GameState, PlayerState, Pot, Street

from holdem_bot.capture import CapturedFrame
from holdem_bot.recognize import RecognitionResult, Recognizer
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
        metadata: dict[str, object] = {
            "source": "poker_legends_llm",
            "screen_kind": screen.kind.value,
            "recognized_table": _recognized_table_to_dict(table),
        }
        if screen.kind is not ScreenKind.ACTIONABLE_TABLE:
            return RecognitionResult(
                state=None, confidence=screen.confidence, metadata=metadata, screen=screen
            )
        state, block_reason = self._state_from_table(
            table,
            annotation=annotation,
            screen=screen,
            card_predictions=(),
            number_predictions=(),
        )
        if block_reason is not None:
            metadata["state_block_reason"] = block_reason
        return RecognitionResult(
            state=state, confidence=table.confidence, metadata=metadata, screen=screen
        )

    def recognize(self, frame: CapturedFrame) -> RecognitionResult:
        context = _FrameContext(
            annotation=self._annotation_from_frame(frame),
            image_path=self._image_path_from_frame(frame),
            layout_annotation=self._layout_annotation_from_frame(frame),
        )
        screen = self._screen_state(frame, context)
        metadata: dict[str, object] = {
            "source": frame.source,
            "screen_kind": screen.kind.value,
        }
        if context.annotation is not None:
            metadata["frame_id"] = context.frame_id
        if context.image_path is not None:
            metadata["image"] = str(context.image_path)

        if screen.kind is not ScreenKind.ACTIONABLE_TABLE:
            return RecognitionResult(
                state=None,
                confidence=screen.confidence,
                metadata=metadata,
                screen=screen,
            )
        if context.image_path is None:
            metadata["state_block_reason"] = "missing_image_path"
            return RecognitionResult(
                state=None,
                confidence=0.0,
                metadata=metadata,
                screen=screen,
            )
        if context.layout_annotation is None:
            metadata["state_block_reason"] = "missing_layout_annotation"
            return RecognitionResult(
                state=None,
                confidence=0.0,
                metadata=metadata,
                screen=screen,
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
        return RecognitionResult(
            state=state,
            confidence=table.confidence,
            metadata=metadata,
            screen=screen,
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
                amount = _button_amount(annotation, button.command, number_predictions)
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
        return tuple(actions), to_call, None


class _FrameContext:
    def __init__(
        self,
        *,
        annotation: Mapping[str, object] | None,
        image_path: Path | None,
        layout_annotation: Mapping[str, object] | None,
    ) -> None:
        self.annotation = annotation
        self.image_path = image_path
        self.layout_annotation = layout_annotation

    @property
    def frame_id(self) -> str:
        if self.annotation is None:
            return "unknown"
        return str(self.annotation.get("frame_id") or "unknown")


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
        opponent_stack = _opponent_stack_from_texts(annotation)
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
    buttons = tuple(
        RecognizedButton(
            label=_truth_button_label(annotation, prediction.slot)
            or _number_prediction_raw(number_predictions, "buttons", prediction.slot)
            or (prediction.action_type or ""),
            action_type=prediction.action_type,
            command=prediction.slot,
            confidence=prediction.confidence,
        )
        for prediction in button_predictions
        if prediction.visible and prediction.action_type is not None
    )
    if buttons:
        return buttons
    return _truth_action_buttons(annotation)


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
    return _number_prediction_value(number_predictions, "buttons", slot)


def _button_amount_from_label(label: str) -> int | None:
    normalized = label.lower()
    if "any" in normalized:
        return None
    if "check" in normalized:
        return 0
    return parse_poker_legends_chip_amount(label)


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
