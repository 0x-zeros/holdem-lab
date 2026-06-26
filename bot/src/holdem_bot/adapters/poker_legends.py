"""Poker Legends recognition adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Protocol, cast

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
        controlled_seat: int = 0,
        small_blind: int = 5,
        big_blind: int = 10,
        min_card_confidence: float = 0.20,
        min_button_confidence: float = 0.20,
    ) -> None:
        if controlled_seat < 0:
            raise ValueError("controlled seat cannot be negative")
        if small_blind < 0 or big_blind <= 0:
            raise ValueError("blind amounts must be non-negative with a positive big blind")
        self.card_recognizer = card_recognizer
        self.button_recognizer = button_recognizer
        self.controlled_seat = controlled_seat
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.min_card_confidence = min_card_confidence
        self.min_button_confidence = min_button_confidence

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
            controlled_seat=controlled_seat,
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
        table = _recognized_table_from_predictions(
            source=frame.source,
            image=str(context.image_path),
            frame_id=context.frame_id,
            screen=screen,
            annotation=context.annotation,
            card_predictions=card_predictions,
            button_predictions=button_predictions,
            controlled_seat=self.controlled_seat,
        )
        state, block_reason = self._state_from_table(
            table,
            annotation=context.annotation,
            screen=screen,
            card_predictions=card_predictions,
        )
        metadata["recognized_table"] = _recognized_table_to_dict(table)
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
        legal_actions, to_call, action_block_reason = self._legal_actions(
            table.buttons,
            annotation=annotation,
            hero_stack=hero.stack,
            hero_committed=hero.committed,
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
                button_seat=self.controlled_seat,
                small_blind=self.small_blind,
                big_blind=self.big_blind,
                min_raise=self.big_blind,
                to_call=to_call,
                legal_actions=legal_actions,
                metadata={
                    "source": "poker_legends_prototype",
                    "recognizer": "PokerLegendsTableRecognizer",
                    "screen_kind": screen.kind.value,
                },
            ),
            None,
        )

    def _legal_actions(
        self,
        buttons: tuple[RecognizedButton, ...],
        *,
        annotation: Mapping[str, object],
        hero_stack: int,
        hero_committed: int,
    ) -> tuple[tuple[Action, ...], int, str | None]:
        actions: list[Action] = []
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
                amount = _button_amount(annotation, button.command)
                if amount is None:
                    return (), 0, "missing_call_amount"
                to_call = amount
                actions.append(Action(ActionType.CALL, amount=amount))
            elif action_type is ActionType.RAISE:
                max_amount = hero_stack + hero_committed
                min_amount = min(max_amount, max(hero_committed + self.big_blind, self.big_blind))
                actions.append(
                    Action(
                        ActionType.RAISE,
                        amount=min_amount,
                        min_amount=min_amount,
                        max_amount=max_amount,
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


def _recognized_table_from_predictions(
    *,
    source: str,
    image: str,
    frame_id: str,
    screen: ScreenState,
    annotation: Mapping[str, object] | None,
    card_predictions: tuple[PokerLegendsCardConsensusPrediction, ...],
    button_predictions: tuple[PokerLegendsButtonPrediction, ...],
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
    buttons = tuple(
        RecognizedButton(
            label=_truth_button_label(annotation, prediction.slot)
            or (prediction.action_type or ""),
            action_type=prediction.action_type,
            command=prediction.slot,
            confidence=prediction.confidence,
        )
        for prediction in button_predictions
        if prediction.visible and prediction.action_type is not None
    )
    seats = _recognized_seats_from_annotation(
        annotation,
        controlled_seat=controlled_seat,
        hero_hole_cards=hero_hole_cards,
    )
    confidences = [
        screen.confidence,
        *(card.confidence for card in board if card.visible),
        *(card.confidence for card in hero_hole_cards if card.visible),
        *(button.confidence for button in buttons),
    ]
    return RecognizedTable(
        source=source,
        image=image,
        hand_id=frame_id,
        street=_street_name_from_annotation(annotation, board),
        current_seat=controlled_seat if screen.hero_turn is True else _current_seat(seats),
        pot=_pot_from_annotation(annotation),
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
            )
        )
    if not seats and hero_hole_cards:
        stack = _hero_stack_from_texts(annotation)
        if stack is not None:
            seats.append(
                RecognizedSeat(
                    seat=controlled_seat,
                    stack=stack,
                    committed=0,
                    active=True,
                    current=True,
                    hole_cards=hero_hole_cards,
                    confidence=1.0,
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


def _street_name_from_annotation(
    annotation: Mapping[str, object] | None,
    board: tuple[RecognizedCard, ...],
) -> str:
    table_state = annotation.get("table_state") if annotation is not None else None
    if isinstance(table_state, Mapping):
        raw = table_state.get("street")
        if isinstance(raw, str) and raw in _STREET_MAP:
            return raw
    visible_board_count = len(tuple(card for card in board if card.visible and card.card))
    inferred_streets: dict[int, str] = {
        0: Street.PREFLOP.value,
        3: Street.FLOP.value,
        4: Street.TURN.value,
        5: Street.RIVER.value,
    }
    return inferred_streets.get(visible_board_count, "unknown")


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


def _pot_from_annotation(annotation: Mapping[str, object] | None) -> int | None:
    for text in _mapping_sequence(None if annotation is None else annotation.get("texts")):
        if str(text.get("name") or "") == "pot" and bool(text.get("visible", True)):
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


def _truth_button_label(annotation: Mapping[str, object] | None, slot: str) -> str | None:
    for button in _mapping_sequence(None if annotation is None else annotation.get("buttons")):
        if str(button.get("name") or "") == slot:
            label = button.get("label")
            if isinstance(label, str) and label.strip():
                return label
    return None


def _button_amount(annotation: Mapping[str, object], slot: str) -> int | None:
    label = _truth_button_label(annotation, slot)
    if label is None:
        return None
    digits = "".join(char for char in label if char.isdigit())
    if not digits:
        return 0 if "check" in label.lower() else None
    return int(digits)


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
