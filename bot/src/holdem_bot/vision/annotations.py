"""Serializable screen annotations for CV/OCR fixtures."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self


@dataclass(frozen=True, slots=True)
class ScreenRect:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("screen rect width and height must be positive")

    @classmethod
    def from_xywh(cls, xywh: tuple[int, int, int, int]) -> Self:
        return cls(x=xywh[0], y=xywh[1], width=xywh[2], height=xywh[3])


@dataclass(frozen=True, slots=True)
class AnnotatedCard:
    slot: str
    rect: ScreenRect
    card: str | None
    visible: bool


@dataclass(frozen=True, slots=True)
class AnnotatedText:
    name: str
    rect: ScreenRect
    value: str
    kind: str


@dataclass(frozen=True, slots=True)
class AnnotatedButton:
    label: str
    rect: ScreenRect
    action_type: str | None = None
    command: str = "action"


@dataclass(frozen=True, slots=True)
class AnnotatedSeat:
    seat: int
    rect: ScreenRect
    stack: int
    committed: int
    active: bool
    current: bool
    dealer: bool
    small_blind: bool
    big_blind: bool
    hole_cards: tuple[AnnotatedCard, ...]


@dataclass(frozen=True, slots=True)
class TableAnnotation:
    schema_version: int
    source: str
    image: str
    width: int
    height: int
    hand_id: str
    street: str
    current_seat: int | None
    board: tuple[AnnotatedCard, ...]
    seats: tuple[AnnotatedSeat, ...]
    texts: tuple[AnnotatedText, ...]
    buttons: tuple[AnnotatedButton, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported table annotation schema version")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("annotation image size must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, data: str) -> Self:
        return cls.from_dict(json.loads(data))

    @classmethod
    def read_json(cls, path: str | Path) -> Self:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            schema_version=int(data["schema_version"]),
            source=str(data["source"]),
            image=str(data["image"]),
            width=int(data["width"]),
            height=int(data["height"]),
            hand_id=str(data["hand_id"]),
            street=str(data["street"]),
            current_seat=_optional_int(data["current_seat"]),
            board=tuple(_card_from_dict(card) for card in data["board"]),
            seats=tuple(_seat_from_dict(seat) for seat in data["seats"]),
            texts=tuple(_text_from_dict(text) for text in data["texts"]),
            buttons=tuple(_button_from_dict(button) for button in data["buttons"]),
            metadata=dict(data.get("metadata", {})),
        )


def _rect_from_dict(data: Mapping[str, Any]) -> ScreenRect:
    return ScreenRect(
        x=int(data["x"]),
        y=int(data["y"]),
        width=int(data["width"]),
        height=int(data["height"]),
    )


def _card_from_dict(data: Mapping[str, Any]) -> AnnotatedCard:
    raw_card = data["card"]
    return AnnotatedCard(
        slot=str(data["slot"]),
        rect=_rect_from_dict(data["rect"]),
        card=None if raw_card is None else str(raw_card),
        visible=bool(data["visible"]),
    )


def _text_from_dict(data: Mapping[str, Any]) -> AnnotatedText:
    return AnnotatedText(
        name=str(data["name"]),
        rect=_rect_from_dict(data["rect"]),
        value=str(data["value"]),
        kind=str(data["kind"]),
    )


def _button_from_dict(data: Mapping[str, Any]) -> AnnotatedButton:
    raw_action_type = data["action_type"]
    return AnnotatedButton(
        label=str(data["label"]),
        rect=_rect_from_dict(data["rect"]),
        action_type=None if raw_action_type is None else str(raw_action_type),
        command=str(data["command"]),
    )


def _seat_from_dict(data: Mapping[str, Any]) -> AnnotatedSeat:
    return AnnotatedSeat(
        seat=int(data["seat"]),
        rect=_rect_from_dict(data["rect"]),
        stack=int(data["stack"]),
        committed=int(data["committed"]),
        active=bool(data["active"]),
        current=bool(data["current"]),
        dealer=bool(data["dealer"]),
        small_blind=bool(data["small_blind"]),
        big_blind=bool(data["big_blind"]),
        hole_cards=tuple(_card_from_dict(card) for card in data["hole_cards"]),
    )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is int:
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"expected int, string, or null: {value!r}")
