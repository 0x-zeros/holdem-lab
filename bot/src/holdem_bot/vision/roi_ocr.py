"""ROI-based OpenCV/Tesseract recognizer for annotated fixtures."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pytesseract  # type: ignore[import-untyped]
from numpy.typing import NDArray

from holdem_bot.vision.annotations import ScreenRect, TableAnnotation
from holdem_bot.vision.recognition import (
    RecognizedButton,
    RecognizedCard,
    RecognizedSeat,
    RecognizedTable,
)

RgbImage = NDArray[np.uint8]

_CARD_WHITELIST = "AKQJT98765432CDHScdhs"
_TEXT_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "
_RANKS = set("AKQJT98765432")
_SUITS = set("cdhs")


@dataclass(frozen=True, slots=True)
class RoiOcrConfig:
    crop_padding: int = 4
    ocr_scale: int = 5
    face_brightness_threshold: float = 185.0
    active_panel_rgb: tuple[int, int, int] = (44, 83, 70)
    inactive_panel_rgb: tuple[int, int, int] = (31, 41, 44)


class RoiOcrRecognizer:
    def __init__(self, config: RoiOcrConfig | None = None) -> None:
        self.config = config or RoiOcrConfig()

    def recognize(self, image_path: str | Path, layout: TableAnnotation) -> RecognizedTable:
        image = _load_rgb_image(image_path)
        seats = tuple(self._recognize_seat(image, layout, seat) for seat in layout.seats)
        return RecognizedTable(
            source=layout.source,
            image=layout.image,
            hand_id=layout.hand_id,
            street=self._recognize_street(image, layout),
            current_seat=_current_seat_from_seats(seats),
            pot=self._recognize_pot(image, layout),
            board=tuple(self._recognize_card(image, card.slot, card.rect) for card in layout.board),
            seats=seats,
            buttons=tuple(self._recognize_button(image, button.rect) for button in layout.buttons),
        )

    def _recognize_street(self, image: RgbImage, layout: TableAnnotation) -> str:
        status = next((text for text in layout.texts if text.name == "street_status"), None)
        if status is None:
            return "unknown"
        text = _normalize_text(self._ocr_text(_crop(image, status.rect, self.config.crop_padding)))
        for street in ("preflop", "flop", "turn", "river", "showdown"):
            if street.upper() in text.upper():
                return street
        return "unknown"

    def _recognize_pot(self, image: RgbImage, layout: TableAnnotation) -> int | None:
        pot = next((text for text in layout.texts if text.name == "pot"), None)
        if pot is None:
            return None
        return _first_int(self._ocr_text(_crop(image, pot.rect, self.config.crop_padding)))

    def _recognize_seat(
        self,
        image: RgbImage,
        layout: TableAnnotation,
        seat: Any,
    ) -> RecognizedSeat:
        stack, committed = self._recognize_seat_chips(image, layout, seat.seat)
        return RecognizedSeat(
            seat=seat.seat,
            stack=stack,
            committed=committed,
            active=seat.active,
            current=self._is_active_panel(image, seat.rect),
            hole_cards=tuple(
                self._recognize_card(image, card.slot, card.rect) for card in seat.hole_cards
            ),
        )

    def _recognize_seat_chips(
        self,
        image: RgbImage,
        layout: TableAnnotation,
        seat: int,
    ) -> tuple[int, int]:
        text_annotation = next(
            (text for text in layout.texts if text.name == f"seat_{seat}_stack_committed"),
            None,
        )
        if text_annotation is None:
            return 0, 0
        raw_text = self._ocr_text(_crop(image, text_annotation.rect, self.config.crop_padding))
        numbers = _ints_from_text(raw_text)
        if len(numbers) >= 2:
            return numbers[0], numbers[1]
        if len(numbers) == 1:
            return numbers[0], 0
        return 0, 0

    def _recognize_card(
        self,
        image: RgbImage,
        slot: str,
        rect: ScreenRect,
    ) -> RecognizedCard:
        roi = _crop(image, rect, pad=0)
        visible = _mean_brightness(roi) >= self.config.face_brightness_threshold
        if not visible:
            return RecognizedCard(slot=slot, card=None, visible=False, confidence=0.95)
        text = self._ocr_text(roi, whitelist=_CARD_WHITELIST, psm=10)
        return RecognizedCard(
            slot=slot,
            card=_card_code_from_text(text),
            visible=True,
            confidence=0.60,
        )

    def _recognize_button(self, image: RgbImage, rect: ScreenRect) -> RecognizedButton:
        text = self._ocr_text(_crop(image, rect, self.config.crop_padding))
        action_type = _action_type_from_text(text)
        return RecognizedButton(
            label=_normalize_text(text),
            action_type=action_type,
            command="action",
            confidence=0.60 if action_type is not None else 0.25,
        )

    def _is_active_panel(self, image: RgbImage, rect: ScreenRect) -> bool:
        sample = image[rect.y + 4 : rect.y + 18, rect.x + 4 : rect.x + 36]
        active_distance = _rgb_distance(_mean_rgb(sample), self.config.active_panel_rgb)
        inactive_distance = _rgb_distance(_mean_rgb(sample), self.config.inactive_panel_rgb)
        return active_distance < inactive_distance

    def _ocr_text(
        self,
        image: RgbImage,
        *,
        whitelist: str = _TEXT_WHITELIST,
        psm: int = 7,
    ) -> str:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        scaled = cv2.resize(
            gray,
            None,
            fx=self.config.ocr_scale,
            fy=self.config.ocr_scale,
            interpolation=cv2.INTER_CUBIC,
        )
        _, threshold = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        config = f"--psm {psm} -c tessedit_char_whitelist={whitelist}"
        return cast(str, pytesseract.image_to_string(threshold, config=config)).strip()


def _load_rgb_image(path: str | Path) -> RgbImage:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cast(RgbImage, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _crop(image: RgbImage, rect: ScreenRect, pad: int) -> RgbImage:
    height, width = image.shape[:2]
    x = max(0, rect.x - pad)
    y = max(0, rect.y - pad)
    right = min(width, rect.x + rect.width + pad)
    bottom = min(height, rect.y + rect.height + pad)
    return image[y:bottom, x:right]


def _mean_rgb(image: RgbImage) -> tuple[float, float, float]:
    mean = image.reshape(-1, 3).mean(axis=0)
    return (float(mean[0]), float(mean[1]), float(mean[2]))


def _mean_brightness(image: RgbImage) -> float:
    return sum(_mean_rgb(image)) / 3.0


def _rgb_distance(first: tuple[float, float, float], second: tuple[int, int, int]) -> float:
    return sum((first[index] - second[index]) ** 2 for index in range(3))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _ints_from_text(text: str) -> list[int]:
    normalized = text.translate(str.maketrans("Il|", "111"))
    return [int(match) for match in re.findall(r"\d+", normalized)]


def _first_int(text: str) -> int | None:
    numbers = _ints_from_text(text)
    return numbers[0] if numbers else None


def _card_code_from_text(text: str) -> str | None:
    normalized = _normalize_text(text).replace("10", "T")
    rank = next((character for character in normalized.upper() if character in _RANKS), None)
    suit = next(
        (character.lower() for character in normalized if character.lower() in _SUITS), None
    )
    if rank is None or suit is None:
        return None
    return f"{rank}{suit}"


def _action_type_from_text(text: str) -> str | None:
    normalized = _normalize_text(text).lower().replace(" ", "")
    if "fold" in normalized:
        return "fold"
    if "check" in normalized:
        return "check"
    if "call" in normalized:
        return "call"
    if "raise" in normalized:
        return "raise"
    if "all" in normalized:
        return "all_in"
    return None


def _current_seat_from_seats(seats: tuple[RecognizedSeat, ...]) -> int | None:
    current = [seat.seat for seat in seats if seat.current]
    if len(current) != 1:
        return None
    return current[0]
