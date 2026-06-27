"""Vision recognition outputs and fixture evaluation helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable
from dataclasses import dataclass

from holdem_bot.vision.annotations import TableAnnotation


@dataclass(frozen=True, slots=True)
class RecognizedCard:
    slot: str
    card: str | None
    visible: bool
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class RecognizedButton:
    label: str
    action_type: str | None
    command: str
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class RecognizedSeat:
    seat: int
    stack: int
    committed: int
    active: bool
    current: bool
    hole_cards: tuple[RecognizedCard, ...]
    confidence: float = 1.0
    #: Table position label if recognized (e.g. "button"/"sb"/"bb"/"hero"); None
    #: when unknown. Used to place the dealer button instead of assuming hero.
    position: str | None = None


@dataclass(frozen=True, slots=True)
class RecognizedTable:
    source: str
    image: str
    hand_id: str
    street: str
    current_seat: int | None
    pot: int | None
    board: tuple[RecognizedCard, ...]
    seats: tuple[RecognizedSeat, ...]
    buttons: tuple[RecognizedButton, ...]
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class CategoryScore:
    category: str
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 1.0
        return self.correct / self.total


@dataclass(frozen=True, slots=True)
class RecognitionScore:
    correct: int
    total: int
    categories: tuple[CategoryScore, ...]

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 1.0
        return self.correct / self.total

    def category(self, name: str) -> CategoryScore:
        for score in self.categories:
            if score.category == name:
                return score
        raise KeyError(f"unknown score category: {name}")


def recognize_from_annotation(annotation: TableAnnotation) -> RecognizedTable:
    return RecognizedTable(
        source=annotation.source,
        image=annotation.image,
        hand_id=annotation.hand_id,
        street=annotation.street,
        current_seat=annotation.current_seat,
        pot=_pot_from_annotation(annotation),
        board=tuple(
            RecognizedCard(
                slot=card.slot,
                card=card.card,
                visible=card.visible,
            )
            for card in annotation.board
        ),
        seats=tuple(
            RecognizedSeat(
                seat=seat.seat,
                stack=seat.stack,
                committed=seat.committed,
                active=seat.active,
                current=seat.current,
                hole_cards=tuple(
                    RecognizedCard(
                        slot=card.slot,
                        card=card.card,
                        visible=card.visible,
                    )
                    for card in seat.hole_cards
                ),
            )
            for seat in annotation.seats
        ),
        buttons=tuple(
            RecognizedButton(
                label=button.label,
                action_type=button.action_type,
                command=button.command,
            )
            for button in annotation.buttons
        ),
    )


def evaluate_recognition(
    recognized: RecognizedTable,
    expected: TableAnnotation,
) -> RecognitionScore:
    scores: defaultdict[str, list[bool]] = defaultdict(list)

    _record(scores, "table", recognized.street == expected.street)
    _record(scores, "table", recognized.current_seat == expected.current_seat)
    _record(scores, "chips", recognized.pot == _pot_from_annotation(expected))

    recognized_board = {card.slot: card for card in recognized.board}
    for expected_card in expected.board:
        card = recognized_board.get(expected_card.slot)
        _record(
            scores,
            "cards",
            card is not None
            and card.card == expected_card.card
            and card.visible == expected_card.visible,
        )
    _record_extras(scores, "cards", recognized_board.keys(), {card.slot for card in expected.board})

    recognized_seats = {seat.seat: seat for seat in recognized.seats}
    for expected_seat in expected.seats:
        seat = recognized_seats.get(expected_seat.seat)
        if seat is None:
            _record(scores, "chips", False)
            _record(scores, "chips", False)
            _record(scores, "seats", False)
            _record(scores, "seats", False)
            for _expected_card in expected_seat.hole_cards:
                _record(scores, "cards", False)
            continue
        _record(scores, "chips", seat.stack == expected_seat.stack)
        _record(scores, "chips", seat.committed == expected_seat.committed)
        _record(scores, "seats", seat.active == expected_seat.active)
        _record(scores, "seats", seat.current == expected_seat.current)
        recognized_hole = {card.slot: card for card in seat.hole_cards}
        for expected_card in expected_seat.hole_cards:
            card = recognized_hole.get(expected_card.slot)
            _record(
                scores,
                "cards",
                card is not None
                and card.card == expected_card.card
                and card.visible == expected_card.visible,
            )
        _record_extras(
            scores,
            "cards",
            recognized_hole.keys(),
            {card.slot for card in expected_seat.hole_cards},
        )
    _record_extras(scores, "seats", recognized_seats.keys(), {seat.seat for seat in expected.seats})

    recognized_buttons = {
        (button.action_type, button.command): button for button in recognized.buttons
    }
    for expected_button in expected.buttons:
        button = recognized_buttons.get((expected_button.action_type, expected_button.command))
        _record(
            scores,
            "buttons",
            button is not None
            and button.action_type == expected_button.action_type
            and button.command == expected_button.command,
        )
    _record_extras(
        scores,
        "buttons",
        recognized_buttons.keys(),
        {(button.action_type, button.command) for button in expected.buttons},
    )

    categories = tuple(
        CategoryScore(
            category=category,
            correct=sum(results),
            total=len(results),
        )
        for category, results in sorted(scores.items())
    )
    return RecognitionScore(
        correct=sum(score.correct for score in categories),
        total=sum(score.total for score in categories),
        categories=categories,
    )


def _record(scores: defaultdict[str, list[bool]], category: str, correct: bool) -> None:
    scores[category].append(correct)


def _record_extras(
    scores: defaultdict[str, list[bool]],
    category: str,
    recognized: Iterable[Hashable],
    expected: Iterable[Hashable],
) -> None:
    recognized_set = set(recognized)
    expected_set = set(expected)
    for _extra in recognized_set - expected_set:
        scores[category].append(False)


def _pot_from_annotation(annotation: TableAnnotation) -> int | None:
    for text in annotation.texts:
        if text.name == "pot":
            return int(text.value)
    return None
