from pathlib import Path
from typing import cast

from holdem_bot import CapturedFrame, ScreenKind
from holdem_bot.adapters import PokerLegendsTableRecognizer
from holdem_bot.vision import PokerLegendsCardConsensusPrediction, PokerLegendsNumberPrediction
from holdem_bot.vision.poker_legends_buttons import PokerLegendsButtonPrediction
from holdem_common import Action, ActionType, Street


def test_poker_legends_table_recognizer_builds_prototype_state(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    layout = {"image": str(image), "regions": {"cards": [], "board": [], "buttons": []}}
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer(
            (
                button_prediction("primary_left", "check", 0.90),
                button_prediction("primary_middle", "raise", 0.90),
                button_prediction("primary_right", "fold", 0.90),
            )
        ),
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": layout,
            },
        )
    )

    assert result.screen.kind is ScreenKind.ACTIONABLE_TABLE
    assert result.state is not None
    assert result.state.metadata["source"] == "poker_legends_prototype"
    assert result.state.street is Street.FLOP
    assert result.state.current_seat == 0
    assert [card.code for card in result.state.player(0).hole_cards] == ["As", "Kh"]
    assert [card.code for card in result.state.board] == ["2c", "7d", "Ts"]
    assert {action.action_type for action in result.state.legal_actions} == {
        ActionType.CHECK,
        ActionType.RAISE,
        ActionType.FOLD,
    }
    table = result.metadata["recognized_table"]
    assert isinstance(table, dict)
    assert table["pot"] == 150
    assert result.confidence == 0.90


def test_poker_legends_table_recognizer_skips_non_actionable_without_image_work(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    annotation["screen"] = {"kind": "table_observe", "confidence": 0.99, "hero_turn": False}
    card_recognizer = FakeCardRecognizer(())
    button_recognizer = FakeButtonRecognizer(())

    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=card_recognizer,
        button_recognizer=button_recognizer,
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {"image": str(image), "regions": {}},
            },
        )
    )

    assert result.screen.kind is ScreenKind.TABLE_OBSERVE
    assert result.state is None
    assert card_recognizer.calls == 0
    assert button_recognizer.calls == 0


def test_poker_legends_table_recognizer_fails_closed_without_pot(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    annotation["texts"] = []
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer((button_prediction("primary_left", "check", 0.90),)),
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {"image": str(image), "regions": {}},
            },
        )
    )

    assert result.screen.kind is ScreenKind.ACTIONABLE_TABLE
    assert result.state is None
    assert result.metadata["state_block_reason"] == "missing_pot"


def test_poker_legends_table_recognizer_rejects_low_confidence_number_ocr(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    annotation["texts"] = []
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer((button_prediction("primary_left", "check", 0.90),)),
        number_recognizer=FakeNumberRecognizer((number_prediction("texts", "pot", 150, 0.55),)),
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {"image": str(image), "regions": {}},
            },
        )
    )

    assert result.state is None
    assert result.metadata["state_block_reason"] == "missing_pot"


def test_poker_legends_table_recognizer_uses_number_ocr_fallbacks(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    annotation["texts"] = []
    for seat in cast(list[dict[str, object]], annotation["seats"]):
        if seat.get("name") == "hero":
            seat.pop("stack")
    annotation["buttons"] = [
        {
            "name": "primary_left",
            "visible": True,
            "action_type": "call",
            "label": "Call",
        },
        {
            "name": "primary_right",
            "visible": True,
            "action_type": "fold",
            "label": "Fold",
        },
    ]
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer(
            (
                button_prediction("primary_left", "call", 0.90),
                button_prediction("primary_right", "fold", 0.90),
            )
        ),
        number_recognizer=FakeNumberRecognizer(
            (
                number_prediction("texts", "pot", 150),
                number_prediction("texts", "hero_stack", 900),
                number_prediction("buttons", "primary_left", 25),
            ),
            expected_text_names=("pot", "hero_stack"),
            expected_button_names=("primary_left",),
        ),
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {"image": str(image), "regions": {}},
            },
        )
    )

    assert result.state is not None
    assert result.state.pots[0].amount == 150
    assert result.state.to_call == 25
    assert result.state.player(0).stack == 900
    assert Action(ActionType.CALL, amount=25) in result.state.legal_actions


class FakeCardRecognizer:
    def __init__(self, predictions: tuple[PokerLegendsCardConsensusPrediction, ...]) -> None:
        self.predictions = predictions
        self.calls = 0

    def recognize(
        self,
        _image_path: str | Path,
        _annotation: object,
        *,
        frame_id: str,
        exclude_frame_id: str | None = None,
        exclude_card: str | None = None,
    ) -> tuple[PokerLegendsCardConsensusPrediction, ...]:
        self.calls += 1
        assert frame_id == "frame_001"
        assert exclude_frame_id is None
        assert exclude_card is None
        return self.predictions


class FakeButtonRecognizer:
    def __init__(self, predictions: tuple[PokerLegendsButtonPrediction, ...]) -> None:
        self.predictions = predictions
        self.calls = 0

    def recognize(
        self,
        _image_path: str | Path,
        _annotation: object,
        *,
        frame_id: str,
        exclude_frame_id: str | None = None,
    ) -> tuple[PokerLegendsButtonPrediction, ...]:
        self.calls += 1
        assert frame_id == "frame_001"
        assert exclude_frame_id is None
        return self.predictions


class FakeNumberRecognizer:
    def __init__(
        self,
        predictions: tuple[PokerLegendsNumberPrediction, ...],
        *,
        expected_text_names: tuple[str, ...] | None = None,
        expected_button_names: tuple[str, ...] | None = None,
    ) -> None:
        self.predictions = predictions
        self.expected_text_names = expected_text_names
        self.expected_button_names = expected_button_names

    def recognize(
        self,
        _image_path: str | Path,
        _annotation: object,
        *,
        text_names: tuple[str, ...] = ("pot", "hero_stack", "right_top_stack"),
        button_names: tuple[str, ...] = ("primary_left",),
    ) -> tuple[PokerLegendsNumberPrediction, ...]:
        if self.expected_text_names is not None:
            assert text_names == self.expected_text_names
        if self.expected_button_names is not None:
            assert button_names == self.expected_button_names
        return self.predictions


def actionable_truth() -> dict[str, object]:
    return {
        "frame_id": "frame_001",
        "screen": {
            "kind": "actionable_table",
            "confidence": 0.99,
            "hero_turn": True,
            "reason": "hero can act",
        },
        "table_state": {"street": "flop"},
        "buttons": [
            {
                "name": "primary_left",
                "visible": True,
                "action_type": "check",
                "label": "Check",
            },
            {
                "name": "primary_middle",
                "visible": True,
                "action_type": "raise",
                "label": "Raise",
            },
            {
                "name": "primary_right",
                "visible": True,
                "action_type": "fold",
                "label": "Fold",
            },
        ],
        "texts": [
            {
                "name": "pot",
                "visible": True,
                "normalized_number": 150,
                "confidence": 1.0,
            }
        ],
        "seats": [
            {
                "name": "hero",
                "visible": True,
                "stack": 900,
                "committed": 0,
                "active": True,
                "current": True,
                "confidence": 1.0,
            },
            {
                "name": "villain",
                "visible": True,
                "stack": 1200,
                "committed": 0,
                "active": True,
                "current": False,
                "confidence": 1.0,
            },
        ],
    }


def card_prediction(
    group: str,
    slot: str,
    card: str,
    confidence: float,
) -> PokerLegendsCardConsensusPrediction:
    return PokerLegendsCardConsensusPrediction(
        frame_id="frame_001",
        group=group,
        slot=slot,
        visible=True,
        card=card,
        confidence=confidence,
        method="test",
        full_card=card,
        part_card=card,
        classifier_card=card,
        full_confidence=confidence,
        part_confidence=confidence,
        classifier_confidence=confidence,
    )


def button_prediction(
    slot: str,
    action_type: str,
    confidence: float,
) -> PokerLegendsButtonPrediction:
    return PokerLegendsButtonPrediction(
        frame_id="frame_001",
        slot=slot,
        visible=True,
        action_type=action_type,
        confidence=confidence,
    )


def number_prediction(
    group: str,
    name: str,
    number: int,
    confidence: float = 0.88,
) -> PokerLegendsNumberPrediction:
    return PokerLegendsNumberPrediction(
        name=name,
        group=group,
        visible=True,
        raw=str(number),
        numbers=(number,),
        first_number=number,
        sum_number=None,
        normalized_number=number,
        confidence=confidence,
    )
