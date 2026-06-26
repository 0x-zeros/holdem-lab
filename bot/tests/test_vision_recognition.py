from dataclasses import replace

from holdem_bot.vision import (
    AnnotatedButton,
    AnnotatedCard,
    AnnotatedSeat,
    AnnotatedText,
    RecognizedCard,
    ScreenRect,
    TableAnnotation,
    evaluate_recognition,
    recognize_from_annotation,
)


def sample_annotation() -> TableAnnotation:
    return TableAnnotation(
        schema_version=1,
        source="test",
        image="sample.png",
        width=640,
        height=480,
        hand_id="hand-1",
        street="preflop",
        current_seat=0,
        board=(
            AnnotatedCard(
                slot="board_0",
                rect=ScreenRect(10, 20, 30, 40),
                card=None,
                visible=False,
            ),
        ),
        seats=(
            AnnotatedSeat(
                seat=0,
                rect=ScreenRect(100, 120, 190, 104),
                stack=200,
                committed=2,
                active=True,
                current=True,
                dealer=True,
                small_blind=False,
                big_blind=True,
                hole_cards=(
                    AnnotatedCard(
                        slot="seat_0_hole_0",
                        rect=ScreenRect(110, 160, 38, 50),
                        card="As",
                        visible=True,
                    ),
                ),
            ),
        ),
        texts=(
            AnnotatedText(
                name="pot",
                rect=ScreenRect(300, 260, 180, 30),
                value="3",
                kind="chips",
            ),
        ),
        buttons=(
            AnnotatedButton(
                label="Check",
                rect=ScreenRect(300, 400, 132, 48),
                action_type="check",
            ),
        ),
    )


def test_annotation_oracle_recognition_scores_perfectly() -> None:
    annotation = sample_annotation()
    recognized = recognize_from_annotation(annotation)

    score = evaluate_recognition(recognized, annotation)

    assert recognized.pot == 3
    assert score.accuracy == 1.0
    assert score.category("cards").accuracy == 1.0
    assert score.category("buttons").accuracy == 1.0


def test_evaluator_reports_card_mismatch() -> None:
    annotation = sample_annotation()
    recognized = recognize_from_annotation(annotation)
    seat = recognized.seats[0]
    bad_seat = replace(
        seat,
        hole_cards=(
            RecognizedCard(
                slot="seat_0_hole_0",
                card="Ks",
                visible=True,
            ),
        ),
    )
    bad_recognition = replace(recognized, seats=(bad_seat,))

    score = evaluate_recognition(bad_recognition, annotation)

    assert score.accuracy < 1.0
    assert score.category("cards").accuracy < 1.0
