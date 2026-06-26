from holdem_bot.vision import (
    AnnotatedButton,
    AnnotatedCard,
    AnnotatedSeat,
    AnnotatedText,
    ScreenRect,
    TableAnnotation,
)


def test_table_annotation_round_trips_json() -> None:
    annotation = TableAnnotation(
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

    restored = TableAnnotation.from_json(annotation.to_json())

    assert restored == annotation
