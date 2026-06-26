import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
from holdem_common import Card
from holdem_engine import HoldemConfig
from holdem_game.app import HoldemGameApp, build_arg_parser


def cards(*codes: str) -> tuple[Card, ...]:
    return tuple(Card.from_code(code) for code in codes)


def test_app_renders_table_in_dummy_video_driver() -> None:
    app = HoldemGameApp(
        HoldemConfig(
            starting_stacks=(100, 100),
            deck=cards("As", "Ks", "Ah", "Kh", "2c", "7d", "9s", "Jc", "3h"),
        ),
        size=(900, 620),
    )
    try:
        app.draw()

        assert app.screen.get_at((450, 310)) != pygame.Color(0, 0, 0, 255)
        assert app.buttons
    finally:
        app.close()


def test_app_handles_human_action_click() -> None:
    app = HoldemGameApp(
        HoldemConfig(
            starting_stacks=(100, 100),
            deck=cards("As", "Ks", "Ah", "Kh", "2c", "7d", "9s", "Jc", "3h"),
        ),
        size=(900, 620),
    )
    try:
        before = app.state
        button = app.buttons[0]

        app.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                {"button": 1, "pos": button.rect.center},
            ),
        )

        assert app.state is not before
    finally:
        app.close()


def test_app_can_drive_human_seat_through_bot_pipeline() -> None:
    app = HoldemGameApp(
        HoldemConfig(
            starting_stacks=(100, 100),
            deck=cards("As", "Ks", "Ah", "Kh", "2c", "7d", "9s", "Jc", "3h"),
        ),
        bot_seat=0,
        bot_delay_ms=0,
        size=(900, 620),
    )
    try:
        before = app.state

        result = app.tick(force_bot=True)

        assert result is not None
        assert result.acted
        assert app.state is not before
    finally:
        app.close()


def test_app_hides_human_buttons_when_bot_controls_that_seat() -> None:
    app = HoldemGameApp(
        HoldemConfig(
            starting_stacks=(100, 100),
            deck=cards("As", "Ks", "Ah", "Kh", "2c", "7d", "9s", "Jc", "3h"),
        ),
        bot_seat=0,
        size=(900, 620),
    )
    try:
        app.draw()

        assert app.buttons == []
    finally:
        app.close()


def test_arg_parser_accepts_bot_seat() -> None:
    args = build_arg_parser().parse_args(["--bot-seat", "0", "--bot-delay-ms", "25"])

    assert args.bot_seat == 0
    assert args.bot_delay_ms == 25
