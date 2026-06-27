import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
from holdem_common import ActionType, Card
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


def test_app_exposes_multiple_raise_sizes() -> None:
    app = HoldemGameApp(
        HoldemConfig(
            starting_stacks=(100, 100, 100),
            deck=cards("As", "Ks", "Qh", "Ah", "Kh", "Jd", "2c", "7d", "9s", "Jc", "3h"),
        ),
        size=(900, 620),
    )
    try:
        raise_amounts = [
            button.action.amount
            for button in app.buttons
            if button.action is not None and button.action.action_type is ActionType.RAISE
        ]

        assert len(raise_amounts) >= 2
        assert raise_amounts == sorted(set(raise_amounts))
    finally:
        app.close()


def test_app_handles_keyboard_call_shortcut() -> None:
    app = HoldemGameApp(
        HoldemConfig(
            starting_stacks=(100, 100, 100),
            deck=cards("As", "Ks", "Qh", "Ah", "Kh", "Jd", "2c", "7d", "9s", "Jc", "3h"),
        ),
        size=(900, 620),
    )
    try:
        before = app.state

        app.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_c}))

        assert app.state is not before
        assert any(line.startswith("You:") for line in app.action_log)
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


def test_app_logs_ai_policy_reason() -> None:
    app = HoldemGameApp(
        HoldemConfig(
            starting_stacks=(100, 100, 100),
            deck=cards("As", "Ks", "Qh", "Ah", "Kh", "Jd", "2c", "7d", "9s", "Jc", "3h"),
        ),
        size=(900, 620),
    )
    try:
        assert any("AI " in line and "(" in line for line in app.action_log)
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


def test_arg_parser_accepts_table_and_bot_config() -> None:
    args = build_arg_parser().parse_args(
        [
            "--players",
            "4",
            "--human-seat",
            "2",
            "--starting-stack",
            "500",
            "--small-blind",
            "5",
            "--big-blind",
            "10",
            "--bot-seat",
            "0",
            "--bot-delay-ms",
            "25",
        ]
    )

    assert args.players == 4
    assert args.human_seat == 2
    assert args.starting_stack == 500
    assert args.small_blind == 5
    assert args.big_blind == 10
    assert args.bot_seat == 0
    assert args.bot_delay_ms == 25
