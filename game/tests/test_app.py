import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
from holdem_common import ActionType, Card
from holdem_engine import HoldemConfig
from holdem_game.app import (
    HoldemGameApp,
    blinds_from_args,
    build_arg_parser,
    players_from_args,
    starting_stack_from_args,
)


def cards(*codes: str) -> tuple[Card, ...]:
    return tuple(Card.from_code(code) for code in codes)


def drive_ai_until_controlled(app: HoldemGameApp, *, max_steps: int = 20) -> None:
    for _ in range(max_steps):
        if app.state.current_seat in {None, app.human_seat, app.bot_seat}:
            return
        app.tick(force_ai=True)
    raise AssertionError("AI did not return control to a human or bot seat")


def test_app_renders_table_in_dummy_video_driver() -> None:
    app = HoldemGameApp(
        HoldemConfig(
            starting_stacks=(100, 100),
            deck=cards("As", "Ks", "Ah", "Kh", "2c", "7d", "9s", "Jc", "3h"),
        ),
        size=(900, 620),
    )
    try:
        drive_ai_until_controlled(app)
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
        drive_ai_until_controlled(app)
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
        drive_ai_until_controlled(app)
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
        drive_ai_until_controlled(app)
        before = app.state

        app.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_c}))

        assert app.state is not before
        assert any(line.startswith("You:") for line in app.action_log)
    finally:
        app.close()


def test_app_accepts_custom_raise_amount_from_keyboard() -> None:
    app = HoldemGameApp(
        HoldemConfig(
            starting_stacks=(100, 100, 100),
            deck=cards("As", "Ks", "Qh", "Ah", "Kh", "Jd", "2c", "7d", "9s", "Jc", "3h"),
        ),
        size=(900, 620),
    )
    try:
        drive_ai_until_controlled(app)
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_9, "unicode": "9"}))

        assert app.bet_input == "9"
        assert any(
            button.action is not None
            and button.action.action_type is ActionType.RAISE
            and button.action.amount == 9
            for button in app.buttons
        )

        app.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN}))

        assert app.bet_input == ""
        assert any(line.startswith("You: Raise 9") for line in app.action_log)
    finally:
        app.close()


def test_app_edits_custom_bet_amount_with_backspace_and_arrows() -> None:
    app = HoldemGameApp(
        HoldemConfig(
            starting_stacks=(100, 100, 100),
            deck=cards("As", "Ks", "Qh", "Ah", "Kh", "Jd", "2c", "7d", "9s", "Jc", "3h"),
        ),
        size=(900, 620),
    )
    try:
        drive_ai_until_controlled(app)
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_1, "unicode": "1"}))
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_0, "unicode": "0"}))
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_BACKSPACE}))

        assert app.bet_input == "1"

        app.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_UP}))

        assert app.bet_input == "4"

        app.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DELETE}))

        assert app.bet_input == ""
    finally:
        app.close()


def test_app_rotates_button_and_blinds_between_hands() -> None:
    app = HoldemGameApp(
        HoldemConfig(
            starting_stacks=(100, 100, 100),
            deck=cards("As", "Ks", "Qh", "Ah", "Kh", "Jd", "2c", "7d", "9s", "Jc", "3h"),
        ),
        size=(900, 620),
    )
    try:
        assert app.state.button_seat == 0
        assert app.state.player(1).small_blind
        assert app.state.player(2).big_blind

        app.reset_hand()

        assert app.state.button_seat == 1
        assert app.state.player(2).small_blind
        assert app.state.player(0).big_blind
    finally:
        app.close()


def test_app_reports_readable_terminal_summary() -> None:
    app = HoldemGameApp(
        HoldemConfig(
            starting_stacks=(100, 100),
            deck=cards("As", "Ks", "Ah", "Kh", "2c", "7d", "9s", "Jc", "3h"),
        ),
        size=(900, 620),
    )
    try:
        drive_ai_until_controlled(app)
        fold_button = next(
            button
            for button in app.buttons
            if button.action is not None and button.action.action_type is ActionType.FOLD
        )

        app.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                {"button": 1, "pos": fold_button.rect.center},
            ),
        )

        assert app.state.current_seat is None
        assert "Winners" in app.message
        assert "Payoffs" in app.message
        assert app.action_log[-1] == app.message
        assert [button.label for button in app.buttons] == ["Next hand"]
        settlement = app._settlement_view()
        assert settlement is not None
        assert settlement.title.startswith("You lose")
        assert "AI 1" in settlement.subtitle
        assert any(row.startswith("You  Payoff") for row in settlement.rows)
        assert sum(app.session_profit) == 0
        assert app.session_stacks == tuple(player.stack for player in app.state.players)

        next_stacks = app.session_stacks
        app.reset_hand()

        assert app.env.facade.config.starting_stacks == next_stacks
    finally:
        app.close()


def test_app_logs_showdown_hand_categories() -> None:
    app = HoldemGameApp(
        HoldemConfig(
            starting_stacks=(10, 10),
            deck=cards("As", "Ks", "Ah", "Kh", "2c", "7d", "9s", "Jc", "3h", "4d", "5s", "6h"),
        ),
        size=(900, 620),
    )
    try:
        drive_ai_until_controlled(app)
        all_in_button = next(
            button
            for button in app.buttons
            if button.action is not None and button.action.action_type is ActionType.ALL_IN
        )

        app.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                {"button": 1, "pos": all_in_button.rect.center},
            ),
        )
        drive_ai_until_controlled(app)

        assert app.state.current_seat is None
        assert any(line.startswith("Showdown") for line in app.action_log)
        settlement = app._settlement_view()
        assert settlement is not None
        assert any("straight" in row or "flush" in row or "pair" in row for row in settlement.rows)
        assert sum(app.session_profit) == 0
        assert any(line.startswith("Seat ") and "rebuy" in line for line in app.action_log)
        assert all(stack > 0 for stack in app.session_stacks)
    finally:
        app.close()


def test_app_can_pause_ai_auto_advance() -> None:
    app = HoldemGameApp(
        HoldemConfig(
            starting_stacks=(100, 100, 100),
            deck=cards("As", "Ks", "Qh", "Ah", "Kh", "Jd", "2c", "7d", "9s", "Jc", "3h"),
        ),
        size=(900, 620),
    )
    try:
        drive_ai_until_controlled(app)
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_p}))
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_c}))

        assert app.ai_paused
        assert app.state.current_seat not in {None, app.human_seat}
        assert app.message.startswith("AI paused")

        app.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_p}))

        assert not app.ai_paused
        drive_ai_until_controlled(app)
        assert app.state.current_seat in {None, app.human_seat}
    finally:
        app.close()


def test_app_delays_local_ai_actions_until_tick_ready() -> None:
    app = HoldemGameApp(
        HoldemConfig(
            starting_stacks=(100, 100, 100),
            deck=cards("As", "Ks", "Qh", "Ah", "Kh", "Jd", "2c", "7d", "9s", "Jc", "3h"),
        ),
        ai_delay_ms=10_000,
        size=(900, 620),
    )
    try:
        assert app.state.current_seat not in {None, app.human_seat}
        before = app.state

        app.tick()

        assert app.state is before
        assert app.local_ai_ready_at_ms is not None

        app.tick(force_ai=True)

        assert app.state is not before
        assert any("AI " in line and "(" in line for line in app.action_log)
    finally:
        app.close()


def test_app_exposes_turn_indicator_for_current_seat() -> None:
    app = HoldemGameApp(
        HoldemConfig(
            starting_stacks=(100, 100, 100),
            deck=cards("As", "Ks", "Qh", "Ah", "Kh", "Jd", "2c", "7d", "9s", "Jc", "3h"),
        ),
        turn_timeout_ms=20_000,
        size=(900, 620),
    )
    try:
        indicator = app._turn_indicator_view()

        assert indicator is not None
        assert indicator.title == "Turn: AI 2"
        assert "Max 20s" in indicator.subtitle
    finally:
        app.close()


def test_app_keeps_fold_button_when_check_is_free_for_human() -> None:
    app = HoldemGameApp(
        HoldemConfig(
            starting_stacks=(100, 100, 100),
            deck=cards("As", "Ks", "Qh", "Ah", "Kh", "Jd", "2c", "7d", "9s", "Jc", "3h"),
        ),
        human_seat=1,
        size=(900, 620),
    )
    try:
        for action_type in (ActionType.CALL, ActionType.CALL):
            action = next(
                action for action in app.state.legal_actions if action.action_type is action_type
            )
            app.state = app.env.step(action).observation
        app._after_state_change(force_new_turn=True)

        assert app.state.current_seat == app.human_seat
        assert app.state.to_call == 0
        assert ActionType.FOLD not in {action.action_type for action in app.state.legal_actions}
        assert any(
            button.action is not None and button.action.action_type is ActionType.FOLD
            for button in app.buttons
        )

        before_active = app.state.player(app.human_seat).active
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_f}))

        assert before_active
        assert not app.state.player(app.human_seat).active
        assert any(line.startswith("You: Fold") for line in app.action_log)
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
        drive_ai_until_controlled(app)
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
        app.tick(force_ai=True)
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


def test_app_accepts_ai_profile() -> None:
    app = HoldemGameApp(
        HoldemConfig(
            starting_stacks=(100, 100),
            deck=cards("As", "Ks", "Ah", "Kh", "2c", "7d", "9s", "Jc", "3h"),
        ),
        ai_profile_name="tight",
        size=(900, 620),
    )
    try:
        assert app.ai_profile.name == "tight"
    finally:
        app.close()


def test_arg_parser_accepts_table_and_bot_config() -> None:
    args = build_arg_parser().parse_args(
        [
            "--players",
            "4",
            "--heads-up",
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
            "--ai-delay-ms",
            "900",
            "--turn-timeout-sec",
            "45",
            "--ai-profile",
            "loose",
        ]
    )

    assert args.players == 4
    assert args.heads_up
    assert players_from_args(args) == 2
    assert args.human_seat == 2
    assert args.starting_stack == 500
    assert args.small_blind == 5
    assert args.big_blind == 10
    assert args.bot_seat == 0
    assert args.bot_delay_ms == 25
    assert args.ai_delay_ms == 900
    assert args.turn_timeout_sec == 45
    assert args.ai_profile == "loose"
    assert blinds_from_args(args) == (5, 10)
    assert starting_stack_from_args(args, big_blind=10) == 500


def test_arg_parser_defaults_to_five_ten_stake() -> None:
    args = build_arg_parser().parse_args([])

    small_blind, big_blind = blinds_from_args(args)

    assert players_from_args(args) == 3
    assert (small_blind, big_blind) == (5, 10)
    assert starting_stack_from_args(args, big_blind=big_blind) == 1000


def test_arg_parser_doubles_blinds_by_stake_level() -> None:
    args = build_arg_parser().parse_args(["--stake-level", "3"])

    small_blind, big_blind = blinds_from_args(args)

    assert (small_blind, big_blind) == (20, 40)
    assert starting_stack_from_args(args, big_blind=big_blind) == 4000
