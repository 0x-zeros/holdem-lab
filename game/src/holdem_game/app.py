"""Playable pygame host for holdem-lab."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from typing import NoReturn

import pygame
from holdem_ai import decide
from holdem_bot import BotOrchestrator, BotStepResult
from holdem_bot.adapters import ActionCallbackAutomator, StateCapture, StateRecognizer
from holdem_common import Action, GameState
from holdem_engine import HoldemConfig, HoldemEnv

from holdem_game.table_view import ActionButton, TableView, label_for_action

DEFAULT_SIZE = (1180, 760)


class HoldemGameApp:
    def __init__(
        self,
        config: HoldemConfig | None = None,
        *,
        human_seat: int = 0,
        bot_seat: int | None = None,
        bot_delay_ms: int = 450,
        size: tuple[int, int] = DEFAULT_SIZE,
    ) -> None:
        pygame.init()
        pygame.display.set_caption("holdem-lab")
        self.screen = pygame.display.set_mode(size)
        self.clock = pygame.time.Clock()
        self.size = size
        self.human_seat = human_seat
        self.bot_seat = bot_seat
        self.bot_delay_ms = bot_delay_ms
        self.last_bot_action_ms = 0
        self.base_config = config or HoldemConfig(starting_stacks=(200, 200, 200))
        if self.human_seat < 0 or self.human_seat >= len(self.base_config.starting_stacks):
            raise ValueError("human_seat is outside the table")
        if self.bot_seat is not None and (
            self.bot_seat < 0 or self.bot_seat >= len(self.base_config.starting_stacks)
        ):
            raise ValueError("bot_seat is outside the table")
        self.hand_number = 0
        self.env = HoldemEnv(self.base_config)
        self.state: GameState
        self.buttons: list[ActionButton] = []
        self.message = "New hand"
        self.view = TableView(size)
        self.bot_orchestrator = self._build_bot_orchestrator()
        self.reset_hand()

    def reset_hand(self) -> None:
        self.hand_number += 1
        config = replace(self.base_config, hand_id=f"hand-{self.hand_number}")
        self.env = HoldemEnv(config)
        self.state = self.env.reset()
        self.message = "New hand"
        self._advance_ai_to_controlled_seat()
        self._refresh_buttons()

    def run(self) -> NoReturn:
        while True:
            for event in pygame.event.get():
                if not self.handle_event(event):
                    self.close()
                    raise SystemExit(0)
            self.tick()
            self.draw()
            self.clock.tick(30)

    def tick(self, *, force_bot: bool = False) -> BotStepResult | None:
        if self.bot_orchestrator is None or self.state.current_seat != self.bot_seat:
            return None

        now = pygame.time.get_ticks()
        if not force_bot and now - self.last_bot_action_ms < self.bot_delay_ms:
            return None

        self.last_bot_action_ms = now
        result = self.bot_orchestrator.run_once()
        if not result.acted:
            self.message = f"Bot seat {self.bot_seat}: {result.reason}"
        self._refresh_buttons()
        return result

    def draw(self) -> None:
        self._refresh_buttons()
        self.view.draw(
            self.screen,
            self.visible_state(),
            human_seat=self.human_seat,
            buttons=self.buttons,
            message=self.message,
        )

    def visible_state(self) -> GameState:
        if self.state.current_seat is None:
            return self.state
        return self.env.observe(seat=self.human_seat)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_n:
            self.reset_hand()
            return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._handle_click(event.pos)
        return True

    def close(self) -> None:
        pygame.quit()

    def _handle_click(self, pos: tuple[int, int]) -> None:
        for button in self.buttons:
            if not button.rect.collidepoint(pos):
                continue
            if button.command == "new_hand":
                self.reset_hand()
                return
            if button.action is not None:
                self._apply_human_action(button.action)
                return

    def _apply_human_action(self, action: Action) -> None:
        result = self.env.step(action)
        self.state = result.observation
        self.message = f"You: {label_for_action(action)}"
        self._advance_ai_to_controlled_seat()
        self._refresh_buttons()

    def _apply_bot_action(self, action: Action) -> None:
        result = self.env.step(action)
        self.state = result.observation
        self.message = f"Bot {self.bot_seat}: {label_for_action(action)}"
        self._advance_ai_to_controlled_seat()
        self._refresh_buttons()

    def _advance_ai_to_controlled_seat(self) -> None:
        steps = 0
        while (
            self.state.current_seat is not None
            and self.state.current_seat != self.human_seat
            and self.state.current_seat != self.bot_seat
        ):
            if steps >= 100:
                raise RuntimeError("AI action loop exceeded 100 steps")
            seat = self.state.current_seat
            action = decide(self.env.observe(seat=seat))
            result = self.env.step(action)
            self.state = result.observation
            self.message = f"AI {seat}: {label_for_action(action)}"
            steps += 1

        if self.state.current_seat is None:
            rewards = self.state.metadata.get("payoffs")
            self.message = f"Hand complete  Payoffs {rewards}"

    def _refresh_buttons(self) -> None:
        if self.state.current_seat is None:
            self.buttons = self._layout_buttons(
                (("New hand", None, "new_hand"),),
            )
            return

        if self.state.current_seat == self.bot_seat:
            self.buttons = []
            return

        if self.state.current_seat != self.human_seat:
            self.buttons = []
            return

        button_specs = tuple(
            (label_for_action(action), action, "action") for action in self.state.legal_actions
        )
        self.buttons = self._layout_buttons(button_specs)

    def _layout_buttons(
        self,
        specs: Sequence[tuple[str, Action | None, str]],
    ) -> list[ActionButton]:
        if not specs:
            return []

        width, height = self.size
        button_width = 132
        button_height = 48
        gap = 14
        total_width = button_width * len(specs) + gap * (len(specs) - 1)
        start_x = int((width - total_width) / 2)
        y = height - 76

        buttons: list[ActionButton] = []
        for index, (label, action, command) in enumerate(specs):
            rect = pygame.Rect(
                start_x + index * (button_width + gap), y, button_width, button_height
            )
            buttons.append(ActionButton(rect=rect, label=label, action=action, command=command))
        return buttons

    def _bot_visible_state(self) -> GameState:
        if self.bot_seat is None or self.state.current_seat is None:
            return self.state
        return self.env.observe(seat=self.bot_seat)

    def _build_bot_orchestrator(self) -> BotOrchestrator | None:
        if self.bot_seat is None:
            return None
        return BotOrchestrator(
            capture=StateCapture(self._bot_visible_state, source="holdem_game"),
            recognizer=StateRecognizer(),
            automator=ActionCallbackAutomator(self._apply_bot_action),
            seat=self.bot_seat,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the holdem-lab pygame table.")
    parser.add_argument(
        "--bot-seat",
        type=int,
        default=None,
        help="Seat controlled through the bot Capture/Recognizer/Automator pipeline.",
    )
    parser.add_argument(
        "--bot-delay-ms",
        type=int,
        default=450,
        help="Delay between automated bot actions.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> NoReturn:
    args = build_arg_parser().parse_args(argv)
    app = HoldemGameApp(bot_seat=args.bot_seat, bot_delay_ms=args.bot_delay_ms)
    app.run()


if __name__ == "__main__":
    main()
