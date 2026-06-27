"""Playable pygame host for holdem-lab."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from typing import NoReturn

import pygame
from holdem_ai import PROFILE_NAMES, PolicyDecision, evaluate_best_hand, profile_from_name
from holdem_bot import BotOrchestrator, BotStepResult
from holdem_bot.adapters import ActionCallbackAutomator, StateCapture, StateRecognizer
from holdem_common import Action, ActionType, GameState
from holdem_engine import HoldemConfig, HoldemEnv

from holdem_game.table_view import (
    ActionButton,
    SettlementView,
    TableView,
    TurnIndicatorView,
    label_for_action,
)

DEFAULT_SIZE = (1180, 760)
DEFAULT_SMALL_BLIND = 5
DEFAULT_BIG_BLIND = 10
DEFAULT_STARTING_BIG_BLINDS = 100
DEFAULT_PLAYERS = 3
DEFAULT_AI_DELAY_MS = 700
DEFAULT_TURN_TIMEOUT_SEC = 30
DEFAULT_SETTLEMENT_REVEAL_DELAY_MS = 1_000
_HAND_RANK_NAMES = {
    8: "straight flush",
    7: "quads",
    6: "full house",
    5: "flush",
    4: "straight",
    3: "trips",
    2: "two pair",
    1: "pair",
    0: "high card",
}


class HoldemGameApp:
    def __init__(
        self,
        config: HoldemConfig | None = None,
        *,
        human_seat: int = 0,
        bot_seat: int | None = None,
        bot_delay_ms: int = 450,
        ai_delay_ms: int = DEFAULT_AI_DELAY_MS,
        turn_timeout_ms: int = DEFAULT_TURN_TIMEOUT_SEC * 1000,
        ai_profile_name: str = "current",
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
        self.ai_delay_ms = max(0, ai_delay_ms)
        self.turn_timeout_ms = max(1_000, turn_timeout_ms)
        self.ai_profile = profile_from_name(ai_profile_name)
        self.last_bot_action_ms = 0
        self.local_ai_ready_at_ms: int | None = None
        self.turn_seat: int | None = None
        self.turn_started_ms = pygame.time.get_ticks()
        self.base_config = config or HoldemConfig(
            starting_stacks=tuple(
                DEFAULT_BIG_BLIND * DEFAULT_STARTING_BIG_BLINDS for _ in range(DEFAULT_PLAYERS)
            ),
            small_blind=DEFAULT_SMALL_BLIND,
            big_blind=DEFAULT_BIG_BLIND,
        )
        if self.human_seat < 0 or self.human_seat >= len(self.base_config.starting_stacks):
            raise ValueError("human_seat is outside the table")
        if self.bot_seat is not None and (
            self.bot_seat < 0 or self.bot_seat >= len(self.base_config.starting_stacks)
        ):
            raise ValueError("bot_seat is outside the table")
        self.initial_stacks = self.base_config.starting_stacks
        self.session_stacks = self.initial_stacks
        self.session_profit = tuple(0 for _ in self.initial_stacks)
        self.hand_number = 0
        self.hand_settled = False
        self.terminal_announced = False
        self.settlement_reveal_at_ms: int | None = None
        self.ai_paused = False
        self.env = HoldemEnv(self.base_config)
        self.state: GameState
        self.buttons: list[ActionButton] = []
        self.message = "New hand"
        self.action_log: list[str] = []
        self.bet_input = ""
        self.show_session_panel = True
        self.view = TableView(size)
        self.bot_orchestrator = self._build_bot_orchestrator()
        self.reset_hand()

    def reset_hand(self) -> None:
        self.hand_number += 1
        config = self._config_for_hand(self.hand_number)
        self.env = HoldemEnv(config)
        self.state = self.env.reset()
        self.message = "New hand"
        self.action_log = [f"Hand {self.hand_number}"]
        self.bet_input = ""
        self.hand_settled = False
        self.terminal_announced = False
        self.settlement_reveal_at_ms = None
        self._after_state_change(force_new_turn=True)

    def run(self) -> NoReturn:
        while True:
            for event in pygame.event.get():
                if not self.handle_event(event):
                    self.close()
                    raise SystemExit(0)
            self.tick()
            self.draw()
            self.clock.tick(30)

    def tick(self, *, force_bot: bool = False, force_ai: bool = False) -> BotStepResult | None:
        self._sync_turn_state()
        if self.state.current_seat is None:
            self._handle_terminal_state()
            self._refresh_buttons()
            return None
        if self.ai_paused:
            return None
        if self._tick_local_ai(force_ai=force_ai):
            return None
        if self.bot_orchestrator is None or self.state.current_seat != self.bot_seat:
            return None

        now = pygame.time.get_ticks()
        if not force_bot and now - self.last_bot_action_ms < self.bot_delay_ms:
            return None

        self.last_bot_action_ms = now
        result = self.bot_orchestrator.run_once()
        if not result.acted:
            self.message = f"Bot seat {self.bot_seat}: {result.reason}"
        elif result.action is not None and result.policy_decision is not None:
            self._annotate_recent_action(
                actor=f"Bot {self.bot_seat}",
                action=result.action,
                policy_decision=result.policy_decision,
            )
        self._after_state_change()
        return result

    def draw(self) -> None:
        self._refresh_buttons()
        self.view.draw(
            self.screen,
            self.visible_state(),
            human_seat=self.human_seat,
            buttons=self.buttons,
            message=self.message,
            action_log=self.action_log,
            amount_text=self._amount_text(),
            session_text=self._session_text() if self.show_session_panel else None,
            settlement=self._settlement_view(),
            turn_indicator=self._turn_indicator_view(),
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
        if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
            self._toggle_ai_pause()
            return True
        if event.type == pygame.KEYDOWN and self._handle_bet_input(event):
            return True
        if event.type == pygame.KEYDOWN and self._handle_key_action(event.key):
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
        self.bet_input = ""
        self._record_action("You", action)
        self._after_state_change()

    def _apply_bot_action(self, action: Action) -> None:
        result = self.env.step(action)
        self.state = result.observation
        self._record_action(f"Bot {self.bot_seat}", action)
        self._after_state_change(terminal_delay_ms=DEFAULT_SETTLEMENT_REVEAL_DELAY_MS)

    def _after_state_change(
        self,
        *,
        force_new_turn: bool = False,
        terminal_delay_ms: int = 0,
    ) -> None:
        self._sync_turn_state(force_new_turn=force_new_turn)
        if self.state.current_seat is None:
            self._handle_terminal_state(delay_ms=terminal_delay_ms)
        elif self.ai_paused and self._is_local_ai_turn():
            self.message = f"AI paused  Seat {self.state.current_seat}"
        elif self._is_local_ai_turn():
            self._schedule_local_ai()
            if not self.message.startswith(f"AI {self.state.current_seat}:"):
                self.message = f"AI {self.state.current_seat} thinking"
        elif self.state.current_seat == self.bot_seat:
            self.message = f"Bot seat {self.bot_seat} thinking"
        elif self.state.current_seat == self.human_seat:
            self.message = f"Your turn  Seat {self.human_seat}"
        self._refresh_buttons()

    def _sync_turn_state(self, *, force_new_turn: bool = False) -> None:
        seat = self.state.current_seat
        if not force_new_turn and seat == self.turn_seat:
            return

        now = pygame.time.get_ticks()
        self.turn_seat = seat
        self.turn_started_ms = now
        self.local_ai_ready_at_ms = None
        if seat == self.bot_seat:
            self.last_bot_action_ms = now

    def _is_local_ai_turn(self) -> bool:
        seat = self.state.current_seat
        return seat is not None and seat != self.human_seat and seat != self.bot_seat

    def _schedule_local_ai(self) -> None:
        if self.local_ai_ready_at_ms is None:
            self.local_ai_ready_at_ms = pygame.time.get_ticks() + self.ai_delay_ms

    def _tick_local_ai(self, *, force_ai: bool = False) -> bool:
        if not self._is_local_ai_turn():
            return False

        self._schedule_local_ai()
        now = pygame.time.get_ticks()
        if (
            not force_ai
            and self.local_ai_ready_at_ms is not None
            and now < self.local_ai_ready_at_ms
        ):
            return False

        seat = self.state.current_seat
        if seat is None:
            return False

        policy_decision = self.ai_profile.policy.explain(self.env.observe(seat=seat))
        action = policy_decision.action
        result = self.env.step(action)
        self.state = result.observation
        self._record_action(f"AI {seat}", action, policy_decision=policy_decision)
        self._after_state_change(terminal_delay_ms=DEFAULT_SETTLEMENT_REVEAL_DELAY_MS)
        return True

    def _handle_terminal_state(self, *, delay_ms: int = 0) -> None:
        if self.terminal_announced:
            return

        self._settle_session_stacks()
        now = pygame.time.get_ticks()
        if delay_ms > 0 and self.settlement_reveal_at_ms is None:
            self.settlement_reveal_at_ms = now + delay_ms
        if self.settlement_reveal_at_ms is not None and now < self.settlement_reveal_at_ms:
            return

        self.message = self._terminal_summary()
        if not self.action_log or self.action_log[-1] != self.message:
            self.action_log.append(self.message)
            self.action_log = self.action_log[-8:]
        showdown = self._showdown_summary()
        if showdown is not None and self.action_log[-1] != showdown:
            self.action_log.append(showdown)
            self.action_log = self.action_log[-8:]
        self.terminal_announced = True

    def _config_for_hand(self, hand_number: int) -> HoldemConfig:
        player_count = len(self.base_config.starting_stacks)
        button_seat = (self.base_config.button_seat + hand_number - 1) % player_count
        if player_count == 2:
            small_blind_seat = button_seat
            big_blind_seat = (button_seat + 1) % player_count
        else:
            small_blind_seat = (button_seat + 1) % player_count
            big_blind_seat = (button_seat + 2) % player_count
        return replace(
            self.base_config,
            hand_id=f"hand-{hand_number}",
            starting_stacks=self.session_stacks,
            button_seat=button_seat,
            small_blind_seat=small_blind_seat,
            big_blind_seat=big_blind_seat,
        )

    def _refresh_buttons(self) -> None:
        if self.state.current_seat is None:
            self.bet_input = ""
            if not self.terminal_announced:
                self.buttons = []
                return
            self.buttons = self._layout_buttons(
                (("Next hand", None, "new_hand"),),
            )
            return

        if self.state.current_seat == self.bot_seat:
            self.bet_input = ""
            self.buttons = []
            return

        if self.state.current_seat != self.human_seat:
            self.bet_input = ""
            self.buttons = []
            return

        button_specs = self._button_specs_for_actions(self._human_button_actions())
        self.buttons = self._layout_buttons(button_specs)

    def _human_button_actions(self) -> tuple[Action, ...]:
        actions = list(self.state.legal_actions)
        action_types = {action.action_type for action in actions}
        if (
            self.state.current_seat == self.human_seat
            and ActionType.FOLD not in action_types
            and ActionType.CHECK in action_types
        ):
            actions.append(Action(ActionType.FOLD))
        return tuple(actions)

    def _button_specs_for_actions(
        self,
        actions: Sequence[Action],
    ) -> tuple[tuple[str, Action | None, str], ...]:
        specs: list[tuple[str, Action | None, str]] = []
        sized_amounts: set[tuple[ActionType, int]] = set()
        for action in actions:
            if action.action_type in {ActionType.BET, ActionType.RAISE}:
                for sized_action in self._sized_bet_actions(action):
                    sized_amounts.add((sized_action.action_type, sized_action.amount))
                    specs.append((label_for_action(sized_action), sized_action, "action"))
            else:
                specs.append((label_for_action(action), action, "action"))
        custom_action = self._custom_bet_action()
        if (
            custom_action is not None
            and (custom_action.action_type, custom_action.amount) not in sized_amounts
        ):
            specs.append((label_for_action(custom_action), custom_action, "action"))
        return tuple(specs)

    def _sized_bet_actions(self, action: Action) -> tuple[Action, ...]:
        if action.min_amount is None or action.max_amount is None:
            return (action,)
        if self.state.current_seat is None:
            return (action,)

        player = self.state.player(self.state.current_seat)
        min_amount = action.min_amount
        max_amount = action.max_amount
        pot_after_call = self.state.pot_total + self.state.to_call
        raw_amounts = [
            min_amount,
            player.committed
            + self.state.to_call
            + max(self.state.big_blind, int(pot_after_call * 0.5)),
            player.committed + self.state.to_call + max(self.state.big_blind, pot_after_call),
        ]
        amounts: list[int] = []
        for raw_amount in raw_amounts:
            amount = min(max_amount, max(min_amount, raw_amount))
            if amount not in amounts:
                amounts.append(amount)

        return tuple(
            Action(
                action.action_type,
                amount=amount,
                min_amount=min_amount,
                max_amount=max_amount,
            )
            for amount in amounts
        )

    def _bet_action_template(self) -> Action | None:
        for action in self.state.legal_actions:
            if action.action_type in {ActionType.BET, ActionType.RAISE}:
                return action
        return None

    def _custom_bet_action(self) -> Action | None:
        template = self._bet_action_template()
        if template is None or template.min_amount is None or template.max_amount is None:
            return None
        if not self.bet_input:
            return None
        try:
            amount = int(self.bet_input)
        except ValueError:
            return None
        if amount < template.min_amount or amount > template.max_amount:
            return None
        return Action(
            template.action_type,
            amount=amount,
            min_amount=template.min_amount,
            max_amount=template.max_amount,
        )

    def _layout_buttons(
        self,
        specs: Sequence[tuple[str, Action | None, str]],
    ) -> list[ActionButton]:
        if not specs:
            return []

        width, height = self.size
        gap = 14
        available_width = max(280, width - 80)
        button_width = max(96, min(132, (available_width - gap * (len(specs) - 1)) // len(specs)))
        button_height = 48
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

    def _handle_bet_input(self, event: pygame.event.Event) -> bool:
        template = self._bet_action_template()
        if template is None or template.min_amount is None or template.max_amount is None:
            return False

        if event.key in {pygame.K_RETURN, pygame.K_KP_ENTER}:
            action = self._custom_bet_action()
            if action is not None:
                self._apply_human_action(action)
            return True
        if event.key == pygame.K_BACKSPACE:
            self.bet_input = self.bet_input[:-1]
            self._set_amount_message()
            self._refresh_buttons()
            return True
        if event.key == pygame.K_DELETE:
            self.bet_input = ""
            self._set_amount_message()
            self._refresh_buttons()
            return True
        if event.key in {pygame.K_UP, pygame.K_DOWN}:
            step = self.state.big_blind
            current = int(self.bet_input) if self.bet_input else template.min_amount
            delta = step if event.key == pygame.K_UP else -step
            amount = min(template.max_amount, max(template.min_amount, current + delta))
            self.bet_input = str(amount)
            self._set_amount_message()
            self._refresh_buttons()
            return True

        text = getattr(event, "unicode", "")
        if text.isdecimal():
            self.bet_input = f"{self.bet_input}{text}"[:6]
            self._set_amount_message()
            self._refresh_buttons()
            return True
        return False

    def _handle_key_action(self, key: int) -> bool:
        action_types_by_key = {
            pygame.K_f: (ActionType.FOLD,),
            pygame.K_c: (ActionType.CALL, ActionType.CHECK),
            pygame.K_SPACE: (ActionType.CHECK, ActionType.CALL),
            pygame.K_b: (ActionType.BET, ActionType.RAISE),
            pygame.K_r: (ActionType.RAISE, ActionType.BET),
            pygame.K_a: (ActionType.ALL_IN,),
        }
        desired_types = action_types_by_key.get(key)
        if desired_types is None:
            return False
        for action_type in desired_types:
            for button in self.buttons:
                if button.action is not None and button.action.action_type is action_type:
                    self._apply_human_action(button.action)
                    return True
        return False

    def _amount_text(self) -> str | None:
        template = self._bet_action_template()
        if template is None or template.min_amount is None or template.max_amount is None:
            return None
        value = self.bet_input or "-"
        return f"Amount {value}  Range {template.min_amount}-{template.max_amount}"

    def _turn_indicator_view(self) -> TurnIndicatorView | None:
        seat = self.state.current_seat
        if seat is None:
            return None

        now = pygame.time.get_ticks()
        elapsed_ms = max(0, now - self.turn_started_ms)
        timeout_ms = max(1, self.turn_timeout_ms)
        remaining_ms = max(0, timeout_ms - elapsed_ms)
        progress = min(1.0, elapsed_ms / timeout_ms)
        urgent = remaining_ms <= 5_000

        if seat == self.human_seat:
            title = f"Turn: {self._seat_label(seat)}"
            subtitle = f"{remaining_ms / 1000:.1f}s left"
        elif seat == self.bot_seat:
            wait_ms = max(0, self.bot_delay_ms - (now - self.last_bot_action_ms))
            title = f"Turn: Bot {seat}"
            subtitle = f"Acting in {wait_ms / 1000:.1f}s  Max {timeout_ms / 1000:.0f}s"
        elif self.ai_paused:
            title = f"Turn: {self._seat_label(seat)}"
            subtitle = f"Paused  Max {timeout_ms / 1000:.0f}s"
        else:
            ready_at = self.local_ai_ready_at_ms or now + self.ai_delay_ms
            wait_ms = max(0, ready_at - now)
            title = f"Turn: {self._seat_label(seat)}"
            subtitle = f"Thinking {wait_ms / 1000:.1f}s  Max {timeout_ms / 1000:.0f}s"

        return TurnIndicatorView(
            title=title,
            subtitle=subtitle,
            progress=progress,
            urgent=urgent,
        )

    def _set_amount_message(self) -> None:
        amount_text = self._amount_text()
        if amount_text is not None:
            self.message = amount_text

    def _record_action(
        self,
        actor: str,
        action: Action,
        *,
        policy_decision: PolicyDecision | None = None,
    ) -> None:
        line = self._action_line(actor, action, policy_decision)
        self.message = line
        self.action_log.append(line)
        self.action_log = self.action_log[-8:]

    def _annotate_recent_action(
        self,
        *,
        actor: str,
        action: Action,
        policy_decision: PolicyDecision,
    ) -> None:
        prefix = f"{actor}: {label_for_action(action)}"
        line = self._action_line(actor, action, policy_decision)
        for index in range(len(self.action_log) - 1, -1, -1):
            if self.action_log[index].startswith(prefix):
                self.action_log[index] = line
                if self.message.startswith(prefix):
                    self.message = line
                return

    def _action_line(
        self,
        actor: str,
        action: Action,
        policy_decision: PolicyDecision | None,
    ) -> str:
        line = f"{actor}: {label_for_action(action)}"
        if policy_decision is not None:
            line = f"{line} ({policy_decision.reason}, {policy_decision.strength:.2f})"
        return line

    def _terminal_summary(self) -> str:
        raw_payoffs = self.state.metadata.get("payoffs")
        if not isinstance(raw_payoffs, tuple):
            return "Hand complete"
        payoffs = tuple(int(payoff) for payoff in raw_payoffs)
        winners = tuple(index for index, payoff in enumerate(payoffs) if payoff > 0)
        if winners:
            winner_text = ",".join(str(seat) for seat in winners)
        else:
            winner_text = "split"
        payoff_text = " ".join(f"{seat}:{payoff:+d}" for seat, payoff in enumerate(payoffs))
        return f"Hand complete  Winners {winner_text}  Payoffs {payoff_text}"

    def _settlement_view(self) -> SettlementView | None:
        if self.state.current_seat is not None:
            return None
        if not self.terminal_announced:
            return None
        raw_payoffs = self.state.metadata.get("payoffs")
        if not isinstance(raw_payoffs, tuple):
            return SettlementView(
                title="Hand complete",
                subtitle="Press N or click Next hand",
                rows=(),
            )

        payoffs = tuple(int(payoff) for payoff in raw_payoffs)
        winners = tuple(index for index, payoff in enumerate(payoffs) if payoff > 0)
        hero_payoff = payoffs[self.human_seat]
        title = self._settlement_title(hero_payoff, winners)
        winner_text = (
            ", ".join(self._seat_label(seat) for seat in winners) if winners else "Split pot"
        )
        subtitle = f"Winners: {winner_text}   Press N or click Next hand"
        hand_rows = self._showdown_rows()
        rows = tuple(
            f"{self._seat_label(seat)}  Payoff {payoff:+d}  Stack {self.state.player(seat).stack}"
            for seat, payoff in enumerate(payoffs)
        )
        return SettlementView(title=title, subtitle=subtitle, rows=(*rows, *hand_rows))

    def _settlement_title(self, hero_payoff: int, winners: tuple[int, ...]) -> str:
        if hero_payoff > 0:
            return f"You win {hero_payoff:+d}"
        if hero_payoff < 0:
            return f"You lose {hero_payoff:+d}"
        if self.human_seat in winners:
            return "You split the pot"
        return "You break even"

    def _seat_label(self, seat: int) -> str:
        return "You" if seat == self.human_seat else f"AI {seat}"

    def _settle_session_stacks(self) -> None:
        if self.hand_settled:
            return
        raw_payoffs = self.state.metadata.get("payoffs")
        if isinstance(raw_payoffs, tuple):
            self.session_profit = tuple(
                profit + int(payoff)
                for profit, payoff in zip(self.session_profit, raw_payoffs, strict=True)
            )
        next_stacks = []
        for player, initial_stack in zip(self.state.players, self.initial_stacks, strict=True):
            if player.stack <= 0:
                next_stacks.append(initial_stack)
                self.action_log.append(f"Seat {player.seat} rebuy {initial_stack}")
            else:
                next_stacks.append(player.stack)
        self.session_stacks = tuple(next_stacks)
        self.action_log = self.action_log[-8:]
        self.hand_settled = True

    def _showdown_summary(self) -> str | None:
        if len(self.state.board) < 5:
            return None
        hands = self._showdown_rows()
        if not hands:
            return None
        return f"Showdown  {'  '.join(row.replace('  ', ': ', 1) for row in hands)}"

    def _showdown_rows(self) -> tuple[str, ...]:
        if len(self.state.board) < 5:
            return ()
        hands: list[str] = []
        for player in self.state.players:
            if len(player.hole_cards) < 2:
                continue
            rank = evaluate_best_hand((*player.hole_cards[:2], *self.state.board))[0]
            hands.append(f"{self._seat_label(player.seat)}  {_HAND_RANK_NAMES[rank]}")
        return tuple(hands)

    def _session_text(self) -> str:
        profit_text = " ".join(
            f"{seat}:{profit:+d}" for seat, profit in enumerate(self.session_profit)
        )
        stack_text = " ".join(f"{seat}:{stack}" for seat, stack in enumerate(self.session_stacks))
        return f"Session {profit_text}  Stacks {stack_text}"

    def _toggle_ai_pause(self) -> None:
        self.ai_paused = not self.ai_paused
        if self.ai_paused:
            self.message = "AI paused"
            return
        self.message = "AI resumed"
        self._after_state_change(force_new_turn=True)

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
            policy_explainer=self.ai_profile.policy.explain,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the holdem-lab pygame table.")
    parser.add_argument(
        "--players", type=_positive_int, default=DEFAULT_PLAYERS, help="Number of table seats."
    )
    parser.add_argument(
        "--heads-up",
        action="store_true",
        help="Start a two-seat human-vs-AI table.",
    )
    parser.add_argument("--human-seat", type=int, default=0, help="Seat controlled by the player.")
    parser.add_argument(
        "--starting-stack",
        type=_positive_int,
        default=None,
        help="Starting stack per seat. Defaults to 100 big blinds for the selected stake.",
    )
    parser.add_argument(
        "--stake-level",
        type=_positive_int,
        default=1,
        help="Blind level starting at 5/10 and doubling each level.",
    )
    parser.add_argument(
        "--small-blind",
        type=_positive_int,
        default=None,
        help="Override small blind amount.",
    )
    parser.add_argument(
        "--big-blind",
        type=_positive_int,
        default=None,
        help="Override big blind amount.",
    )
    parser.add_argument(
        "--ai-profile",
        choices=PROFILE_NAMES,
        default="current",
        help="AI policy profile for automated seats.",
    )
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
    parser.add_argument(
        "--ai-delay-ms",
        type=_non_negative_int,
        default=DEFAULT_AI_DELAY_MS,
        help="Delay before each local AI action.",
    )
    parser.add_argument(
        "--turn-timeout-sec",
        type=_positive_int,
        default=DEFAULT_TURN_TIMEOUT_SEC,
        help="Displayed maximum wait time for the current turn.",
    )
    return parser


def players_from_args(args: argparse.Namespace) -> int:
    return 2 if args.heads_up else int(args.players)


def blinds_from_args(args: argparse.Namespace) -> tuple[int, int]:
    multiplier = 2 ** (int(args.stake_level) - 1)
    small_blind = args.small_blind or DEFAULT_SMALL_BLIND * multiplier
    big_blind = args.big_blind or DEFAULT_BIG_BLIND * multiplier
    if big_blind < small_blind:
        raise ValueError("big blind must be at least the small blind")
    return int(small_blind), int(big_blind)


def starting_stack_from_args(args: argparse.Namespace, *, big_blind: int) -> int:
    return int(args.starting_stack or big_blind * DEFAULT_STARTING_BIG_BLINDS)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def main(argv: Sequence[str] | None = None) -> NoReturn:
    args = build_arg_parser().parse_args(argv)
    players = players_from_args(args)
    small_blind, big_blind = blinds_from_args(args)
    starting_stack = starting_stack_from_args(args, big_blind=big_blind)
    app = HoldemGameApp(
        HoldemConfig(
            starting_stacks=tuple(starting_stack for _ in range(players)),
            small_blind=small_blind,
            big_blind=big_blind,
        ),
        human_seat=args.human_seat,
        bot_seat=args.bot_seat,
        bot_delay_ms=args.bot_delay_ms,
        ai_delay_ms=args.ai_delay_ms,
        turn_timeout_ms=args.turn_timeout_sec * 1000,
        ai_profile_name=args.ai_profile,
    )
    app.run()


def main_heads_up(argv: Sequence[str] | None = None) -> NoReturn:
    forwarded = tuple(sys.argv[1:] if argv is None else argv)
    main(("--heads-up", *forwarded))


if __name__ == "__main__":
    main()
