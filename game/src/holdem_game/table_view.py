"""Pygame rendering for the local Texas Hold'em table."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, pi, sin

import pygame
from holdem_common import Action, ActionType, Card, GameState, PlayerState

Color = tuple[int, int, int]

BACKGROUND: Color = (18, 31, 28)
TABLE_FILL: Color = (22, 108, 74)
TABLE_RAIL: Color = (92, 64, 43)
FELT_LINE: Color = (180, 213, 190)
PANEL: Color = (31, 41, 44)
PANEL_ACTIVE: Color = (44, 83, 70)
TEXT: Color = (239, 244, 238)
MUTED: Color = (166, 181, 174)
CARD_FACE: Color = (244, 241, 232)
CARD_BACK: Color = (39, 73, 131)
CARD_EDGE: Color = (22, 24, 25)
RED_SUIT: Color = (185, 45, 58)
BLACK_SUIT: Color = (24, 24, 24)
BUTTON_FILL: Color = (220, 190, 92)
BUTTON_HOVER: Color = (238, 210, 118)
BUTTON_TEXT: Color = (25, 28, 25)

BOARD_CARD_SIZE = (58, 82)
BOARD_CARD_GAP = 12
PLAYER_PANEL_SIZE = (190, 104)
PLAYER_CARD_SIZE = (38, 50)
PLAYER_CARD_GAP = 46


@dataclass(frozen=True, slots=True)
class ActionButton:
    rect: pygame.Rect
    label: str
    action: Action | None = None
    command: str = "action"


class TableView:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self.title_font = pygame.font.SysFont("dejavusans", 30, bold=True)
        self.body_font = pygame.font.SysFont("dejavusans", 20)
        self.small_font = pygame.font.SysFont("dejavusans", 16)
        self.card_font = pygame.font.SysFont("dejavusans", 22, bold=True)

    def draw(
        self,
        surface: pygame.Surface,
        state: GameState,
        *,
        human_seat: int,
        buttons: Sequence[ActionButton],
        message: str,
    ) -> None:
        width, height = self.size
        surface.fill(BACKGROUND)
        self._draw_table(surface)
        self._draw_board(surface, state)
        self._draw_players(surface, state, human_seat=human_seat)
        self._draw_status(surface, state, message)
        self._draw_buttons(surface, buttons)
        pygame.display.flip()

    def _draw_table(self, surface: pygame.Surface) -> None:
        width, height = self.size
        rail_rect = pygame.Rect(width * 0.10, height * 0.13, width * 0.80, height * 0.58)
        felt_rect = rail_rect.inflate(-24, -24)
        pygame.draw.ellipse(surface, TABLE_RAIL, rail_rect)
        pygame.draw.ellipse(surface, TABLE_FILL, felt_rect)
        pygame.draw.ellipse(surface, FELT_LINE, felt_rect, width=2)

    def _draw_board(self, surface: pygame.Surface, state: GameState) -> None:
        for index, rect in enumerate(self.board_card_rects()):
            card = state.board[index] if index < len(state.board) else None
            self._draw_card(surface, rect, card)

        pot_text = f"Pot {state.pot_total}"
        self._draw_text(surface, pot_text, self.body_font, TEXT, self.pot_text_rect().center)

    def _draw_players(
        self,
        surface: pygame.Surface,
        state: GameState,
        *,
        human_seat: int,
    ) -> None:
        for player in state.players:
            panel = self.player_panel_rect(
                player.seat,
                player_count=len(state.players),
                human_seat=human_seat,
            )
            self._draw_player(surface, player, state.current_seat == player.seat, panel, human_seat)

    def _draw_player(
        self,
        surface: pygame.Surface,
        player: PlayerState,
        is_current: bool,
        panel: pygame.Rect,
        human_seat: int,
    ) -> None:
        pygame.draw.rect(surface, PANEL_ACTIVE if is_current else PANEL, panel, border_radius=8)
        pygame.draw.rect(
            surface, FELT_LINE if is_current else (61, 76, 76), panel, width=2, border_radius=8
        )

        role = "You" if player.seat == human_seat else f"AI {player.seat}"
        badges = "".join(
            label
            for enabled, label in (
                (player.dealer, " D"),
                (player.small_blind, " SB"),
                (player.big_blind, " BB"),
            )
            if enabled
        )
        self._draw_text(
            surface, f"{role}{badges}", self.small_font, TEXT, (panel.centerx, panel.y + 18)
        )
        self._draw_text(
            surface,
            f"Stack {player.stack}  Bet {player.committed}",
            self.small_font,
            MUTED,
            (panel.centerx, panel.y + 42),
        )

        if player.hole_cards:
            for index, card in enumerate(player.hole_cards[:2]):
                self._draw_card(surface, self.player_hole_card_rects(panel)[index], card)
        else:
            for rect in self.player_hole_card_rects(panel):
                self._draw_card(
                    surface,
                    rect,
                    None,
                    hidden=True,
                )

    def _draw_status(self, surface: pygame.Surface, state: GameState, message: str) -> None:
        title = (
            f"{state.street.value.upper()}  To call {state.to_call}  Min raise {state.min_raise}"
        )
        self._draw_text(surface, title, self.title_font, TEXT, self.title_text_rect().center)
        self._draw_text(surface, message, self.body_font, MUTED, self.message_text_rect().center)

    def _draw_buttons(self, surface: pygame.Surface, buttons: Sequence[ActionButton]) -> None:
        mouse_pos = pygame.mouse.get_pos()
        for button in buttons:
            hovered = button.rect.collidepoint(mouse_pos)
            pygame.draw.rect(
                surface,
                BUTTON_HOVER if hovered else BUTTON_FILL,
                button.rect,
                border_radius=8,
            )
            pygame.draw.rect(surface, (79, 62, 24), button.rect, width=2, border_radius=8)
            self._draw_text(surface, button.label, self.body_font, BUTTON_TEXT, button.rect.center)

    def _draw_card(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        card: Card | None,
        *,
        hidden: bool = False,
    ) -> None:
        if hidden or card is None:
            pygame.draw.rect(surface, CARD_BACK, rect, border_radius=6)
            pygame.draw.rect(surface, CARD_EDGE, rect, width=2, border_radius=6)
            pygame.draw.line(surface, (93, 135, 201), rect.topleft, rect.bottomright, width=2)
            pygame.draw.line(surface, (93, 135, 201), rect.topright, rect.bottomleft, width=2)
            return

        pygame.draw.rect(surface, CARD_FACE, rect, border_radius=6)
        pygame.draw.rect(surface, CARD_EDGE, rect, width=2, border_radius=6)
        color = RED_SUIT if card.suit.value in {"h", "d"} else BLACK_SUIT
        self._draw_text(surface, card.code, self.card_font, color, rect.center)

    def _draw_text(
        self,
        surface: pygame.Surface,
        text: str,
        font: pygame.font.Font,
        color: Color,
        center: tuple[int, int],
    ) -> None:
        rendered = font.render(text, True, color)
        rect = rendered.get_rect(center=center)
        surface.blit(rendered, rect)

    def board_card_rects(self) -> tuple[pygame.Rect, ...]:
        width, height = self.size
        card_width, card_height = BOARD_CARD_SIZE
        total_width = card_width * 5 + BOARD_CARD_GAP * 4
        start_x = int((width - total_width) / 2)
        y = int(height * 0.36)
        return tuple(
            pygame.Rect(
                start_x + index * (card_width + BOARD_CARD_GAP),
                y,
                card_width,
                card_height,
            )
            for index in range(5)
        )

    def player_panel_rect(
        self,
        seat: int,
        *,
        player_count: int,
        human_seat: int,
    ) -> pygame.Rect:
        width, height = self.size
        center_x, center_y = width / 2.0, height * 0.41
        radius_x, radius_y = width * 0.34, height * 0.29
        offset = (seat - human_seat) % player_count
        angle = (pi / 2.0) + (2.0 * pi * offset / player_count)
        panel = pygame.Rect(0, 0, *PLAYER_PANEL_SIZE)
        panel.center = (
            int(center_x + cos(angle) * radius_x),
            int(center_y + sin(angle) * radius_y),
        )
        return panel

    def player_hole_card_rects(self, panel: pygame.Rect) -> tuple[pygame.Rect, pygame.Rect]:
        card_y = panel.y + 58
        card_x = panel.centerx - 44
        card_width, card_height = PLAYER_CARD_SIZE
        return (
            pygame.Rect(card_x, card_y, card_width, card_height),
            pygame.Rect(card_x + PLAYER_CARD_GAP, card_y, card_width, card_height),
        )

    def pot_text_rect(self) -> pygame.Rect:
        board_rect = self.board_card_rects()[0]
        width = self.size[0]
        rect = pygame.Rect(0, 0, 180, 30)
        rect.center = (width // 2, board_rect.y + board_rect.height + 30)
        return rect

    def title_text_rect(self) -> pygame.Rect:
        width = self.size[0]
        rect = pygame.Rect(0, 0, min(720, width - 40), 42)
        rect.center = (width // 2, 34)
        return rect

    def message_text_rect(self) -> pygame.Rect:
        width, height = self.size
        rect = pygame.Rect(0, 0, min(720, width - 40), 34)
        rect.center = (width // 2, height - 104)
        return rect


def label_for_action(action: Action) -> str:
    match action.action_type:
        case ActionType.FOLD:
            return "Fold"
        case ActionType.CHECK:
            return "Check"
        case ActionType.CALL:
            return f"Call {action.amount}"
        case ActionType.BET:
            return f"Bet {action.amount}"
        case ActionType.RAISE:
            return f"Raise {action.amount}"
        case ActionType.ALL_IN:
            return "All in"
