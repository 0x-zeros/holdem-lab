"""Pygame rendering for the local Texas Hold'em table."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, pi, sin

import pygame
from holdem_common import Action, ActionType, Card, GameState, PlayerState, Rank, Suit

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
CARD_FACE_HIGHLIGHT: Color = (252, 250, 244)
CARD_BACK: Color = (24, 28, 54)
CARD_EDGE: Color = (22, 24, 25)
RED_SUIT: Color = (185, 45, 58)
BLACK_SUIT: Color = (24, 24, 24)
CYAN_ACCENT: Color = (64, 220, 236)
MAGENTA_ACCENT: Color = (236, 72, 174)
GOLD_ACCENT: Color = (215, 171, 73)
BUTTON_FILL: Color = (220, 190, 92)
BUTTON_HOVER: Color = (238, 210, 118)
BUTTON_TEXT: Color = (25, 28, 25)

BOARD_CARD_SIZE = (58, 82)
BOARD_CARD_GAP = 12
PLAYER_PANEL_SIZE = (190, 104)
PLAYER_CARD_SIZE = (38, 50)
PLAYER_CARD_GAP = 46
SUIT_SYMBOLS = {
    Suit.CLUBS: "♣",
    Suit.DIAMONDS: "♦",
    Suit.HEARTS: "♥",
    Suit.SPADES: "♠",
}


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
        self._font_cache: dict[tuple[int, bool], pygame.font.Font] = {}

    def draw(
        self,
        surface: pygame.Surface,
        state: GameState,
        *,
        human_seat: int,
        buttons: Sequence[ActionButton],
        message: str,
        action_log: Sequence[str] = (),
        amount_text: str | None = None,
        session_text: str | None = None,
    ) -> None:
        width, height = self.size
        surface.fill(BACKGROUND)
        self._draw_table(surface)
        self._draw_board(surface, state)
        self._draw_players(surface, state, human_seat=human_seat)
        self._draw_status(surface, state, message)
        self._draw_session(surface, session_text)
        self._draw_action_log(surface, action_log)
        self._draw_amount_input(surface, amount_text)
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

    def _draw_action_log(self, surface: pygame.Surface, action_log: Sequence[str]) -> None:
        if not action_log:
            return
        rect = self.action_log_rect()
        pygame.draw.rect(surface, (24, 33, 35), rect, border_radius=8)
        pygame.draw.rect(surface, (61, 76, 76), rect, width=2, border_radius=8)
        self._draw_text_left(surface, "Actions", self.small_font, TEXT, (rect.x + 12, rect.y + 12))
        line_y = rect.y + 38
        for line in action_log[-5:]:
            self._draw_text_left(
                surface,
                self._ellipsize(line, self.small_font, rect.width - 24),
                self.small_font,
                MUTED,
                (rect.x + 12, line_y),
            )
            line_y += 22

    def _draw_session(self, surface: pygame.Surface, session_text: str | None) -> None:
        if session_text is None:
            return
        rect = self.session_rect()
        pygame.draw.rect(surface, (24, 33, 35), rect, border_radius=8)
        pygame.draw.rect(surface, (61, 76, 76), rect, width=2, border_radius=8)
        self._draw_text_left(
            surface,
            self._ellipsize(session_text, self.small_font, rect.width - 24),
            self.small_font,
            MUTED,
            (rect.x + 12, rect.y + 10),
        )

    def _draw_amount_input(self, surface: pygame.Surface, amount_text: str | None) -> None:
        if amount_text is None:
            return
        rect = self.amount_input_rect()
        pygame.draw.rect(surface, (24, 33, 35), rect, border_radius=8)
        pygame.draw.rect(surface, BUTTON_FILL, rect, width=2, border_radius=8)
        self._draw_text(surface, amount_text, self.small_font, TEXT, rect.center)

    def _draw_card(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        card: Card | None,
        *,
        hidden: bool = False,
    ) -> None:
        if hidden or card is None:
            self._draw_card_shadow(surface, rect)
            self._draw_card_back(surface, rect)
            return

        self._draw_card_shadow(surface, rect)
        self._draw_card_face(surface, rect, card)

    def _draw_card_shadow(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        offset = max(2, rect.width // 18)
        shadow = rect.move(offset, offset + 1)
        pygame.draw.rect(surface, (7, 10, 15), shadow, border_radius=max(5, rect.width // 8))

    def _draw_card_back(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        radius = max(5, rect.width // 8)
        pygame.draw.rect(surface, CARD_BACK, rect, border_radius=radius)
        pygame.draw.rect(
            surface,
            CYAN_ACCENT,
            rect,
            width=max(1, rect.width // 28),
            border_radius=radius,
        )
        inner = rect.inflate(-max(6, rect.width // 7), -max(6, rect.width // 7))
        pygame.draw.rect(surface, (31, 37, 76), inner, border_radius=max(3, radius - 2))
        pygame.draw.line(surface, MAGENTA_ACCENT, inner.topleft, inner.bottomright, width=1)
        pygame.draw.line(surface, CYAN_ACCENT, inner.topright, inner.bottomleft, width=1)
        self._draw_text(
            surface,
            "HL",
            self._card_font(max(9, rect.height // 5), bold=True),
            (211, 221, 234),
            rect.center,
        )

    def _draw_card_face(self, surface: pygame.Surface, rect: pygame.Rect, card: Card) -> None:
        radius = max(5, rect.width // 8)
        suit_color = RED_SUIT if card.suit in {Suit.HEARTS, Suit.DIAMONDS} else BLACK_SUIT
        accent = MAGENTA_ACCENT if card.suit in {Suit.HEARTS, Suit.DIAMONDS} else CYAN_ACCENT
        margin = max(3, rect.width // 12)

        pygame.draw.rect(surface, CARD_FACE, rect, border_radius=radius)
        highlight = rect.inflate(-2, -2)
        pygame.draw.rect(surface, CARD_FACE_HIGHLIGHT, highlight, border_radius=max(3, radius - 1))
        pygame.draw.rect(surface, CARD_EDGE, rect, width=1, border_radius=radius)
        pygame.draw.rect(
            surface, accent, rect.inflate(-2, -2), width=1, border_radius=max(3, radius - 1)
        )
        self._draw_card_corner_lines(surface, rect, accent)

        if rect.width < 52:
            self._draw_text(
                surface, _card_code_label(card), self.card_font, suit_color, rect.center
            )
            return

        rank = _rank_label(card.rank)
        suit_symbol = SUIT_SYMBOLS[card.suit]
        index_font = self._card_font(max(11, int(rect.height * 0.22)), bold=True)
        suit_font = self._card_font(max(8, int(rect.height * 0.16)), bold=True)
        code_font = self._card_font(max(7, int(rect.height * 0.11)), bold=True)

        self._draw_text_left(
            surface,
            rank,
            index_font,
            suit_color,
            (rect.x + margin, rect.y + margin),
        )
        self._draw_text_left(
            surface,
            card.suit.value.upper(),
            code_font,
            suit_color,
            (rect.x + margin + 1, rect.y + margin + index_font.get_height() - 2),
        )

        rank_surface = index_font.render(rank, True, suit_color)
        suit_letter_surface = code_font.render(card.suit.value.upper(), True, suit_color)
        rotated_rank = pygame.transform.rotate(rank_surface, 180)
        surface.blit(
            rotated_rank,
            rotated_rank.get_rect(bottomright=(rect.right - margin, rect.bottom - margin)),
        )
        rotated_suit_letter = pygame.transform.rotate(suit_letter_surface, 180)
        surface.blit(
            rotated_suit_letter,
            rotated_suit_letter.get_rect(
                bottomright=(
                    rect.right - margin - 1,
                    rect.bottom - margin - index_font.get_height() + 2,
                )
            ),
        )

        if rect.width >= 52:
            self._draw_card_pips(surface, rect, card, suit_symbol, suit_color, suit_font)

    def _draw_card_corner_lines(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        accent: Color,
    ) -> None:
        inset = max(5, rect.width // 8)
        cut = max(8, rect.width // 5)
        pygame.draw.line(
            surface,
            accent,
            (rect.x + inset, rect.y + 3),
            (rect.x + cut, rect.y + 3),
            width=1,
        )
        pygame.draw.line(
            surface,
            accent,
            (rect.x + 3, rect.y + inset),
            (rect.x + 3, rect.y + cut),
            width=1,
        )
        pygame.draw.line(
            surface,
            accent,
            (rect.right - cut, rect.bottom - 3),
            (rect.right - inset, rect.bottom - 3),
            width=1,
        )
        pygame.draw.line(
            surface,
            accent,
            (rect.right - 3, rect.bottom - cut),
            (rect.right - 3, rect.bottom - inset),
            width=1,
        )

    def _draw_card_pips(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        card: Card,
        suit_symbol: str,
        color: Color,
        fallback_font: pygame.font.Font,
    ) -> None:
        if card.rank in {Rank.JACK, Rank.QUEEN, Rank.KING}:
            self._draw_face_card_mark(surface, rect, card, suit_symbol, color)
            return

        pip_count = _pip_count(card.rank)
        if pip_count is None:
            self._draw_text(surface, suit_symbol, fallback_font, color, rect.center)
            return

        pip_font = self._card_font(max(14, int(rect.height * 0.21)), bold=True)
        for x_factor, y_factor in _pip_positions(pip_count):
            center = (
                rect.x + int(rect.width * x_factor),
                rect.y + int(rect.height * y_factor),
            )
            self._draw_text(surface, suit_symbol, pip_font, color, center)

    def _draw_face_card_mark(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        card: Card,
        suit_symbol: str,
        color: Color,
    ) -> None:
        center_box = rect.inflate(-rect.width // 3, -rect.height // 3)
        pygame.draw.rect(
            surface, (239, 232, 213), center_box, border_radius=max(3, rect.width // 12)
        )
        pygame.draw.rect(
            surface, GOLD_ACCENT, center_box, width=1, border_radius=max(3, rect.width // 12)
        )
        self._draw_text(
            surface,
            _rank_label(card.rank),
            self._card_font(max(16, int(rect.height * 0.25)), bold=True),
            color,
            (center_box.centerx, center_box.centery - max(6, rect.height // 12)),
        )
        self._draw_text(
            surface,
            suit_symbol,
            self._card_font(max(16, int(rect.height * 0.24)), bold=True),
            color,
            (center_box.centerx, center_box.centery + max(8, rect.height // 10)),
        )

    def _card_font(self, size: int, *, bold: bool) -> pygame.font.Font:
        key = (size, bold)
        font = self._font_cache.get(key)
        if font is None:
            font = pygame.font.SysFont("dejavusans", size, bold=bold)
            self._font_cache[key] = font
        return font

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

    def _draw_text_left(
        self,
        surface: pygame.Surface,
        text: str,
        font: pygame.font.Font,
        color: Color,
        top_left: tuple[int, int],
    ) -> None:
        rendered = font.render(text, True, color)
        surface.blit(rendered, rendered.get_rect(topleft=top_left))

    def _ellipsize(self, text: str, font: pygame.font.Font, max_width: int) -> str:
        if font.size(text)[0] <= max_width:
            return text
        suffix = "..."
        remaining = text
        while remaining and font.size(f"{remaining}{suffix}")[0] > max_width:
            remaining = remaining[:-1]
        return f"{remaining}{suffix}" if remaining else suffix

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

    def action_log_rect(self) -> pygame.Rect:
        width, height = self.size
        rect = pygame.Rect(0, 0, min(260, width - 40), 158)
        rect.topright = (width - 24, max(64, int(height * 0.10)))
        return rect

    def session_rect(self) -> pygame.Rect:
        width, height = self.size
        rect = pygame.Rect(0, 0, min(360, width - 40), 42)
        rect.topleft = (24, max(64, int(height * 0.10)))
        return rect

    def amount_input_rect(self) -> pygame.Rect:
        width, height = self.size
        rect = pygame.Rect(0, 0, min(360, width - 80), 34)
        rect.center = (width // 2, height - 148)
        return rect


def _rank_label(rank: Rank) -> str:
    if rank is Rank.TEN:
        return "10"
    return rank.value


def _card_code_label(card: Card) -> str:
    return f"{card.rank.value}{card.suit.value.upper()}"


def _pip_count(rank: Rank) -> int | None:
    match rank:
        case Rank.ACE:
            return 1
        case Rank.TWO:
            return 2
        case Rank.THREE:
            return 3
        case Rank.FOUR:
            return 4
        case Rank.FIVE:
            return 5
        case Rank.SIX:
            return 6
        case Rank.SEVEN:
            return 7
        case Rank.EIGHT:
            return 8
        case Rank.NINE:
            return 9
        case Rank.TEN:
            return 10
        case _:
            return None


def _pip_positions(count: int) -> tuple[tuple[float, float], ...]:
    left, center, right = 0.36, 0.50, 0.64
    top, high, mid, low, bottom = 0.30, 0.40, 0.52, 0.64, 0.74
    match count:
        case 1:
            return ((center, mid),)
        case 2:
            return ((center, high), (center, low))
        case 3:
            return ((center, top), (center, mid), (center, bottom))
        case 4:
            return ((left, high), (right, high), (left, low), (right, low))
        case 5:
            return (
                (left, high),
                (right, high),
                (center, mid),
                (left, low),
                (right, low),
            )
        case 6:
            return (
                (left, top),
                (right, top),
                (left, mid),
                (right, mid),
                (left, bottom),
                (right, bottom),
            )
        case 7:
            return (
                (left, top),
                (right, top),
                (center, high),
                (left, mid),
                (right, mid),
                (left, bottom),
                (right, bottom),
            )
        case 8:
            return (
                (left, top),
                (right, top),
                (center, high),
                (left, mid),
                (right, mid),
                (center, low),
                (left, bottom),
                (right, bottom),
            )
        case 9:
            return (
                (left, top),
                (right, top),
                (left, high),
                (right, high),
                (center, mid),
                (left, low),
                (right, low),
                (left, bottom),
                (right, bottom),
            )
        case 10:
            return (
                (left, top),
                (right, top),
                (center, 0.36),
                (left, high),
                (right, high),
                (left, low),
                (right, low),
                (center, 0.68),
                (left, bottom),
                (right, bottom),
            )
        case _:
            return ()


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
