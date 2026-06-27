"""Generate annotated screenshots for CV/OCR development."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame
from holdem_bot.vision import (
    AnnotatedButton,
    AnnotatedCard,
    AnnotatedSeat,
    AnnotatedText,
    ScreenRect,
    TableAnnotation,
)
from holdem_common import Card, GameState, PlayerState
from holdem_engine import HoldemConfig

from holdem_game.app import HoldemGameApp

DEFAULT_OUTPUT_DIR = Path("artifacts/vision-fixtures")
DEFAULT_DECK = (
    "As",
    "Ks",
    "Qs",
    "Ah",
    "Kh",
    "Qh",
    "2c",
    "7d",
    "9s",
    "Jc",
    "3h",
    "4d",
)


def write_pygame_fixture(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    stem: str = "pygame-preflop",
    size: tuple[int, int] = (1180, 760),
) -> TableAnnotation:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_name = f"{stem}.png"
    json_path = output_path / f"{stem}.json"
    image_path = output_path / image_name

    app = HoldemGameApp(
        HoldemConfig(
            starting_stacks=(200, 200, 200),
            deck=tuple(Card.from_code(code) for code in DEFAULT_DECK),
        ),
        size=size,
    )
    try:
        app.action_log = []
        app.draw()
        pygame.image.save(app.screen, image_path)
        annotation = build_table_annotation(app, image=image_name, source="holdem_game")
        annotation.write_json(json_path)
        return annotation
    finally:
        app.close()


def build_table_annotation(
    app: HoldemGameApp,
    *,
    image: str,
    source: str,
) -> TableAnnotation:
    state = app.visible_state()
    return TableAnnotation(
        schema_version=1,
        source=source,
        image=image,
        width=app.size[0],
        height=app.size[1],
        hand_id=state.hand_id,
        street=state.street.value,
        current_seat=state.current_seat,
        board=_annotate_board(app, state),
        seats=tuple(_annotate_seat(app, state, player) for player in state.players),
        texts=_annotate_texts(app, state),
        buttons=tuple(
            AnnotatedButton(
                label=button.label,
                rect=_screen_rect(button.rect),
                action_type=None if button.action is None else button.action.action_type.value,
                command=button.command,
            )
            for button in app.buttons
        ),
        metadata={
            "human_seat": app.human_seat,
            "bot_seat": app.bot_seat,
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture an annotated pygame table fixture.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for PNG and JSON fixture files.",
    )
    parser.add_argument("--stem", default="pygame-preflop", help="Fixture file stem.")
    parser.add_argument("--width", type=int, default=1180, help="Screenshot width.")
    parser.add_argument("--height", type=int, default=760, help="Screenshot height.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    write_pygame_fixture(args.output_dir, stem=args.stem, size=(args.width, args.height))


def _annotate_board(app: HoldemGameApp, state: GameState) -> tuple[AnnotatedCard, ...]:
    annotations: list[AnnotatedCard] = []
    for index, rect in enumerate(app.view.board_card_rects()):
        card = state.board[index] if index < len(state.board) else None
        annotations.append(
            AnnotatedCard(
                slot=f"board_{index}",
                rect=_screen_rect(rect),
                card=None if card is None else card.code,
                visible=card is not None,
            ),
        )
    return tuple(annotations)


def _annotate_seat(
    app: HoldemGameApp,
    state: GameState,
    player: PlayerState,
) -> AnnotatedSeat:
    panel = app.view.player_panel_rect(
        player.seat,
        player_count=len(state.players),
        human_seat=app.human_seat,
    )
    card_rects = app.view.player_hole_card_rects(panel)
    cards = tuple(
        AnnotatedCard(
            slot=f"seat_{player.seat}_hole_{index}",
            rect=_screen_rect(rect),
            card=player.hole_cards[index].code if index < len(player.hole_cards) else None,
            visible=index < len(player.hole_cards),
        )
        for index, rect in enumerate(card_rects)
    )
    return AnnotatedSeat(
        seat=player.seat,
        rect=_screen_rect(panel),
        stack=player.stack,
        committed=player.committed,
        active=player.active,
        current=state.current_seat == player.seat,
        dealer=player.dealer,
        small_blind=player.small_blind,
        big_blind=player.big_blind,
        hole_cards=cards,
    )


def _annotate_texts(app: HoldemGameApp, state: GameState) -> tuple[AnnotatedText, ...]:
    status_text = (
        f"{state.street.value.upper()}  To call {state.to_call}  Min raise {state.min_raise}"
    )
    texts = [
        AnnotatedText(
            name="street_status",
            rect=_screen_rect(app.view.title_text_rect()),
            value=status_text,
            kind="status",
        ),
        AnnotatedText(
            name="pot",
            rect=_screen_rect(app.view.pot_text_rect()),
            value=str(state.pot_total),
            kind="chips",
        ),
        AnnotatedText(
            name="message",
            rect=_screen_rect(app.view.message_text_rect()),
            value=app.message,
            kind="message",
        ),
    ]
    for player in state.players:
        panel = app.view.player_panel_rect(
            player.seat,
            player_count=len(state.players),
            human_seat=app.human_seat,
        )
        texts.append(
            AnnotatedText(
                name=f"seat_{player.seat}_stack_committed",
                rect=_screen_rect(pygame.Rect(panel.x + 8, panel.y + 28, panel.width - 16, 26)),
                value=f"Stack {player.stack}  Bet {player.committed}",
                kind="chips",
            ),
        )
    return tuple(texts)


def _screen_rect(rect: pygame.Rect) -> ScreenRect:
    return ScreenRect(x=rect.x, y=rect.y, width=rect.width, height=rect.height)
