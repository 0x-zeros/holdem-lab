"""pygame host application package."""

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from holdem_game.app import HoldemGameApp, build_arg_parser, main
from holdem_game.fixtures import build_table_annotation, write_pygame_fixture
from holdem_game.table_view import ActionButton, TableView

__all__ = [
    "ActionButton",
    "HoldemGameApp",
    "TableView",
    "build_arg_parser",
    "build_table_annotation",
    "main",
    "write_pygame_fixture",
]
