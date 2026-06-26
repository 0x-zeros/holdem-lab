"""Automation abstractions for applying bot actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from holdem_common import Action, GameState


@dataclass(frozen=True, slots=True)
class ActionRecord:
    action: Action
    state: GameState


class Automator(Protocol):
    def perform(self, action: Action, state: GameState) -> None:
        """Apply one selected action to the target application."""
        ...
