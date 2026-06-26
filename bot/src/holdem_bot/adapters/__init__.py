"""Concrete bot adapters."""

from holdem_bot.adapters.in_process import (
    ActionCallbackAutomator,
    StateCapture,
    StateRecognizer,
)
from holdem_bot.adapters.poker_legends import (
    PokerLegendsScreenStateRecognizer,
    PokerLegendsTableRecognizer,
)

__all__ = [
    "ActionCallbackAutomator",
    "PokerLegendsScreenStateRecognizer",
    "PokerLegendsTableRecognizer",
    "StateCapture",
    "StateRecognizer",
]
