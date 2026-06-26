"""Concrete bot adapters."""

from holdem_bot.adapters.in_process import (
    ActionCallbackAutomator,
    StateCapture,
    StateRecognizer,
)
from holdem_bot.adapters.poker_legends import PokerLegendsScreenStateRecognizer

__all__ = [
    "ActionCallbackAutomator",
    "PokerLegendsScreenStateRecognizer",
    "StateCapture",
    "StateRecognizer",
]
