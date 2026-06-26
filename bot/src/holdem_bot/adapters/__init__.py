"""Concrete bot adapters."""

from holdem_bot.adapters.in_process import (
    ActionCallbackAutomator,
    StateCapture,
    StateRecognizer,
)

__all__ = [
    "ActionCallbackAutomator",
    "StateCapture",
    "StateRecognizer",
]
