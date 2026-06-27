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
from holdem_bot.adapters.poker_legends_host import (
    MacOSScreenCapture,
    PokerLegendsClickPlan,
    PokerLegendsDryRunAutomator,
    PokerLegendsLayoutClickPlanner,
)

__all__ = [
    "ActionCallbackAutomator",
    "MacOSScreenCapture",
    "PokerLegendsClickPlan",
    "PokerLegendsDryRunAutomator",
    "PokerLegendsLayoutClickPlanner",
    "PokerLegendsScreenStateRecognizer",
    "PokerLegendsTableRecognizer",
    "StateCapture",
    "StateRecognizer",
]
