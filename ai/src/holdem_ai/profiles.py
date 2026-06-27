"""Named local AI policy profiles shared by game and evaluation tools."""

from __future__ import annotations

from dataclasses import dataclass

from holdem_ai.heuristic import HeuristicConfig, HeuristicPolicy

PROFILE_NAMES = ("current", "no_equity", "tight", "loose")


@dataclass(frozen=True, slots=True)
class PolicyProfile:
    name: str
    policy: HeuristicPolicy


def profile_from_name(name: str) -> PolicyProfile:
    match name:
        case "current":
            return PolicyProfile("current", HeuristicPolicy())
        case "no_equity":
            return PolicyProfile(
                "no_equity",
                HeuristicPolicy(HeuristicConfig(equity_samples=0, equity_weight=0.0)),
            )
        case "tight":
            return PolicyProfile(
                "tight",
                HeuristicPolicy(
                    HeuristicConfig(
                        value_raise_threshold=0.84,
                        protection_bet_threshold=0.70,
                        semi_bluff_threshold=0.66,
                        continue_threshold=0.52,
                        marginal_threshold=0.42,
                    )
                ),
            )
        case "loose":
            return PolicyProfile(
                "loose",
                HeuristicPolicy(
                    HeuristicConfig(
                        value_raise_threshold=0.72,
                        protection_bet_threshold=0.58,
                        semi_bluff_threshold=0.52,
                        continue_threshold=0.40,
                        marginal_threshold=0.30,
                        marginal_call_price_fraction=0.28,
                    )
                ),
            )
        case _:
            raise ValueError(f"unknown profile: {name}")
