"""Named local AI policy profiles shared by game and evaluation tools."""

from __future__ import annotations

from dataclasses import dataclass

from holdem_ai.baselines import (
    AggressivePolicy,
    CallStationPolicy,
    Policy,
    RandomPolicy,
    RockPolicy,
)
from holdem_ai.heuristic import HeuristicConfig, HeuristicPolicy

#: Heuristic-family profiles (same policy, different thresholds).
HEURISTIC_PROFILE_NAMES = ("current", "no_equity", "tight", "loose")
#: Deterministic reference opponents that act as an absolute yardstick.
REFERENCE_PROFILE_NAMES = ("random", "call_station", "rock", "maniac")
PROFILE_NAMES = HEURISTIC_PROFILE_NAMES + REFERENCE_PROFILE_NAMES


@dataclass(frozen=True, slots=True)
class PolicyProfile:
    name: str
    policy: Policy


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
        case "random":
            return PolicyProfile("random", RandomPolicy(seed=20260627))
        case "call_station":
            return PolicyProfile("call_station", CallStationPolicy())
        case "rock":
            return PolicyProfile("rock", RockPolicy())
        case "maniac":
            return PolicyProfile("maniac", AggressivePolicy())
        case _:
            raise ValueError(f"unknown profile: {name}")
