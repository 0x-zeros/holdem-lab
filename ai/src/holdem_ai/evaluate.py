"""Head-to-head evaluation utilities for local AI policies."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from holdem_engine import HoldemConfig, HoldemEnv

from holdem_ai.heuristic import HeuristicConfig, HeuristicPolicy


@dataclass(frozen=True, slots=True)
class PolicyProfile:
    name: str
    policy: HeuristicPolicy


@dataclass(slots=True)
class ProfileStats:
    hands: int = 0
    chips: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    actions: Counter[str] = field(default_factory=Counter)
    reasons: Counter[str] = field(default_factory=Counter)

    def to_dict(self, *, big_blind: int) -> dict[str, object]:
        return {
            "hands": self.hands,
            "chips": self.chips,
            "bb_per_100": (self.chips / big_blind / self.hands * 100.0) if self.hands else 0.0,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "actions": dict(sorted(self.actions.items())),
            "reasons": dict(sorted(self.reasons.items())),
        }


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    hands: int
    seed: int
    starting_stack: int
    small_blind: int
    big_blind: int
    profiles: Mapping[str, dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "hands": self.hands,
            "seed": self.seed,
            "starting_stack": self.starting_stack,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "profiles": dict(self.profiles),
        }


def evaluate_heads_up(
    profile_a: PolicyProfile,
    profile_b: PolicyProfile,
    *,
    hands: int = 100,
    seed: int = 1,
    starting_stack: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
    max_steps_per_hand: int = 300,
) -> EvaluationResult:
    if hands <= 0:
        raise ValueError("hands must be positive")
    if max_steps_per_hand <= 0:
        raise ValueError("max_steps_per_hand must be positive")
    if profile_a.name == profile_b.name:
        raise ValueError("profile names must be different")

    stats = {profile_a.name: ProfileStats(), profile_b.name: ProfileStats()}
    profiles = (profile_a, profile_b)

    for hand_index in range(hands):
        button_seat = hand_index % 2
        profile_by_seat = {
            0: profiles[hand_index % 2],
            1: profiles[(hand_index + 1) % 2],
        }
        config = HoldemConfig(
            starting_stacks=(starting_stack, starting_stack),
            small_blind=small_blind,
            big_blind=big_blind,
            hand_id=f"eval-{hand_index + 1}",
            button_seat=button_seat,
            small_blind_seat=button_seat,
            big_blind_seat=(button_seat + 1) % 2,
        )
        env = HoldemEnv(config)
        state = env.reset(seed=seed + hand_index)
        steps = 0
        while state.current_seat is not None:
            if steps >= max_steps_per_hand:
                raise RuntimeError(f"hand {hand_index + 1} exceeded {max_steps_per_hand} steps")
            seat = state.current_seat
            profile = profile_by_seat[seat]
            decision = profile.policy.explain(env.observe(seat=seat))
            stats[profile.name].actions[decision.action.action_type.value] += 1
            stats[profile.name].reasons[decision.reason] += 1
            state = env.step(decision.action).observation
            steps += 1

        payoffs = env.facade.payoffs
        for seat, profile in profile_by_seat.items():
            profile_stats = stats[profile.name]
            payoff = payoffs[seat]
            profile_stats.hands += 1
            profile_stats.chips += payoff
            if payoff > 0:
                profile_stats.wins += 1
            elif payoff < 0:
                profile_stats.losses += 1
            else:
                profile_stats.ties += 1

    return EvaluationResult(
        hands=hands,
        seed=seed,
        starting_stack=starting_stack,
        small_blind=small_blind,
        big_blind=big_blind,
        profiles={name: item.to_dict(big_blind=big_blind) for name, item in stats.items()},
    )


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate two local heads-up AI profiles.")
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--starting-stack", type=int, default=200)
    parser.add_argument("--small-blind", type=int, default=1)
    parser.add_argument("--big-blind", type=int, default=2)
    profile_choices = ("current", "no_equity", "tight", "loose")
    parser.add_argument("--profile-a", default="current", choices=profile_choices)
    parser.add_argument("--profile-b", default="no_equity", choices=profile_choices)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = evaluate_heads_up(
        profile_from_name(args.profile_a),
        profile_from_name(args.profile_b),
        hands=args.hands,
        seed=args.seed,
        starting_stack=args.starting_stack,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
