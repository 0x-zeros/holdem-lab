"""Head-to-head evaluation utilities for local AI policies."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from holdem_engine import HoldemConfig, HoldemEnv

from holdem_ai.profiles import PROFILE_NAMES, PolicyProfile, profile_from_name

__all__ = [
    "EvaluationMatrixResult",
    "EvaluationResult",
    "PolicyProfile",
    "ProfileStats",
    "evaluate_heads_up",
    "evaluate_profile_matrix",
    "main",
    "profile_from_name",
]


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


@dataclass(frozen=True, slots=True)
class EvaluationMatrixResult:
    hands_per_pairing: int
    seed: int
    starting_stack: int
    small_blind: int
    big_blind: int
    profile_names: tuple[str, ...]
    pairings: Mapping[str, dict[str, object]]
    leaderboard: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "hands_per_pairing": self.hands_per_pairing,
            "seed": self.seed,
            "starting_stack": self.starting_stack,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "profile_names": list(self.profile_names),
            "pairings": dict(self.pairings),
            "leaderboard": list(self.leaderboard),
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


def evaluate_profile_matrix(
    profile_names: Sequence[str],
    *,
    hands: int = 100,
    seed: int = 1,
    starting_stack: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
) -> EvaluationMatrixResult:
    names = tuple(profile_names)
    if len(names) < 2:
        raise ValueError("at least two profiles are required")
    if len(set(names)) != len(names):
        raise ValueError("profile names must be unique")

    aggregate = {
        name: {"hands": 0, "chips": 0, "wins": 0, "losses": 0, "ties": 0} for name in names
    }
    pairings: dict[str, dict[str, object]] = {}
    pairing_index = 0
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            pairing_index += 1
            result = evaluate_heads_up(
                profile_from_name(left_name),
                profile_from_name(right_name),
                hands=hands,
                seed=seed + pairing_index * 10_000,
                starting_stack=starting_stack,
                small_blind=small_blind,
                big_blind=big_blind,
            )
            report = result.to_dict()
            key = f"{left_name}_vs_{right_name}"
            pairings[key] = report
            profiles = report["profiles"]
            if not isinstance(profiles, Mapping):
                raise TypeError("pairing report has invalid profiles")
            for name in (left_name, right_name):
                profile_report = profiles[name]
                if not isinstance(profile_report, Mapping):
                    raise TypeError("pairing profile report is invalid")
                stats = aggregate[name]
                stats["hands"] += int(profile_report["hands"])
                stats["chips"] += int(profile_report["chips"])
                stats["wins"] += int(profile_report["wins"])
                stats["losses"] += int(profile_report["losses"])
                stats["ties"] += int(profile_report["ties"])

    leaderboard = tuple(
        sorted(
            (
                {
                    "profile": name,
                    **stats,
                    "bb_per_100": (stats["chips"] / big_blind / stats["hands"] * 100.0)
                    if stats["hands"]
                    else 0.0,
                }
                for name, stats in aggregate.items()
            ),
            key=_leaderboard_score,
            reverse=True,
        )
    )
    return EvaluationMatrixResult(
        hands_per_pairing=hands,
        seed=seed,
        starting_stack=starting_stack,
        small_blind=small_blind,
        big_blind=big_blind,
        profile_names=names,
        pairings=pairings,
        leaderboard=leaderboard,
    )


def _leaderboard_score(item: Mapping[str, object]) -> float:
    value = item["bb_per_100"]
    if not isinstance(value, int | float):
        raise TypeError("leaderboard entry has invalid bb_per_100")
    return float(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate two local heads-up AI profiles.")
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--starting-stack", type=int, default=200)
    parser.add_argument("--small-blind", type=int, default=1)
    parser.add_argument("--big-blind", type=int, default=2)
    parser.add_argument("--profile-a", default="current", choices=PROFILE_NAMES)
    parser.add_argument("--profile-b", default="no_equity", choices=PROFILE_NAMES)
    parser.add_argument(
        "--matrix",
        nargs="+",
        choices=PROFILE_NAMES,
        help="Evaluate every pair among the listed profiles and output a leaderboard.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.matrix is not None:
        matrix_result = evaluate_profile_matrix(
            args.matrix,
            hands=args.hands,
            seed=args.seed,
            starting_stack=args.starting_stack,
            small_blind=args.small_blind,
            big_blind=args.big_blind,
        )
        print(json.dumps(matrix_result.to_dict(), indent=2, sort_keys=True))
        return

    heads_up_result = evaluate_heads_up(
        profile_from_name(args.profile_a),
        profile_from_name(args.profile_b),
        hands=args.hands,
        seed=args.seed,
        starting_stack=args.starting_stack,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
    )
    print(json.dumps(heads_up_result.to_dict(), indent=2, sort_keys=True))
