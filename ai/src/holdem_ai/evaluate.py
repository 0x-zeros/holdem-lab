"""Head-to-head evaluation utilities for local AI policies."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from holdem_engine import HoldemConfig, HoldemEnv

from holdem_ai.profiles import PROFILE_NAMES, PolicyProfile, profile_from_name

__all__ = [
    "EvaluationMatrixResult",
    "EvaluationResult",
    "FieldReport",
    "MatchReport",
    "PolicyProfile",
    "ProfileStats",
    "evaluate_field",
    "evaluate_heads_up",
    "evaluate_match",
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


@dataclass(frozen=True, slots=True)
class MatchReport:
    """A variance-reduced heads-up match: ``focal`` vs ``opponent``.

    Each *pair* plays the same deck twice with the seats swapped (common random
    numbers / "duplicate poker"), so card luck cancels between the two halves and
    the bb/100 estimate is far tighter than naive play. The 95% interval is a
    bootstrap over the per-pair nets; ``by_position`` splits the focal result into
    the ``pairs`` button hands and ``pairs`` big-blind hands it played.
    """

    focal: str
    opponent: str
    pairs: int
    hands: int
    seed: int
    starting_stack: int
    small_blind: int
    big_blind: int
    focal_chips: int
    bb_per_100: float
    ci_low: float
    ci_high: float
    by_position: Mapping[str, dict[str, object]]
    focal_actions: Mapping[str, int]
    focal_reasons: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "focal": self.focal,
            "opponent": self.opponent,
            "pairs": self.pairs,
            "hands": self.hands,
            "seed": self.seed,
            "starting_stack": self.starting_stack,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "focal_chips": self.focal_chips,
            "bb_per_100": self.bb_per_100,
            "ci95_low": self.ci_low,
            "ci95_high": self.ci_high,
            "by_position": dict(self.by_position),
            "focal_actions": dict(sorted(self.focal_actions.items())),
            "focal_reasons": dict(sorted(self.focal_reasons.items())),
        }


def _play_hand(
    config: HoldemConfig,
    profile_by_seat: Mapping[int, PolicyProfile],
    *,
    deck_seed: int,
    max_steps: int,
) -> tuple[Mapping[int, int], dict[int, list[tuple[str, str]]]]:
    """Play one hand to completion; return per-seat payoffs and (action, reason) logs."""
    env = HoldemEnv(config)
    state = env.reset(seed=deck_seed)
    moves: dict[int, list[tuple[str, str]]] = {seat: [] for seat in profile_by_seat}
    steps = 0
    while state.current_seat is not None:
        if steps >= max_steps:
            raise RuntimeError(f"hand {config.hand_id} exceeded {max_steps} steps")
        seat = state.current_seat
        decision = profile_by_seat[seat].policy.explain(env.observe(seat=seat))
        moves[seat].append((decision.action.action_type.value, decision.reason))
        state = env.step(decision.action).observation
        steps += 1
    return env.facade.payoffs, moves


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
        payoffs, moves = _play_hand(
            config, profile_by_seat, deck_seed=seed + hand_index, max_steps=max_steps_per_hand
        )
        for seat, profile in profile_by_seat.items():
            profile_stats = stats[profile.name]
            for action_value, reason in moves[seat]:
                profile_stats.actions[action_value] += 1
                profile_stats.reasons[reason] += 1
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


def evaluate_match(
    focal: PolicyProfile,
    opponent: PolicyProfile,
    *,
    pairs: int = 1000,
    seed: int = 1,
    starting_stack: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
    bootstrap: int = 2000,
    confidence: float = 0.95,
    max_steps_per_hand: int = 300,
) -> MatchReport:
    """Variance-reduced (CRN) heads-up match with a bootstrap confidence interval.

    Plays ``pairs`` mirrored deck-pairs (``2 * pairs`` hands): the focal profile
    sits on the button for one half of each pair and the big blind for the other,
    over the *same* cards, so most of the dealing luck cancels and the bb/100
    estimate is much tighter than naive alternating play at equal hand count.
    """
    if focal.name == opponent.name:
        raise ValueError("profile names must be different")
    if pairs <= 0:
        raise ValueError("pairs must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    pair_nets: list[int] = []
    actions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    position_chips = {"button": 0, "big_blind": 0}
    for pair_index in range(pairs):
        deck_seed = seed + pair_index
        net = 0
        for focal_seat in (0, 1):  # button (seat 0), then big blind (seat 1), same deck
            profile_by_seat = {focal_seat: focal, 1 - focal_seat: opponent}
            config = HoldemConfig(
                starting_stacks=(starting_stack, starting_stack),
                small_blind=small_blind,
                big_blind=big_blind,
                # No focal_seat in the hand_id: the two mirrored halves of a pair
                # are the SAME physical games (same deck, swapped labels), so they
                # must hash to the SAME policy RNG (equity / decision seeds key on
                # hand_id + current_seat). Including focal_seat desynced them and
                # broke the antithetic (CRN) pairing for equity-driven policies.
                hand_id=f"match-{pair_index + 1}",
                button_seat=0,
                small_blind_seat=0,
                big_blind_seat=1,
            )
            payoffs, moves = _play_hand(
                config, profile_by_seat, deck_seed=deck_seed, max_steps=max_steps_per_hand
            )
            for action_value, reason in moves[focal_seat]:
                actions[action_value] += 1
                reasons[reason] += 1
            payoff = payoffs[focal_seat]
            net += payoff
            position_chips["button" if focal_seat == 0 else "big_blind"] += payoff
        pair_nets.append(net)

    hands = 2 * pairs
    focal_chips = sum(pair_nets)
    bb_per_100 = focal_chips / big_blind / hands * 100.0
    rng = random.Random(seed ^ 0x9E3779B9)
    ci_low, ci_high = _bootstrap_ci(
        pair_nets, samples=bootstrap, confidence=confidence, rng=rng, big_blind=big_blind
    )
    by_position = {
        position: _position_slice(chips, pairs, big_blind)
        for position, chips in position_chips.items()
    }
    return MatchReport(
        focal=focal.name,
        opponent=opponent.name,
        pairs=pairs,
        hands=hands,
        seed=seed,
        starting_stack=starting_stack,
        small_blind=small_blind,
        big_blind=big_blind,
        focal_chips=focal_chips,
        bb_per_100=bb_per_100,
        ci_low=ci_low,
        ci_high=ci_high,
        by_position=by_position,
        focal_actions=dict(actions),
        focal_reasons=dict(reasons),
    )


@dataclass(frozen=True, slots=True)
class FieldReport:
    """A focal policy versus a homogeneous ``seats``-handed field of one opponent.

    Each *deck* is played ``seats`` times with the focal rotated through every
    seat (the multiway analogue of duplicate poker), so the focal occupies every
    hole-card slot once per deck and most of the dealing luck cancels. The opponent
    fills the remaining seats; it must be a *stateless* policy because one instance
    is shared across those seats. The interval is a bootstrap over per-deck nets.
    """

    focal: str
    opponent: str
    seats: int
    decks: int
    hands: int
    seed: int
    starting_stack: int
    small_blind: int
    big_blind: int
    focal_chips: int
    bb_per_100: float
    ci_low: float
    ci_high: float
    focal_actions: Mapping[str, int]
    focal_reasons: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "focal": self.focal,
            "opponent": self.opponent,
            "seats": self.seats,
            "decks": self.decks,
            "hands": self.hands,
            "seed": self.seed,
            "starting_stack": self.starting_stack,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "focal_chips": self.focal_chips,
            "bb_per_100": self.bb_per_100,
            "ci95_low": self.ci_low,
            "ci95_high": self.ci_high,
            "focal_actions": dict(sorted(self.focal_actions.items())),
            "focal_reasons": dict(sorted(self.focal_reasons.items())),
        }


def evaluate_field(
    focal: PolicyProfile,
    opponent: PolicyProfile,
    *,
    seats: int = 6,
    decks: int = 500,
    seed: int = 1,
    starting_stack: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
    bootstrap: int = 2000,
    confidence: float = 0.95,
    max_steps_per_hand: int = 400,
) -> FieldReport:
    """CRN-rotated ``seats``-handed match: focal versus a homogeneous opponent field.

    Plays ``decks`` decks (``seats * decks`` hands). For each deck the focal sits in
    every seat once over the *same* cards while the opponent fills the rest, so the
    focal's card luck cancels and the bb/100 estimate is far tighter than naive
    play. Returns the focal's bb/100 with a bootstrap interval over per-deck nets.
    """
    if focal.name == opponent.name:
        raise ValueError("profile names must be different")
    if seats < 2:
        raise ValueError("seats must be at least 2")
    if decks <= 0:
        raise ValueError("decks must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    deck_nets: list[int] = []
    actions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for deck_index in range(decks):
        deck_seed = seed + deck_index
        net = 0
        for focal_seat in range(seats):
            profile_by_seat = {
                seat: (focal if seat == focal_seat else opponent) for seat in range(seats)
            }
            button = deck_index % seats
            config = HoldemConfig(
                starting_stacks=tuple(starting_stack for _ in range(seats)),
                small_blind=small_blind,
                big_blind=big_blind,
                # Unique per rotation (not just per deck): a stateful focal's
                # per-hand opponent read must treat each seating as its own hand.
                # CRN cancellation here comes from the shared deck_seed, not the id.
                hand_id=f"field-{deck_index + 1}-{focal_seat}",
                button_seat=button,
                small_blind_seat=(button + 1) % seats,
                big_blind_seat=(button + 2) % seats,
            )
            payoffs, moves = _play_hand(
                config, profile_by_seat, deck_seed=deck_seed, max_steps=max_steps_per_hand
            )
            for action_value, reason in moves[focal_seat]:
                actions[action_value] += 1
                reasons[reason] += 1
            net += payoffs[focal_seat]
        deck_nets.append(net)

    hands = seats * decks
    focal_chips = sum(deck_nets)
    bb_per_100 = focal_chips / big_blind / hands * 100.0
    rng = random.Random(seed ^ 0x9E3779B9)
    ci_low, ci_high = _bootstrap_ci(
        deck_nets,
        samples=bootstrap,
        confidence=confidence,
        rng=rng,
        big_blind=big_blind,
        hands_per_unit=seats,
    )
    return FieldReport(
        focal=focal.name,
        opponent=opponent.name,
        seats=seats,
        decks=decks,
        hands=hands,
        seed=seed,
        starting_stack=starting_stack,
        small_blind=small_blind,
        big_blind=big_blind,
        focal_chips=focal_chips,
        bb_per_100=bb_per_100,
        ci_low=ci_low,
        ci_high=ci_high,
        focal_actions=dict(actions),
        focal_reasons=dict(reasons),
    )


def _position_slice(chips: int, hands: int, big_blind: int) -> dict[str, object]:
    return {
        "hands": hands,
        "chips": chips,
        "bb_per_100": (chips / big_blind / hands * 100.0) if hands else 0.0,
    }


def _bootstrap_ci(
    pair_nets: Sequence[int],
    *,
    samples: int,
    confidence: float,
    rng: random.Random,
    big_blind: int,
    hands_per_unit: int = 2,
) -> tuple[float, float]:
    """Percentile bootstrap CI on bb/100, resampling per-unit nets.

    A "unit" is the block of hands whose net is one resampled datum: a CRN pair is
    two hands (heads-up duplicate), a field deck is ``seats`` hands (focal rotated
    through every seat).
    """
    n = len(pair_nets)
    if n == 0 or samples <= 0:
        return (0.0, 0.0)
    scale = 100.0 / (hands_per_unit * big_blind)  # mean chips per unit -> bb/100
    means: list[float] = []
    for _ in range(samples):
        total = 0
        for _ in range(n):
            total += pair_nets[rng.randrange(n)]
        means.append(total / n * scale)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    low_index = max(0, int(alpha * samples))
    high_index = min(samples - 1, int((1.0 - alpha) * samples))
    return (means[low_index], means[high_index])


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
    parser.add_argument(
        "--match",
        nargs=2,
        metavar=("FOCAL", "OPPONENT"),
        choices=PROFILE_NAMES,
        help="CRN-paired FOCAL vs OPPONENT match with a bootstrap bb/100 interval.",
    )
    parser.add_argument("--pairs", type=int, default=1000, help="CRN deck-pairs for --match.")
    parser.add_argument(
        "--field",
        nargs=2,
        metavar=("FOCAL", "OPPONENT"),
        choices=PROFILE_NAMES,
        help="CRN-rotated FOCAL versus a homogeneous OPPONENT field (6-max default).",
    )
    parser.add_argument("--seats", type=int, default=6, help="Seats at the table for --field.")
    parser.add_argument("--decks", type=int, default=500, help="CRN decks for --field.")
    parser.add_argument(
        "--bootstrap", type=int, default=2000, help="Bootstrap resamples for --match / --field."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.field is not None:
        focal_name, opponent_name = args.field
        field_result = evaluate_field(
            profile_from_name(focal_name),
            profile_from_name(opponent_name),
            seats=args.seats,
            decks=args.decks,
            seed=args.seed,
            starting_stack=args.starting_stack,
            small_blind=args.small_blind,
            big_blind=args.big_blind,
            bootstrap=args.bootstrap,
        )
        print(json.dumps(field_result.to_dict(), indent=2, sort_keys=True))
        return

    if args.match is not None:
        focal_name, opponent_name = args.match
        match_result = evaluate_match(
            profile_from_name(focal_name),
            profile_from_name(opponent_name),
            pairs=args.pairs,
            seed=args.seed,
            starting_stack=args.starting_stack,
            small_blind=args.small_blind,
            big_blind=args.big_blind,
            bootstrap=args.bootstrap,
        )
        print(json.dumps(match_result.to_dict(), indent=2, sort_keys=True))
        return

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
