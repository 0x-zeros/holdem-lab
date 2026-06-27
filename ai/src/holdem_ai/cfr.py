"""CFR training and exploitability evaluation via OpenSpiel.

OpenSpiel owns the game tree and the CFR / exploitability implementations; this
module is a thin, tested harness + CLI that validates the
``solve -> average policy -> exploitability`` loop on small two-player zero-sum
games (``kuhn_poker`` / ``leduc_poker``) before scaling to an abstracted heads-up
no-limit game. See ``docs/ai-strength.md`` (S2).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass

import pyspiel  # type: ignore[import-not-found]
from open_spiel.python.algorithms import cfr, exploitability  # type: ignore[import-untyped]

__all__ = [
    "CFR_VARIANTS",
    "CFRCheckpoint",
    "CFRResult",
    "build_arg_parser",
    "main",
    "nolimit_holdem_abstraction",
    "train_cfr",
]

CFR_VARIANTS = ("cfr", "cfr_plus")
DEFAULT_GAME = "leduc_poker"
#: CLI shortcut name for a small, CFR-solvable no-limit hold'em abstraction.
NLHE_SMALL = "nlhe-small"


def nolimit_holdem_abstraction(
    *,
    suits: int = 2,
    ranks: int = 4,
    hole_cards: int = 1,
    stack: int = 6,
) -> str:
    """Build an OpenSpiel ``universal_poker`` string for a small no-limit hold'em
    abstraction: two betting rounds (one private + one board card) with ``fcpa``
    betting (fold / call / pot / all-in).

    Full HUNL is intractable for tabular CFR, so a reduced deck and short stack
    keep it solvable while staying genuine *no-limit* hold'em (unlike the
    limit-betting ``leduc_poker``). Scale these up — or move to MCCFR — toward the
    real game. The defaults solve to <0.02 exploitability in a couple of seconds.
    """
    return (
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=2,blind=1 2,"
        f"firstPlayer=2 1,numSuits={suits},numRanks={ranks},numHoleCards={hole_cards},"
        f"numBoardCards=0 1,stack={stack} {stack},bettingAbstraction=fcpa)"
    )


@dataclass(frozen=True, slots=True)
class CFRCheckpoint:
    iteration: int
    exploitability: float


@dataclass(frozen=True, slots=True)
class CFRResult:
    game: str
    variant: str
    iterations: int
    checkpoints: tuple[CFRCheckpoint, ...]

    @property
    def final_exploitability(self) -> float:
        if not self.checkpoints:
            raise ValueError("no checkpoints recorded")
        return self.checkpoints[-1].exploitability

    def to_dict(self) -> dict[str, object]:
        return {
            "game": self.game,
            "variant": self.variant,
            "iterations": self.iterations,
            "final_exploitability": self.final_exploitability,
            "checkpoints": [
                {"iteration": point.iteration, "exploitability": point.exploitability}
                for point in self.checkpoints
            ],
        }


def train_cfr(
    game_name: str = DEFAULT_GAME,
    *,
    iterations: int = 200,
    variant: str = "cfr_plus",
    eval_every: int = 50,
) -> CFRResult:
    """Run CFR on a two-player zero-sum OpenSpiel game, tracking exploitability.

    Exploitability is the average best-response value against the running average
    policy; for these games CFR drives it toward zero (the Nash equilibrium).
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if variant not in CFR_VARIANTS:
        raise ValueError(f"unknown CFR variant: {variant!r} (choose from {CFR_VARIANTS})")

    game = pyspiel.load_game(game_name)
    if game.num_players() != 2:
        raise ValueError("CFR harness only supports two-player zero-sum games")

    solver = cfr.CFRPlusSolver(game) if variant == "cfr_plus" else cfr.CFRSolver(game)
    checkpoints: list[CFRCheckpoint] = []
    for iteration in range(1, iterations + 1):
        solver.evaluate_and_update_policy()
        if iteration == iterations or (eval_every > 0 and iteration % eval_every == 0):
            value = float(exploitability.exploitability(game, solver.average_policy()))
            checkpoints.append(CFRCheckpoint(iteration=iteration, exploitability=value))

    return CFRResult(
        game=game_name,
        variant=variant,
        iterations=iterations,
        checkpoints=tuple(checkpoints),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CFR and report exploitability.")
    parser.add_argument(
        "--game",
        default=DEFAULT_GAME,
        help=(
            f"OpenSpiel game name / string, or {NLHE_SMALL!r} for the small "
            "no-limit hold'em abstraction"
        ),
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--variant", default="cfr_plus", choices=CFR_VARIANTS)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="record exploitability every N iterations (0 = only at the end)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    game = nolimit_holdem_abstraction() if args.game == NLHE_SMALL else args.game
    result = train_cfr(
        game,
        iterations=args.iterations,
        variant=args.variant,
        eval_every=args.eval_every,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
