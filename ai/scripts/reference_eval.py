"""Regenerate the absolute reference-strength table (docs/ai-reference-eval.md).

Runs a CRN-paired, bootstrap-CI match grid of our focal policies against the
reference opponents at several stack depths, using ``holdem_ai.evaluate``. Every
number is reproducible from the fixed seed. Emits a Markdown report to stdout.

    py ai/scripts/reference_eval.py --pairs 2500 --stacks 20,50,200 > docs/ai-reference-eval.md
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import cast

from holdem_ai.evaluate import evaluate_match
from holdem_ai.profiles import profile_from_name

DEFAULT_FOCALS = ("pushfold", "current")
DEFAULT_OPPONENTS = ("random", "call_station", "rock", "maniac", "tag", "three_bet_jammer")


@dataclass(frozen=True, slots=True)
class Cell:
    focal: str
    opponent: str
    stack_bb: float
    bb_per_100: float
    ci_low: float
    ci_high: float
    button_bb_per_100: float
    big_blind_bb_per_100: float

    @property
    def significant(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0


def run_grid(
    focals: tuple[str, ...],
    opponents: tuple[str, ...],
    stacks: tuple[int, ...],
    *,
    pairs: int,
    seed: int,
    bootstrap: int,
    big_blind: int,
) -> list[Cell]:
    cells: list[Cell] = []
    for stack in stacks:
        for focal in focals:
            for opponent in opponents:
                if focal == opponent:
                    continue
                report = evaluate_match(
                    profile_from_name(focal),
                    profile_from_name(opponent),
                    pairs=pairs,
                    seed=seed,
                    starting_stack=stack,
                    small_blind=max(1, big_blind // 2),
                    big_blind=big_blind,
                    bootstrap=bootstrap,
                )
                button = cast(float, report.by_position["button"]["bb_per_100"])
                big = cast(float, report.by_position["big_blind"]["bb_per_100"])
                cells.append(
                    Cell(
                        focal=focal,
                        opponent=opponent,
                        stack_bb=stack / big_blind,
                        bb_per_100=report.bb_per_100,
                        ci_low=report.ci_low,
                        ci_high=report.ci_high,
                        button_bb_per_100=button,
                        big_blind_bb_per_100=big,
                    )
                )
    return cells


def format_markdown(cells: list[Cell], *, pairs: int, seed: int, bootstrap: int) -> str:
    hands = 2 * pairs
    lines = [
        "# AI reference-strength table",
        "",
        "Absolute yardstick: each focal policy vs the reference opponents, measured",
        "with the CRN-paired bootstrap harness (`holdem_ai.evaluate.evaluate_match`).",
        "",
        f"- **{pairs} CRN deck-pairs = {hands} hands** per matchup (same cards played",
        "  both seat orderings, so dealing luck cancels).",
        f"- 95% CI is a **percentile bootstrap** ({bootstrap} resamples) over per-pair nets.",
        f"- Deterministic (seed {seed}). Regenerate with `ai/scripts/reference_eval.py`.",
        "- A row is **bold** when the 95% CI excludes 0 (a statistically clear result).",
        "",
        "All numbers are **bb/100 from the focal policy's perspective** (higher = better).",
        "",
    ]
    by_stack: dict[float, list[Cell]] = {}
    for cell in cells:
        by_stack.setdefault(cell.stack_bb, []).append(cell)
    for stack_bb in sorted(by_stack, reverse=True):
        lines.append(f"## {stack_bb:.0f}bb effective")
        lines.append("")
        lines.append("| focal | opponent | bb/100 | 95% CI | button | big blind |")
        lines.append("|---|---|--:|:--:|--:|--:|")
        for cell in by_stack[stack_bb]:
            label = f"`{cell.focal}`"
            opponent = f"`{cell.opponent}`"
            value = f"{cell.bb_per_100:+.1f}"
            interval = f"[{cell.ci_low:+.1f}, {cell.ci_high:+.1f}]"
            if cell.significant:
                value = f"**{value}**"
            lines.append(
                f"| {label} | {opponent} | {value} | {interval} "
                f"| {cell.button_bb_per_100:+.1f} | {cell.big_blind_bb_per_100:+.1f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--big-blind", type=int, default=2)
    parser.add_argument("--stacks", default="20,50,200", help="Comma-separated starting stacks.")
    parser.add_argument("--focals", default=",".join(DEFAULT_FOCALS))
    parser.add_argument("--opponents", default=",".join(DEFAULT_OPPONENTS))
    args = parser.parse_args()

    stacks = tuple(int(x) for x in args.stacks.split(","))
    focals = tuple(args.focals.split(","))
    opponents = tuple(args.opponents.split(","))
    cells = run_grid(
        focals,
        opponents,
        stacks,
        pairs=args.pairs,
        seed=args.seed,
        bootstrap=args.bootstrap,
        big_blind=args.big_blind,
    )
    print(format_markdown(cells, pairs=args.pairs, seed=args.seed, bootstrap=args.bootstrap))


if __name__ == "__main__":
    main()
