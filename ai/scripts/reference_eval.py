"""Regenerate the absolute reference-strength table (docs/ai-reference-eval.md).

Runs a CRN-paired, bootstrap-CI match grid of our focal policies against the
reference opponents at several stack depths, using ``holdem_ai.evaluate``. Every
number is reproducible from the fixed seed. Per-matchup progress is logged to
stderr; the Markdown report is printed to stdout at the end.

Pairs scale down with depth (deep stacks are slow and high-variance, so they buy
little from extra hands):

    py ai/scripts/reference_eval.py --stacks 20,50,200 --pairs 2500,1500,500 \
        > docs/ai-reference-eval.md
"""

from __future__ import annotations

import argparse
import sys
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
    pairs: int
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
    pairs_by_stack: tuple[int, ...],
    *,
    seed: int,
    bootstrap: int,
    big_blind: int,
) -> list[Cell]:
    jobs = [
        (stack, pairs, focal, opponent)
        for stack, pairs in zip(stacks, pairs_by_stack, strict=True)
        for focal in focals
        for opponent in opponents
        if focal != opponent
    ]
    cells: list[Cell] = []
    for index, (stack, pairs, focal, opponent) in enumerate(jobs, start=1):
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
                pairs=pairs,
                bb_per_100=report.bb_per_100,
                ci_low=report.ci_low,
                ci_high=report.ci_high,
                button_bb_per_100=button,
                big_blind_bb_per_100=big,
            )
        )
        print(
            f"[{index}/{len(jobs)}] {focal} vs {opponent} @{stack / big_blind:.0f}bb: "
            f"{report.bb_per_100:+.1f} bb/100 CI[{report.ci_low:+.1f}, {report.ci_high:+.1f}]",
            file=sys.stderr,
            flush=True,
        )
    return cells


def format_markdown(cells: list[Cell], *, seed: int, bootstrap: int) -> str:
    lines = [
        "# AI reference-strength table",
        "",
        "Absolute yardstick: each focal policy vs the reference opponents, measured",
        "with the CRN-paired bootstrap harness (`holdem_ai.evaluate.evaluate_match`).",
        "",
        "- Each matchup plays **N CRN deck-pairs = 2N hands** (same cards played both",
        "  seat orderings, so dealing luck cancels). N shrinks with depth.",
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
    for stack_bb in sorted(by_stack):
        group = by_stack[stack_bb]
        pairs = group[0].pairs
        lines.append(f"## {stack_bb:.0f}bb effective — {pairs} pairs ({2 * pairs} hands/matchup)")
        lines.append("")
        lines.append("| focal | opponent | bb/100 | 95% CI | button | big blind |")
        lines.append("|---|---|--:|:--:|--:|--:|")
        for cell in group:
            value = f"{cell.bb_per_100:+.1f}"
            if cell.significant:
                value = f"**{value}**"
            lines.append(
                f"| `{cell.focal}` | `{cell.opponent}` | {value} "
                f"| [{cell.ci_low:+.1f}, {cell.ci_high:+.1f}] "
                f"| {cell.button_bb_per_100:+.1f} | {cell.big_blind_bb_per_100:+.1f} |"
            )
        lines.append("")
    return "\n".join(lines)


def _parse_pairs(raw: str, stacks: tuple[int, ...]) -> tuple[int, ...]:
    values = tuple(int(x) for x in raw.split(","))
    if len(values) == 1:
        return values * len(stacks)
    if len(values) != len(stacks):
        raise SystemExit("--pairs must be a single value or match the number of --stacks")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stacks", default="20,50,200", help="Comma-separated starting stacks.")
    parser.add_argument("--pairs", default="2500,1500,500", help="Per-stack (or single) CRN pairs.")
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--bootstrap", type=int, default=1500)
    parser.add_argument("--big-blind", type=int, default=2)
    parser.add_argument("--focals", default=",".join(DEFAULT_FOCALS))
    parser.add_argument("--opponents", default=",".join(DEFAULT_OPPONENTS))
    args = parser.parse_args()

    stacks = tuple(int(x) for x in args.stacks.split(","))
    pairs_by_stack = _parse_pairs(args.pairs, stacks)
    cells = run_grid(
        tuple(args.focals.split(",")),
        tuple(args.opponents.split(",")),
        stacks,
        pairs_by_stack,
        seed=args.seed,
        bootstrap=args.bootstrap,
        big_blind=args.big_blind,
    )
    print(format_markdown(cells, seed=args.seed, bootstrap=args.bootstrap))


if __name__ == "__main__":
    main()
