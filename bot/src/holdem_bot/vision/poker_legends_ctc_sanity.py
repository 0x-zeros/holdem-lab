"""Offline sanity harness for Poker Legends CRNN+CTC OCR experiments."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from holdem_bot.vision.poker_legends_ctc import (
    CtcAlignmentError,
    validate_ctc_time_step_budget,
)
from holdem_bot.vision.poker_legends_number_chars import (
    DEFAULT_CTC_MIN_TIMESTEP_RATIO,
    NUMBER_TEXT_TARGETS,
    NumberSequenceSample,
    NumberTextTarget,
    NumberTorchCtcRecognizer,
    _load_rgb_image,
    _load_rows,
    _normalize_sequence_image,
    _text_mask,
    apply_number_text_target_contract,
)

CTC_SANITY_SUMMARY = "ctc_sanity_summary.json"


@dataclass(frozen=True, slots=True)
class _SanityDataset:
    name: str
    target: NumberTextTarget
    samples: tuple[NumberSequenceSample, ...]
    note: str


def run_poker_legends_ctc_sanity(
    *,
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
    field_name: str = "hero_stack",
    targets: Sequence[NumberTextTarget] = ("base", "overlay"),
    synthetic_count: int = 1000,
    real_count: int = 20,
    epochs: int = 120,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    seed: int = 2028,
    progress: bool = False,
) -> dict[str, object]:
    """Run offline CTC falsification checks and write a JSON summary."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    target_tuple = tuple(targets)
    datasets: list[_SanityDataset] = []
    for target in target_tuple:
        datasets.append(
            _SanityDataset(
                name=f"synthetic_{target}",
                target=target,
                samples=_synthetic_samples(target, count=synthetic_count, rng=rng),
                note="clean synthetic overfit set generated from project-owned code",
            )
        )
        if manifest_path is not None:
            real_samples = _real_manifest_samples(
                Path(manifest_path),
                field_name=field_name,
                target=target,
                count=real_count,
            )
            datasets.append(
                _SanityDataset(
                    name=f"real_{target}",
                    target=target,
                    samples=real_samples,
                    note=(
                        "real overfit set from manifest; caller is responsible for "
                        "passing a reviewed clean/labeled_visible manifest"
                    ),
                )
            )

    dataset_summaries = [
        _run_dataset_overfit(
            dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed + index,
            progress=progress,
        )
        for index, dataset in enumerate(datasets)
    ]
    summary: dict[str, object] = {
        "schema_version": 1,
        "manifest": str(manifest_path) if manifest_path is not None else None,
        "field_name": field_name,
        "targets": list(target_tuple),
        "synthetic_count": synthetic_count,
        "real_count": real_count if manifest_path is not None else 0,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "too_short_case": _too_short_case_summary(),
        "datasets": dataset_summaries,
        "passed": all(bool(dataset["passed"]) for dataset in dataset_summaries)
        and bool(_too_short_case_summary()["failed_loudly"]),
    }
    (output / CTC_SANITY_SUMMARY).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run offline CRNN+CTC sanity checks for Poker Legends number OCR."
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--field-name", default="hero_stack")
    parser.add_argument(
        "--target",
        action="append",
        choices=NUMBER_TEXT_TARGETS,
        dest="targets",
        help="Target component to test; repeatable. Defaults to base and overlay.",
    )
    parser.add_argument("--synthetic-count", type=int, default=1000)
    parser.add_argument("--real-count", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = run_poker_legends_ctc_sanity(
        output_dir=args.out,
        manifest_path=args.manifest,
        field_name=args.field_name,
        targets=tuple(args.targets) if args.targets else ("base", "overlay"),
        synthetic_count=args.synthetic_count,
        real_count=args.real_count,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        progress=not args.quiet,
    )
    print(json.dumps(_stdout_summary(summary), indent=2, sort_keys=True))


def _run_dataset_overfit(
    dataset: _SanityDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    progress: bool,
) -> dict[str, object]:
    if len(dataset.samples) < 2:
        return {
            "name": dataset.name,
            "target": dataset.target,
            "status": "not_run",
            "reason": "insufficient_samples",
            "sample_count": len(dataset.samples),
            "passed": False,
            "note": dataset.note,
        }
    progress_callback: Callable[[int, int, float], None] | None = None
    if progress:
        def progress_callback(epoch: int, total: int, loss: float) -> None:
            _print_progress(dataset.name, epoch, total, loss)

    recognizer = NumberTorchCtcRecognizer(
        dataset.samples,
        architecture="crnn_ctc",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        progress_callback=progress_callback,
    )
    predictions = [
        {
            "row_id": sample.row_id,
            "expected": sample.text,
            "prediction": apply_number_text_target_contract(
                recognizer.recognize(sample.image),
                target=dataset.target,
                is_stack=True,
            ).to_dict(),
        }
        for sample in dataset.samples
    ]
    exact = sum(
        1
        for row in predictions
        if cast(Mapping[str, object], row["prediction"]).get("text") == row["expected"]
    )
    accepted = sum(
        1
        for row in predictions
        if bool(cast(Mapping[str, object], row["prediction"]).get("accepted"))
    )
    accepted_wrong = sum(
        1
        for row in predictions
        if bool(cast(Mapping[str, object], row["prediction"]).get("accepted"))
        and cast(Mapping[str, object], row["prediction"]).get("text") != row["expected"]
    )
    sample_count = len(dataset.samples)
    return {
        "name": dataset.name,
        "target": dataset.target,
        "status": "trained" if recognizer.available else "not_run",
        "reason": recognizer.reason,
        "note": dataset.note,
        "sample_count": sample_count,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "time_step_budget": recognizer.time_step_budget_summary,
        "raw_exact": exact,
        "raw_exact_rate": exact / sample_count if sample_count else 0.0,
        "accepted": accepted,
        "accepted_coverage": accepted / sample_count if sample_count else 0.0,
        "accepted_wrong": accepted_wrong,
        "passed": recognizer.available and exact == sample_count and accepted_wrong == 0,
        "examples": predictions[:10],
    }


def _synthetic_samples(
    target: NumberTextTarget,
    *,
    count: int,
    rng: random.Random,
) -> tuple[NumberSequenceSample, ...]:
    seed_texts = _seed_texts(target)
    texts = list(seed_texts)
    while len(texts) < count:
        texts.append(_random_text(target, rng))
    samples: list[NumberSequenceSample] = []
    for index, text in enumerate(texts[:count]):
        samples.append(
            NumberSequenceSample(
                target=target,
                frame_id=f"synthetic_{target}_{index:05d}",
                row_id=f"synthetic_{target}_{index:05d}",
                crop_path="synthetic",
                text=text,
                image=_synthetic_sequence_image(text),
            )
        )
    return tuple(samples)


def _real_manifest_samples(
    manifest_path: Path,
    *,
    field_name: str,
    target: NumberTextTarget,
    count: int,
) -> tuple[NumberSequenceSample, ...]:
    rows = _load_rows(
        manifest_path,
        field_name=field_name,
        max_crops=None,
        test_frame_modulo=10_000_000,
    )
    samples: list[NumberSequenceSample] = []
    for row in rows:
        expected = row.expected.for_target(target)
        if expected is None:
            continue
        crop = _load_rgb_image(row.crop_path)
        image = _normalize_sequence_image(_text_mask(crop, target=target))
        samples.append(
            NumberSequenceSample(
                target=target,
                frame_id=row.frame_id,
                row_id=row.row_id,
                crop_path=str(row.crop_path),
                text=expected,
                image=image,
            )
        )
        if len(samples) >= count:
            break
    return tuple(samples)


def _synthetic_sequence_image(text: str) -> np.ndarray:
    width = max(220, 18 * len(text) + 20)
    mask = np.zeros((54, width), dtype=np.uint8)
    cv2.putText(
        mask,
        text,
        (6, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        255,
        2,
        cv2.LINE_AA,
    )
    return _normalize_sequence_image(mask)


def _seed_texts(target: NumberTextTarget) -> tuple[str, ...]:
    if target == "base":
        return ("$0", "$5", "$55", "$185", "$399", "$1,005", "$43,044", "$1000")
    if target == "overlay":
        return ("+5", "+55", "+80", "+105", "+160", "+710", "+1000")
    return ("$0+30", "$185+105", "$399+80", "$1,005+10", "$43,044+1,000")


def _random_text(target: NumberTextTarget, rng: random.Random) -> str:
    if target == "base":
        return f"${_random_amount(rng):,}"
    if target == "overlay":
        return f"+{_random_overlay_amount(rng):,}"
    return f"${_random_amount(rng):,}+{_random_overlay_amount(rng):,}"


def _random_amount(rng: random.Random) -> int:
    candidates = [0, 5, 30, 55, 80, 105, 160, 185, 290, 399, 840, 990, 1000, 1005]
    if rng.random() < 0.75:
        return rng.choice(candidates)
    return rng.randint(0, 50_000)


def _random_overlay_amount(rng: random.Random) -> int:
    candidates = [5, 10, 30, 55, 80, 105, 160, 710, 1000]
    if rng.random() < 0.85:
        return rng.choice(candidates)
    return rng.randint(1, 2_000)


def _too_short_case_summary() -> dict[str, object]:
    try:
        validate_ctc_time_step_budget(
            "$1000",
            input_timesteps=13,
            min_ratio=DEFAULT_CTC_MIN_TIMESTEP_RATIO,
        )
    except CtcAlignmentError as exc:
        return {
            "text": "$1000",
            "input_timesteps": 13,
            "failed_loudly": True,
            "error": str(exc),
        }
    return {
        "text": "$1000",
        "input_timesteps": 13,
        "failed_loudly": False,
        "error": None,
    }


def _print_progress(dataset_name: str, epoch: int, total: int, loss: float) -> None:
    if epoch == 1 or epoch == total or epoch % max(1, total // 10) == 0:
        print(
            f"[ctc-sanity] {dataset_name} epoch {epoch}/{total} loss={loss:.4f}",
            file=sys.stderr,
            flush=True,
        )


def _stdout_summary(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "passed": summary["passed"],
        "too_short_case": summary["too_short_case"],
        "datasets": [
            {
                "name": dataset["name"],
                "status": dataset["status"],
                "reason": dataset["reason"],
                "sample_count": dataset["sample_count"],
                "raw_exact": dataset.get("raw_exact"),
                "accepted_wrong": dataset.get("accepted_wrong"),
                "passed": dataset["passed"],
            }
            for dataset in cast(Sequence[Mapping[str, Any]], summary["datasets"])
        ],
        "artifact": CTC_SANITY_SUMMARY,
    }


if __name__ == "__main__":  # pragma: no cover
    main()
