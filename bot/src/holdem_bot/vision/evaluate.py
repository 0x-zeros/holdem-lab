"""CLI helpers for evaluating fixture recognition accuracy."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from holdem_bot.vision.annotations import TableAnnotation
from holdem_bot.vision.recognition import RecognitionScore, evaluate_recognition
from holdem_bot.vision.roi_ocr import RoiOcrRecognizer


def evaluate_fixture(image_path: str | Path, annotation_path: str | Path) -> RecognitionScore:
    annotation = TableAnnotation.read_json(annotation_path)
    recognized = RoiOcrRecognizer().recognize(image_path, annotation)
    return evaluate_recognition(recognized, annotation)


def score_to_dict(score: RecognitionScore) -> dict[str, object]:
    return {
        "accuracy": score.accuracy,
        "correct": score.correct,
        "total": score.total,
        "categories": {
            category.category: {
                "accuracy": category.accuracy,
                "correct": category.correct,
                "total": category.total,
            }
            for category in score.categories
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate ROI/OCR recognition for one fixture.")
    parser.add_argument("image", help="Path to the fixture PNG.")
    parser.add_argument("annotation", help="Path to the fixture JSON annotation.")
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.0,
        help="Exit non-zero if overall accuracy is below this threshold.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    score = evaluate_fixture(args.image, args.annotation)
    print(json.dumps(score_to_dict(score), indent=2, sort_keys=True))
    if score.accuracy < args.min_accuracy:
        raise SystemExit(1)
