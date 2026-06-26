"""Build Poker Legends template libraries from multiple truth sources."""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from holdem_bot.vision.poker_legends_buttons import (
    DEFAULT_LEFT_MAX_DISTANCE,
    build_and_evaluate_poker_legends_button_templates,
)
from holdem_bot.vision.poker_legends_card_parts import (
    DEFAULT_RANK_MAX_DISTANCE,
    DEFAULT_SUIT_MAX_DISTANCE,
    build_and_evaluate_poker_legends_card_part_templates,
)
from holdem_bot.vision.poker_legends_cards import (
    DEFAULT_MAX_DISTANCE as DEFAULT_CARD_MAX_DISTANCE,
)
from holdem_bot.vision.poker_legends_cards import (
    build_and_evaluate_poker_legends_card_templates,
)

TemplateKind = Literal["cards", "card-parts", "buttons", "both", "all"]


@dataclass(frozen=True, slots=True)
class PokerLegendsTemplateSource:
    name: str
    truth_path: str | Path
    annotation_dir: str | Path
    image_root: str | Path

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        return {key: str(value) for key, value in data.items()}


def build_and_evaluate_poker_legends_multi_templates(
    sources: Sequence[PokerLegendsTemplateSource],
    *,
    output_dir: str | Path,
    kind: TemplateKind = "all",
    card_max_distance: float | None = None,
    rank_max_distance: float | None = None,
    suit_max_distance: float | None = None,
    left_max_distance: float | None = None,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = materialize_poker_legends_template_dataset(sources, output_dir=output)
    truth_paths = [Path(str(path)) for path in _sequence(dataset["truth_paths"])]
    annotation_dir = Path(str(dataset["annotation_dir"]))
    image_root = Path(str(dataset["image_root"]))

    summary: dict[str, object] = {
        "schema_version": 1,
        "kind": kind,
        "dataset": {key: value for key, value in dataset.items() if key != "truth_paths"},
        "sources": [source.to_dict() for source in sources],
    }
    if kind in {"cards", "all"}:
        summary["cards"] = build_and_evaluate_poker_legends_card_templates(
            truth_paths,
            annotation_dir=annotation_dir,
            image_root=image_root,
            output_dir=output / "card_templates",
            max_distance=DEFAULT_CARD_MAX_DISTANCE
            if card_max_distance is None
            else card_max_distance,
        )
    if kind in {"card-parts", "both", "all"}:
        summary["card_parts"] = build_and_evaluate_poker_legends_card_part_templates(
            truth_paths,
            annotation_dir=annotation_dir,
            image_root=image_root,
            output_dir=output / "card_part_templates",
            rank_max_distance=(
                DEFAULT_RANK_MAX_DISTANCE if rank_max_distance is None else rank_max_distance
            ),
            suit_max_distance=(
                DEFAULT_SUIT_MAX_DISTANCE if suit_max_distance is None else suit_max_distance
            ),
        )
    if kind in {"buttons", "both", "all"}:
        summary["buttons"] = build_and_evaluate_poker_legends_button_templates(
            truth_paths,
            annotation_dir=annotation_dir,
            image_root=image_root,
            output_dir=output / "button_templates",
            left_max_distance=(
                DEFAULT_LEFT_MAX_DISTANCE if left_max_distance is None else left_max_distance
            ),
        )

    (output / "multi_template_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_multi_template_report(output / "multi_template_report.md", summary)
    return summary


def materialize_poker_legends_template_dataset(
    sources: Sequence[PokerLegendsTemplateSource],
    *,
    output_dir: str | Path,
) -> dict[str, object]:
    if not sources:
        raise ValueError("at least one source is required")
    output = Path(output_dir)
    dataset_dir = output / "merged_dataset"
    truth_dir = dataset_dir / "truth_overlays"
    annotation_dir = dataset_dir / "annotations"
    truth_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, object]] = []
    truth_paths: list[str] = []
    used_frame_ids: set[str] = set()
    for source in sources:
        source_name = _safe_source_name(source.name)
        for truth_path in _resolve_truth_paths(source.truth_path):
            truth = _read_json_object(truth_path)
            original_frame_id = str(truth["frame_id"])
            frame_id = f"{source_name}__{original_frame_id}"
            if frame_id in used_frame_ids:
                raise ValueError(f"duplicate merged frame id: {frame_id}")
            used_frame_ids.add(frame_id)

            original_annotation = _read_json_object(
                Path(source.annotation_dir) / f"{original_frame_id}.json"
            )
            image_path = _resolve_image_path(
                image_root=Path(source.image_root),
                image=original_annotation["image"],
            )

            merged_truth = dict(truth)
            merged_truth["frame_id"] = frame_id
            merged_truth["source"] = {
                "name": source.name,
                "original_frame_id": original_frame_id,
                "truth_path": str(truth_path),
            }
            merged_annotation = dict(original_annotation)
            merged_annotation["image"] = str(image_path)

            merged_truth_path = truth_dir / f"{frame_id}.json"
            merged_annotation_path = annotation_dir / f"{frame_id}.json"
            merged_truth_path.write_text(
                json.dumps(merged_truth, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            merged_annotation_path.write_text(
                json.dumps(merged_annotation, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            truth_paths.append(str(merged_truth_path))
            frames.append(
                {
                    "frame_id": frame_id,
                    "source": source.name,
                    "original_frame_id": original_frame_id,
                    "truth_path": str(truth_path),
                    "image": str(image_path),
                }
            )

    manifest: dict[str, object] = {
        "schema_version": 1,
        "sources": [source.to_dict() for source in sources],
        "frames": frames,
        "frame_count": len(frames),
        "truth_dir": str(truth_dir),
        "annotation_dir": str(annotation_dir),
        "image_root": "/",
        "truth_paths": truth_paths,
    }
    (dataset_dir / "merged_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Poker Legends template recognition from multiple truth sources."
    )
    parser.add_argument(
        "--source",
        nargs=4,
        action="append",
        metavar=("NAME", "TRUTH_PATH_OR_DIR", "ANNOTATION_DIR", "IMAGE_ROOT"),
        required=True,
        help="Repeatable source. TRUTH_PATH_OR_DIR may be a JSON file, directory, or glob.",
    )
    parser.add_argument(
        "--kind",
        choices=("cards", "card-parts", "buttons", "both", "all"),
        default="all",
        help="Template family to build.",
    )
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--card-max-distance", type=float, default=None)
    parser.add_argument("--rank-max-distance", type=float, default=None)
    parser.add_argument("--suit-max-distance", type=float, default=None)
    parser.add_argument("--left-max-distance", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    sources = tuple(
        PokerLegendsTemplateSource(
            name=str(item[0]),
            truth_path=str(item[1]),
            annotation_dir=str(item[2]),
            image_root=str(item[3]),
        )
        for item in args.source
    )
    summary = build_and_evaluate_poker_legends_multi_templates(
        sources,
        output_dir=args.out,
        kind=cast(TemplateKind, args.kind),
        card_max_distance=args.card_max_distance,
        rank_max_distance=args.rank_max_distance,
        suit_max_distance=args.suit_max_distance,
        left_max_distance=args.left_max_distance,
    )
    print(json.dumps(_stdout_summary(summary), indent=2, sort_keys=True))


def _resolve_truth_paths(path_or_pattern: str | Path) -> tuple[Path, ...]:
    raw = str(path_or_pattern)
    path = Path(raw)
    if path.is_dir():
        paths = tuple(sorted(path.glob("*.json")))
    elif _has_glob(raw):
        paths = tuple(sorted(Path(item) for item in glob.glob(raw)))
    elif path.is_file():
        paths = (path,)
    else:
        raise FileNotFoundError(f"truth path does not exist: {raw}")
    if not paths:
        raise FileNotFoundError(f"no truth JSON files found: {raw}")
    return paths


def _resolve_image_path(*, image_root: Path, image: object) -> Path:
    image_path = Path(str(image))
    if not image_path.is_absolute():
        image_path = image_root / image_path
    if not image_path.is_file():
        raise FileNotFoundError(f"could not read source image: {image_path}")
    return image_path.resolve()


def _safe_source_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    if not safe:
        raise ValueError("source name cannot be empty")
    return safe


def _has_glob(value: str) -> bool:
    return any(char in value for char in "*?[]")


def _stdout_summary(summary: Mapping[str, object]) -> dict[str, object]:
    dataset = cast(Mapping[str, object], summary["dataset"])
    result: dict[str, object] = {
        "kind": summary["kind"],
        "frame_count": dataset["frame_count"],
        "manifest": "multi_template_summary.json",
    }
    if "cards" in summary:
        cards = cast(Mapping[str, object], summary["cards"])
        leave_frame = cast(Mapping[str, object], cards["leave_frame_eval"])
        result["cards"] = {
            "templates": cards["templates"],
            "unique_cards": cards["unique_cards"],
            "leave_frame_precision": leave_frame["visible_precision"],
            "leave_frame_coverage": leave_frame["visible_coverage"],
        }
    if "card_parts" in summary:
        card_parts = cast(Mapping[str, object], summary["card_parts"])
        leave_card = cast(Mapping[str, object], card_parts["leave_card_eval"])
        result["card_parts"] = {
            "templates": card_parts["templates"],
            "unique_cards": card_parts["unique_cards"],
            "leave_card_precision": leave_card["visible_precision"],
            "leave_card_coverage": leave_card["visible_coverage"],
        }
    if "buttons" in summary:
        buttons = cast(Mapping[str, object], summary["buttons"])
        leave_frame = cast(Mapping[str, object], buttons["leave_frame_eval"])
        result["buttons"] = {
            "templates": buttons["templates"],
            "action_counts": buttons["action_counts"],
            "leave_frame_precision": leave_frame["precision"],
            "leave_frame_coverage": leave_frame["coverage"],
        }
    return result


def _write_multi_template_report(path: Path, summary: Mapping[str, object]) -> None:
    dataset = cast(Mapping[str, object], summary["dataset"])
    lines = [
        "# Poker Legends Multi-Source Templates",
        "",
        "## Dataset",
        f"- Kind: `{summary['kind']}`",
        f"- Frames: {dataset['frame_count']}",
        f"- Annotation dir: `{dataset['annotation_dir']}`",
        f"- Truth dir: `{dataset['truth_dir']}`",
        "",
        "## Sources",
    ]
    for source in _mapping_sequence(summary["sources"]):
        lines.append(
            f"- `{source['name']}`: truth=`{source['truth_path']}`, "
            f"annotations=`{source['annotation_dir']}`, images=`{source['image_root']}`"
        )
    if "cards" in summary:
        cards = cast(Mapping[str, object], summary["cards"])
        leave_frame = cast(Mapping[str, object], cards["leave_frame_eval"])
        lines.extend(
            [
                "",
                "## Cards",
                f"- Templates: {cards['templates']}",
                f"- Unique cards: {cards['unique_cards']} / 52",
                f"- Leave-frame precision: {_to_float(leave_frame['visible_precision']):.3f}",
                f"- Leave-frame coverage: {_to_float(leave_frame['visible_coverage']):.3f}",
            ]
        )
    if "card_parts" in summary:
        card_parts = cast(Mapping[str, object], summary["card_parts"])
        leave_card = cast(Mapping[str, object], card_parts["leave_card_eval"])
        rank_labels = ", ".join(str(item) for item in _sequence(card_parts["rank_labels"]))
        suit_labels = ", ".join(str(item) for item in _sequence(card_parts["suit_labels"]))
        lines.extend(
            [
                "",
                "## Card Parts",
                f"- Templates: {card_parts['templates']}",
                f"- Unique cards: {card_parts['unique_cards']} / 52",
                f"- Rank labels: {rank_labels}",
                f"- Suit labels: {suit_labels}",
                f"- Leave-card precision: {_to_float(leave_card['visible_precision']):.3f}",
                f"- Leave-card coverage: {_to_float(leave_card['visible_coverage']):.3f}",
            ]
        )
    if "buttons" in summary:
        buttons = cast(Mapping[str, object], summary["buttons"])
        leave_frame = cast(Mapping[str, object], buttons["leave_frame_eval"])
        action_counts = cast(Mapping[str, object], buttons["action_counts"])
        action_count_text = ", ".join(f"{key}={value}" for key, value in action_counts.items())
        lines.extend(
            [
                "",
                "## Buttons",
                f"- Templates: {buttons['templates']}",
                f"- Action counts: {action_count_text}",
                f"- Leave-frame precision: {_to_float(leave_frame['precision']):.3f}",
                f"- Leave-frame coverage: {_to_float(leave_frame['coverage']):.3f}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json_object(path: str | Path) -> dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, object], data)


def _sequence(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _mapping_sequence(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [cast(Mapping[str, object], item) for item in value if isinstance(item, Mapping)]


def _to_float(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        return float(value)
    raise TypeError(f"expected float-like value: {value!r}")
