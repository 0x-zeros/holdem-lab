"""Image-based ScreenState detection for Poker Legends."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from holdem_bot.screen_state import ScreenKind, ScreenState
from holdem_bot.vision.annotations import ScreenRect
from holdem_bot.vision.poker_legends_layout import LAYOUT_NAME, poker_legends_layout_regions

RgbImage = NDArray[np.uint8]

PRIMARY_BUTTONS = ("primary_left", "primary_middle", "primary_right")
CENTER_MODAL = "center_modal"
LEFT_PANEL = "left_panel"
RIGHT_LOBBY_PANEL = "right_lobby_panel"
BOTTOM_BUY_IN_PROMPT = "bottom_buy_in_prompt"
ACTION_BUTTON_STD_THRESHOLD = 65.0
ACTION_BUTTON_MEAN_THRESHOLD = 50.0
OVERLAY_LOW_MEAN_THRESHOLD = 55.0
CENTER_MODAL_LOW_MEAN_THRESHOLD = 70.0
LEFT_PANEL_HIGH_MEAN_THRESHOLD = 85.0
BUY_IN_PROMPT_MAGENTA_THRESHOLD = 0.20


@dataclass(frozen=True, slots=True)
class PokerLegendsRegionFeature:
    name: str
    mean_brightness: float
    std_brightness: float
    magenta_fraction: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        return cls(
            name=str(data["name"]),
            mean_brightness=_to_float(data["mean_brightness"]),
            std_brightness=_to_float(data["std_brightness"]),
            magenta_fraction=_to_float(data.get("magenta_fraction", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class PokerLegendsScreenDetection:
    screen: ScreenState
    button_features: tuple[PokerLegendsRegionFeature, ...]
    overlay_features: tuple[PokerLegendsRegionFeature, ...]
    active_primary_buttons: int
    overlay_signals: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "screen": {
                "kind": self.screen.kind.value,
                "confidence": self.screen.confidence,
                "reason": self.screen.reason,
                "blocking_reason": self.screen.blocking_reason,
                "hero_turn": self.screen.hero_turn,
            },
            "button_features": [feature.to_dict() for feature in self.button_features],
            "overlay_features": [feature.to_dict() for feature in self.overlay_features],
            "active_primary_buttons": self.active_primary_buttons,
            "overlay_signals": list(self.overlay_signals),
        }


def detect_poker_legends_screen_state(
    image_path: str | Path,
    *,
    layout_annotation: Mapping[str, object] | None = None,
) -> PokerLegendsScreenDetection:
    image = _load_rgb_image(image_path)
    regions = _regions_for_image(image, layout_annotation)
    return detect_poker_legends_screen_state_from_image(image, regions=regions)


def detect_poker_legends_screen_state_from_image(
    image: RgbImage,
    *,
    regions: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
) -> PokerLegendsScreenDetection:
    resolved_regions = _regions_for_image(image, {"regions": regions} if regions else None)
    buttons_by_name = _regions_by_name(resolved_regions.get("buttons", ()))
    overlays_by_name = _regions_by_name(resolved_regions.get("overlays", ()))
    button_features = tuple(
        _feature_for_region(image, name, region)
        for name in PRIMARY_BUTTONS
        if (region := buttons_by_name.get(name)) is not None
    )
    overlay_features = tuple(
        _feature_for_region(image, name, region)
        for name in (CENTER_MODAL, LEFT_PANEL, RIGHT_LOBBY_PANEL, BOTTOM_BUY_IN_PROMPT)
        if (region := overlays_by_name.get(name)) is not None
    )
    active_primary_buttons = sum(
        1 for feature in button_features if _looks_like_active_primary_button(feature)
    )
    overlay_signals = _blocking_overlay_signals(overlay_features)
    if overlay_signals:
        confidence = 0.90 if len(overlay_signals) >= 2 else 0.82
        blocking_reason = (
            "buy_in_prompt"
            if any(signal.startswith(BOTTOM_BUY_IN_PROMPT) for signal in overlay_signals)
            else "other_overlay"
        )
        screen = ScreenState.blocked_overlay(
            blocking_reason=blocking_reason,
            confidence=confidence,
            reason=f"blocking overlay visual signals: {', '.join(overlay_signals)}",
        )
    elif active_primary_buttons >= 2:
        confidence = 0.88 if active_primary_buttons >= 3 else 0.80
        screen = ScreenState.actionable_table(
            confidence=confidence,
            hero_turn=True,
            reason=f"{active_primary_buttons} primary action buttons visible",
        )
    else:
        screen = ScreenState.table_observe(
            confidence=0.80,
            reason="no blocking overlay and no primary action button cluster",
        )
    return PokerLegendsScreenDetection(
        screen=screen,
        button_features=button_features,
        overlay_features=overlay_features,
        active_primary_buttons=active_primary_buttons,
        overlay_signals=overlay_signals,
    )


def evaluate_poker_legends_screen_state(
    truth_paths: Sequence[str | Path],
    *,
    annotation_dir: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    truth_files = [Path(path) for path in truth_paths]
    if not truth_files:
        raise ValueError("at least one truth overlay JSON is required")
    annotations = Path(annotation_dir)
    images = Path(image_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    correct = 0
    for truth_path in sorted(truth_files):
        truth = _read_json_object(truth_path)
        frame_id = str(truth["frame_id"])
        annotation_path = annotations / f"{frame_id}.json"
        annotation = _read_json_object(annotation_path)
        image_path = images / str(annotation["image"])
        detection = detect_poker_legends_screen_state(
            image_path,
            layout_annotation=annotation,
        )
        expected = _expected_screen_kind(truth)
        observed = detection.screen.kind.value
        status = "match" if observed == expected else "mismatch"
        if status == "match":
            correct += 1
        rows.append(
            {
                "frame_id": frame_id,
                "expected": expected,
                "observed": observed,
                "status": status,
                "confidence": detection.screen.confidence,
                "active_primary_buttons": detection.active_primary_buttons,
                "overlay_signals": list(detection.overlay_signals),
                "detection": detection.to_dict(),
            }
        )

    summary = {
        "schema_version": 1,
        "frames": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "report": "screen_state_report.md",
        "results": "screen_state_results.json",
        "rows": rows,
    }
    (output / "screen_state_results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_screen_state_report(output / "screen_state_report.md", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate image-based Poker Legends ScreenState detection."
    )
    parser.add_argument("truth_overlays", nargs="+", help="Truth overlay JSON files.")
    parser.add_argument(
        "--annotation-dir",
        required=True,
        help="Directory containing Poker Legends draft annotation JSON files.",
    )
    parser.add_argument("--image-root", required=True, help="Root for annotation image paths.")
    parser.add_argument("--out", required=True, help="Output directory for evaluation artifacts.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = evaluate_poker_legends_screen_state(
        args.truth_overlays,
        annotation_dir=args.annotation_dir,
        image_root=args.image_root,
        output_dir=args.out,
    )
    print(json.dumps(_stdout_summary(summary), indent=2, sort_keys=True))


def _regions_for_image(
    image: RgbImage,
    layout_annotation: Mapping[str, object] | None,
) -> Mapping[str, Sequence[Mapping[str, object]]]:
    raw_regions = None if layout_annotation is None else layout_annotation.get("regions")
    height, width = image.shape[:2]
    default_regions = poker_legends_layout_regions(int(width), int(height))
    if isinstance(raw_regions, Mapping):
        annotation_regions = {
            str(group): [region for region in _mapping_sequence(regions)]
            for group, regions in raw_regions.items()
        }
        if _has_poker_legends_layout_context(layout_annotation):
            return _merge_missing_default_regions(annotation_regions, default_regions)
        return annotation_regions
    return default_regions


def _feature_for_region(
    image: RgbImage,
    name: str,
    region: Mapping[str, object],
) -> PokerLegendsRegionFeature:
    crop = _crop(image, _rect_from_region(region))
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    magenta_mask = (
        ((hsv[:, :, 0] >= 140) | (hsv[:, :, 0] <= 5)) & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 80)
    )
    return PokerLegendsRegionFeature(
        name=name,
        mean_brightness=float(np.mean(crop)),
        std_brightness=float(np.std(crop)),
        magenta_fraction=float(np.mean(magenta_mask)),
    )


def _looks_like_active_primary_button(feature: PokerLegendsRegionFeature) -> bool:
    return (
        feature.mean_brightness >= ACTION_BUTTON_MEAN_THRESHOLD
        and feature.std_brightness >= ACTION_BUTTON_STD_THRESHOLD
    )


def _blocking_overlay_signals(
    features: Sequence[PokerLegendsRegionFeature],
) -> tuple[str, ...]:
    by_name = {feature.name: feature for feature in features}
    signals: list[str] = []
    center = by_name.get(CENTER_MODAL)
    left = by_name.get(LEFT_PANEL)
    right = by_name.get(RIGHT_LOBBY_PANEL)
    if center is not None and center.mean_brightness < CENTER_MODAL_LOW_MEAN_THRESHOLD:
        signals.append("center_modal_low_mean")
    if left is not None and left.mean_brightness < OVERLAY_LOW_MEAN_THRESHOLD:
        signals.append("left_panel_low_mean")
    if right is not None and right.mean_brightness < OVERLAY_LOW_MEAN_THRESHOLD:
        signals.append("right_lobby_panel_low_mean")
    if left is not None and left.mean_brightness > LEFT_PANEL_HIGH_MEAN_THRESHOLD:
        signals.append("left_panel_high_mean")
    buy_in = by_name.get(BOTTOM_BUY_IN_PROMPT)
    if buy_in is not None and buy_in.magenta_fraction >= BUY_IN_PROMPT_MAGENTA_THRESHOLD:
        signals.append("bottom_buy_in_prompt_magenta")
    return tuple(signals)


def _merge_missing_default_regions(
    annotation_regions: Mapping[str, Sequence[Mapping[str, object]]],
    default_regions: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, list[Mapping[str, object]]]:
    merged = {group: list(regions) for group, regions in annotation_regions.items()}
    for group, default_group in default_regions.items():
        merged_group = merged.setdefault(group, [])
        existing_names = {str(region.get("name")) for region in merged_group}
        for default_region in default_group:
            if str(default_region.get("name")) not in existing_names:
                merged_group.append(default_region)
    return merged


def _has_poker_legends_layout_context(layout_annotation: Mapping[str, object] | None) -> bool:
    if layout_annotation is None:
        return False
    if layout_annotation.get("source") == "poker_legends_video":
        return True
    metadata = layout_annotation.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    layout = metadata.get("layout")
    if not isinstance(layout, Mapping):
        return False
    return layout.get("name") == LAYOUT_NAME


def _regions_by_name(
    regions: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    return {str(region.get("name")): region for region in regions}


def _expected_screen_kind(truth: Mapping[str, object]) -> str:
    screen = truth.get("screen")
    if not isinstance(screen, Mapping):
        return ScreenKind.UNKNOWN_OR_TRANSITION.value
    return str(screen.get("kind") or ScreenKind.UNKNOWN_OR_TRANSITION.value)


def _write_screen_state_report(path: Path, summary: Mapping[str, object]) -> None:
    rows = _mapping_sequence(summary["rows"])
    lines = [
        "# Poker Legends ScreenState Evaluation",
        "",
        "## Summary",
        f"- Frames: {summary['frames']}",
        f"- Correct: {summary['correct']}",
        f"- Accuracy: {_to_float(summary['accuracy']):.3f}",
        "",
        "## Frames",
        "| Frame | Expected | Observed | Confidence | Buttons | Overlay signals | Status |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        signals = ", ".join(str(item) for item in _sequence(row.get("overlay_signals"))) or "-"
        lines.append(
            f"| `{row['frame_id']}` | `{row['expected']}` | `{row['observed']}` | "
            f"{_to_float(row['confidence']):.2f} | {row['active_primary_buttons']} | "
            f"{signals} | {row['status']} |"
        )
    mismatches = [row for row in rows if row.get("status") != "match"]
    lines.extend(["", "## Mismatches"])
    if mismatches:
        for row in mismatches:
            lines.append(
                f"- `{row['frame_id']}` expected `{row['expected']}`, observed `{row['observed']}`."
            )
    else:
        lines.append("No mismatches.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stdout_summary(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "frames": summary["frames"],
        "correct": summary["correct"],
        "accuracy": summary["accuracy"],
        "report": summary["report"],
        "results": summary["results"],
    }


def _load_rgb_image(path: str | Path) -> RgbImage:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cast(RgbImage, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _crop(image: RgbImage, rect: ScreenRect) -> RgbImage:
    height, width = image.shape[:2]
    x = max(0, rect.x)
    y = max(0, rect.y)
    right = min(width, rect.x + rect.width)
    bottom = min(height, rect.y + rect.height)
    return image[y:bottom, x:right]


def _rect_from_region(region: Mapping[str, object]) -> ScreenRect:
    raw_rect = region.get("rect")
    if not isinstance(raw_rect, Mapping):
        raise ValueError(f"region has no rect: {region}")
    return ScreenRect(
        x=_to_int(raw_rect["x"]),
        y=_to_int(raw_rect["y"]),
        width=_to_int(raw_rect["width"]),
        height=_to_int(raw_rect["height"]),
    )


def _read_json_object(path: str | Path) -> dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, object], data)


def _mapping_sequence(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [cast(Mapping[str, object], item) for item in value if isinstance(item, Mapping)]


def _sequence(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _to_int(value: object) -> int:
    if type(value) is int:
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"expected int-like value: {value!r}")


def _to_float(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        return float(value)
    raise TypeError(f"expected float-like value: {value!r}")
