"""Poker Legends ROI layout bootstrap helpers."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import cv2
from numpy.typing import NDArray

from holdem_bot.vision.annotations import ScreenRect

RgbImage = NDArray[Any]

LAYOUT_NAME = "poker_legends_1600w_v1"
BASE_WIDTH = 1600
BASE_HEIGHT = 982


@dataclass(frozen=True, slots=True)
class LayoutRegion:
    name: str
    kind: str
    rect: ScreenRect
    value: object | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "rect": asdict(self.rect),
            "value": self.value,
            "notes": self.notes,
        }


def apply_poker_legends_layout(
    annotation_path: str | Path,
    output_path: str | Path | None = None,
    *,
    image_root: str | Path | None = None,
    overlay_dir: str | Path | None = None,
) -> dict[str, object]:
    annotation_file = Path(annotation_path)
    draft = _read_json_object(annotation_file)
    width = _to_int(draft["width"])
    height = _to_int(draft["height"])
    draft["regions"] = _layout_regions(width, height)

    metadata = _metadata(draft)
    metadata["layout"] = {
        "name": LAYOUT_NAME,
        "base_width": BASE_WIDTH,
        "base_height": BASE_HEIGHT,
        "status": "roi_applied_values_pending",
    }
    draft["metadata"] = metadata

    target = annotation_file if output_path is None else Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if image_root is not None and overlay_dir is not None:
        image_path = Path(image_root) / str(draft["image"])
        overlay_path = Path(overlay_dir) / f"{annotation_file.stem}_layout.png"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        draw_layout_overlay(image_path, target, overlay_path)

    return draft


def draw_layout_overlay(
    image_path: str | Path,
    annotation_path: str | Path,
    output_path: str | Path,
) -> None:
    annotation = _read_json_object(Path(annotation_path))
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    for group, regions in _region_groups(annotation).items():
        color = _group_color(group)
        for region in regions:
            rect = _rect_from_region(region)
            cv2.rectangle(
                image,
                (rect.x, rect.y),
                (rect.x + rect.width, rect.y + rect.height),
                color,
                2,
            )
            label = f"{group}:{region.get('name', '')}"
            cv2.putText(
                image,
                label[:32],
                (rect.x, max(16, rect.y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"could not write overlay image: {output}")


def poker_legends_layout_regions(width: int, height: int) -> dict[str, list[dict[str, object]]]:
    """Return the first-pass scalable Poker Legends ROI layout."""
    return _layout_regions(width, height)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply the first-pass Poker Legends ROI layout to draft annotations."
    )
    parser.add_argument("annotations", nargs="+", help="Draft annotation JSON files.")
    parser.add_argument(
        "--out-dir",
        help="Optional directory for updated annotations. Defaults to editing files in place.",
    )
    parser.add_argument(
        "--image-root",
        help="Root directory used to resolve each annotation's image path for overlays.",
    )
    parser.add_argument("--overlay-dir", help="Optional directory for layout overlay PNGs.")
    parser.add_argument("--report", help="Optional Markdown report path for the layout pass.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    outputs: list[str] = []
    for annotation in args.annotations:
        annotation_path = Path(annotation)
        output_path = None if args.out_dir is None else Path(args.out_dir) / annotation_path.name
        apply_poker_legends_layout(
            annotation_path,
            output_path,
            image_root=None if args.image_root is None else Path(args.image_root),
            overlay_dir=None if args.overlay_dir is None else Path(args.overlay_dir),
        )
        outputs.append(str(output_path or annotation_path))
    if args.report is not None:
        _write_layout_report(
            Path(args.report),
            outputs=outputs,
            overlay_dir=None if args.overlay_dir is None else str(args.overlay_dir),
        )
    print(
        json.dumps(
            {
                "annotations": outputs,
                "count": len(outputs),
                "report": args.report,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _layout_regions(width: int, height: int) -> dict[str, list[dict[str, object]]]:
    raw = _raw_regions()
    if width == BASE_WIDTH and height == BASE_HEIGHT:
        return raw
    return _scale_all_regions(raw, width, height)


def _scale_all_regions(
    raw: Mapping[str, Sequence[Mapping[str, object]]],
    width: int,
    height: int,
) -> dict[str, list[dict[str, object]]]:
    scaled: dict[str, list[dict[str, object]]] = {}
    for group, regions in raw.items():
        scaled[group] = []
        for region in regions:
            raw_rect = cast(dict[str, int], region["rect"])
            region = dict(region)
            region["rect"] = asdict(_scale_rect(raw_rect, width, height))
            scaled[group].append(region)
    return scaled


def _raw_regions() -> dict[str, list[dict[str, object]]]:
    return {
        "board": [
            _card_region(f"board_{index}", x, 432, 84, 112).to_dict()
            for index, x in enumerate((508, 599, 689, 779, 869))
        ],
        "buttons": [
            _button_region("primary_left", 1110, 776, 110, 124).to_dict(),
            _button_region("primary_middle", 1268, 776, 110, 124).to_dict(),
            _button_region("primary_right", 1425, 776, 110, 124).to_dict(),
            _button_region("raise_shortcut_top", 1268, 506, 110, 118).to_dict(),
            _button_region("raise_shortcut_middle", 1268, 638, 110, 118).to_dict(),
        ],
        "seats": [
            _seat_region("hero", 470, 642, 480, 146).to_dict(),
            _seat_region("top", 620, 250, 300, 160).to_dict(),
            _seat_region("right_top", 922, 280, 330, 150).to_dict(),
            _seat_region("right_middle", 1370, 486, 170, 92).to_dict(),
            _seat_region("right_bottom", 1370, 588, 170, 92).to_dict(),
        ],
        "cards": [
            _card_region("hero_hole_0", 652, 570, 102, 136).to_dict(),
            _card_region("hero_hole_1", 738, 570, 112, 136).to_dict(),
        ],
        "texts": [
            _text_region("pot", 660, 360, 180, 50).to_dict(),
            _text_region("street_prompt", 586, 472, 306, 48).to_dict(),
            _text_region("hero_current_bet", 705, 515, 125, 50).to_dict(),
            _text_region("hero_stack", 660, 708, 210, 52).to_dict(),
            _text_region("hero_hand_rank", 760, 674, 180, 36).to_dict(),
            _text_region("top_action_banner", 1060, 108, 460, 74).to_dict(),
            _text_region("right_top_stack", 972, 300, 160, 58).to_dict(),
        ],
        "overlays": [
            _overlay_region("center_modal", 470, 320, 660, 300).to_dict(),
            _overlay_region("left_panel", 70, 86, 430, 710).to_dict(),
            _overlay_region("right_lobby_panel", 1010, 190, 500, 520).to_dict(),
            _overlay_region("bottom_buy_in_prompt", 1085, 780, 470, 135).to_dict(),
        ],
    }


def _write_layout_report(path: Path, *, outputs: Sequence[str], overlay_dir: str | None) -> None:
    raw = _raw_regions()
    lines = [
        "# Poker Legends ROI Layout Pass",
        "",
        "## Layout",
        f"- Name: `{LAYOUT_NAME}`",
        f"- Base size: {BASE_WIDTH}x{BASE_HEIGHT}",
        "- Status: first-pass ROI bootstrap; values are still pending manual annotation.",
        "",
        "## Region Groups",
    ]
    for group, regions in sorted(raw.items()):
        lines.append(f"- `{group}`: {len(regions)} regions")
    lines.extend(
        [
            "",
            "## Outputs",
            f"- Updated annotations: {len(outputs)} files",
        ]
    )
    if overlay_dir is not None:
        lines.append(f"- Overlay images: `{overlay_dir}`")
    lines.extend(
        [
            "",
            "## Notes",
            "- Card/button/text boxes are intended as starting ROIs, not final measured truth.",
            "- Table frames and overlay/modal frames share the same template so no-action "
            "states can be labeled alongside normal poker states.",
            "- After manual value entry, the next step is a Poker Legends recognizer that "
            "emits the existing `RecognizedTable` structure.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _card_region(name: str, x: int, y: int, width: int, height: int) -> LayoutRegion:
    return LayoutRegion(name=name, kind="card", rect=ScreenRect(x, y, width, height), value=None)


def _button_region(name: str, x: int, y: int, width: int, height: int) -> LayoutRegion:
    return LayoutRegion(
        name=name,
        kind="button",
        rect=ScreenRect(x, y, width, height),
        value=None,
        notes="Fill visible label/action_type when present.",
    )


def _seat_region(name: str, x: int, y: int, width: int, height: int) -> LayoutRegion:
    return LayoutRegion(
        name=name,
        kind="seat",
        rect=ScreenRect(x, y, width, height),
        value=None,
        notes="Seat panel/avatar area; fill stack/current/active after manual review.",
    )


def _text_region(name: str, x: int, y: int, width: int, height: int) -> LayoutRegion:
    return LayoutRegion(name=name, kind="text", rect=ScreenRect(x, y, width, height), value=None)


def _overlay_region(name: str, x: int, y: int, width: int, height: int) -> LayoutRegion:
    return LayoutRegion(
        name=name,
        kind="blocking_overlay",
        rect=ScreenRect(x, y, width, height),
        value=None,
        notes="Use for modal/menu/no-action blocking states.",
    )


def _scale_rect(rect: Mapping[str, int], width: int, height: int) -> ScreenRect:
    scale_x = width / BASE_WIDTH
    scale_y = height / BASE_HEIGHT
    return ScreenRect(
        x=round(rect["x"] * scale_x),
        y=round(rect["y"] * scale_y),
        width=max(1, round(rect["width"] * scale_x)),
        height=max(1, round(rect["height"] * scale_y)),
    )


def _metadata(draft: dict[str, object]) -> dict[str, object]:
    raw_metadata = draft.get("metadata", {})
    if not isinstance(raw_metadata, dict):
        return {}
    return dict(raw_metadata)


def _region_groups(annotation: Mapping[str, object]) -> dict[str, list[dict[str, object]]]:
    raw_regions = annotation.get("regions", {})
    if not isinstance(raw_regions, dict):
        return {}
    groups: dict[str, list[dict[str, object]]] = {}
    for group, regions in raw_regions.items():
        if not isinstance(group, str) or not isinstance(regions, list):
            continue
        groups[group] = [dict(region) for region in regions if isinstance(region, dict)]
    return groups


def _rect_from_region(region: Mapping[str, object]) -> ScreenRect:
    raw_rect = region["rect"]
    if not isinstance(raw_rect, dict):
        raise TypeError(f"expected rect object: {raw_rect!r}")
    return ScreenRect(
        x=_to_int(raw_rect["x"]),
        y=_to_int(raw_rect["y"]),
        width=_to_int(raw_rect["width"]),
        height=_to_int(raw_rect["height"]),
    )


def _group_color(group: str) -> tuple[int, int, int]:
    colors = {
        "board": (0, 255, 255),
        "buttons": (0, 255, 0),
        "cards": (255, 0, 255),
        "overlays": (0, 128, 255),
        "seats": (255, 180, 0),
        "texts": (255, 255, 255),
    }
    return colors.get(group, (200, 200, 200))


def _read_json_object(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"expected JSON object: {path}")
    return cast(dict[str, object], data)


def _to_int(value: object) -> int:
    if type(value) is int:
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"expected int or string: {value!r}")
