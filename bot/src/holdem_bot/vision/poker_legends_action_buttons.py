"""Locate Poker Legends' round action buttons (Call/Check, Raise/Bet, Fold) in a frame.

The LLM reads *what* the hero can do, but it cannot return usable pixel coordinates (it
hallucinates boxes), so the precise click target comes from CV. The action buttons are large,
bright, saturated **circles** in a row at the bottom-right of the table: blue = check/call,
gold = bet/raise, red = fold. We segment those bright-saturated color blobs, drop the dim
panels / grey corners / smaller raise-shortcut chips, anchor on the red Fold circle's row, and
return one button per colour with its centre in the input image's pixel space.

This is robust where template matching is not: the circle is detected by colour+shape, so a
changing label ("Call $50" -> "Call $100") and window scaling do not break it. Pure numpy/cv2,
so it is unit-testable headless. The detector is colour/aspect/area based and deliberately
returns nothing rather than guess when no clean button row is present (fail-closed upstream).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

BgrImage = NDArray[np.uint8]

#: Bottom-right search window (fractions of the frame). The action row always lives here; this
#: keeps table-centre clutter (avatars, pot, board) out of the candidate set.
SEARCH_TOP_FRACTION = 0.58
SEARCH_LEFT_FRACTION = 0.38

#: A button-ring pixel is bright AND saturated; this drops dark UI panels and the grey felt.
_MIN_SATURATION = 80
_MIN_VALUE = 120

#: OpenCV hue (0..179) -> action colour class. Red wraps around 0/180 and is handled separately.
_HUE_BANDS: dict[str, tuple[float, float]] = {"raise": (12.0, 45.0), "call": (80.0, 125.0)}

#: Colour class -> the canonical button slot (matches ``_command_for_action`` in the host).
COLOR_CLASS_TO_SLOT: dict[str, str] = {
    "call": "primary_left",  # check / call
    "raise": "primary_middle",  # bet / raise / all-in
    "fold": "primary_right",  # fold
}


@dataclass(frozen=True, slots=True)
class ActionButtonDetection:
    """One detected action button, centre in the input image's pixel coordinates."""

    slot: str
    color_class: str
    x: int
    y: int
    radius: int
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _classify_hue(hue: float) -> str | None:
    if hue <= 12.0 or hue >= 168.0:
        return "fold"  # red
    for cls, (lo, hi) in _HUE_BANDS.items():
        if lo <= hue <= hi:
            return cls
    return None


@dataclass(frozen=True, slots=True)
class _Blob:
    color_class: str
    x: int
    y: int
    radius: int
    area: int
    saturation: float


def _candidate_blobs(image: BgrImage) -> tuple[list[_Blob], float]:
    """Bright-saturated colour blobs in the bottom-right region, plus the offset+median radius."""
    height, width = image.shape[:2]
    y0 = int(height * SEARCH_TOP_FRACTION)
    x0 = int(width * SEARCH_LEFT_FRACTION)
    roi = image[y0:height, x0:width]
    if roi.size == 0:
        return [], 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue_ch, sat_ch, val_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    bright_sat = (sat_ch > _MIN_SATURATION) & (val_ch > _MIN_VALUE)
    raw_mask = bright_sat.astype(np.uint8) * 255
    kernel_side = int(height * 0.022) | 1  # odd; ~ centre-icon size, so ring+icon merge into a disk
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, np.ones((kernel_side, kernel_side), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_side = height * 0.045  # button diameter floor (scales with resolution)
    min_area = min_side * min_side * 0.45
    blobs: list[_Blob] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        blob_w = int(stats[index, cv2.CC_STAT_WIDTH])
        blob_h = int(stats[index, cv2.CC_STAT_HEIGHT])
        if area < min_area or blob_h == 0:
            continue
        aspect = blob_w / blob_h
        if aspect < 0.6 or aspect > 1.7:  # a circle's bbox is ~square; drop elongated panels
            continue
        component = (labels == index) & bright_sat
        hue_px = hue_ch[component]
        if hue_px.size < 30:
            continue
        if (hue_px < 12).sum() + (hue_px > 168).sum() > 0.5 * hue_px.size:
            hue = 0.0  # red split across the 0/180 wrap
        else:
            hue = float(np.median(hue_px))
        color_class = _classify_hue(hue)
        if color_class is None:
            continue
        cx, cy = centroids[index]
        blobs.append(
            _Blob(
                color_class=color_class,
                x=int(round(cx)) + x0,
                y=int(round(cy)) + y0,
                radius=int(round((blob_w + blob_h) / 4)),
                area=area,
                saturation=float(np.median(sat_ch[component])),
            )
        )
    radii = [blob.radius for blob in blobs]
    return blobs, float(np.median(radii)) if radii else 0.0


def detect_action_buttons(image: BgrImage) -> tuple[ActionButtonDetection, ...]:
    """Detect the action-button row; one button per colour, left-to-right. Empty if none found."""
    blobs, radius_median = _candidate_blobs(image)
    if not blobs:
        return ()
    reds = [blob for blob in blobs if blob.color_class == "fold"]
    anchor_y = (
        float(np.median([blob.y for blob in reds]))
        if reds
        else float(max(blob.y for blob in blobs))
    )
    tolerance = 0.8 * radius_median if radius_median > 0 else float("inf")
    row = [blob for blob in blobs if abs(blob.y - anchor_y) <= tolerance]
    best_by_color: dict[str, _Blob] = {}
    for blob in row:
        current = best_by_color.get(blob.color_class)
        if current is None or blob.area > current.area:
            best_by_color[blob.color_class] = blob
    detections = [
        ActionButtonDetection(
            slot=COLOR_CLASS_TO_SLOT[blob.color_class],
            color_class=blob.color_class,
            x=blob.x,
            y=blob.y,
            radius=blob.radius,
            confidence=_blob_confidence(blob),
        )
        for blob in best_by_color.values()
    ]
    return tuple(sorted(detections, key=lambda detection: detection.x))


def _blob_confidence(blob: _Blob) -> float:
    """A modest confidence from how saturated the ring is (1.0 at full saturation)."""
    return max(0.0, min(1.0, blob.saturation / 200.0))


def draw_action_buttons(
    image: BgrImage, detections: Sequence[ActionButtonDetection]
) -> BgrImage:
    """Draw each detected button's circle + centre crosshair + label. Mutates and returns image."""
    colors = {"fold": (0, 0, 255), "call": (0, 200, 0), "raise": (0, 215, 255)}
    for detection in detections:
        color = colors.get(detection.color_class, (255, 255, 255))
        cv2.circle(image, (detection.x, detection.y), detection.radius, color, 3)
        cv2.drawMarker(image, (detection.x, detection.y), color, cv2.MARKER_CROSS, 22, 2)
        label = f"{detection.color_class} ({detection.x},{detection.y})"
        cv2.putText(
            image,
            label,
            (detection.x - 50, max(detection.y - detection.radius - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return image


def main(argv: Sequence[str] | None = None) -> None:
    """Validation CLI: detect the action buttons in one frame, draw them, print their centres."""
    parser = argparse.ArgumentParser(
        description="Detect Poker Legends action-button click targets in a saved frame (no API)."
    )
    parser.add_argument("--image", required=True, help="input frame PNG/JPG")
    parser.add_argument("--out", help="annotated output PNG (default: <image>.buttons.png)")
    args = parser.parse_args(argv)

    loaded = cv2.imread(args.image)
    if loaded is None:
        raise SystemExit(f"could not read image: {args.image}")
    image = cast(BgrImage, loaded)
    detections = detect_action_buttons(image)
    out_path = Path(args.out) if args.out else Path(args.image).with_suffix(".buttons.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), draw_action_buttons(image.copy(), detections))
    print(
        json.dumps(
            {
                "image": args.image,
                "annotated": str(out_path),
                "count": len(detections),
                "buttons": [detection.to_dict() for detection in detections],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
