"""Pure rendering for the Poker Legends perception HUD.

Draws the layout ROIs (scaled to the frame) and a text panel onto a captured BGR
frame, so a human watching live -- or an offline reviewer looking at a dumped PNG
-- can see at a glance what the bot perceives and where recognition breaks. No
screen capture, no GUI, no game dependency: pure numpy/cv2 so it is unit-testable
headless.

The ROIs are stored in the layout annotation's ``base_width x base_height`` space.
We scale them to the actual frame size, so when the live capture does not match the
calibrated resolution the rectangles visibly drift off the real cards/buttons --
which is exactly the layout-calibration signal we want to surface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

BgrImage = NDArray[np.uint8]

# BGR, one colour per layout region group.
GROUP_COLORS: dict[str, tuple[int, int, int]] = {
    "board": (0, 200, 0),
    "cards": (0, 215, 255),  # hero hole cards
    "buttons": (255, 128, 0),
    "seats": (200, 120, 255),
    "texts": (0, 255, 255),
    "overlays": (0, 0, 230),
}
_DEFAULT_COLOR = (180, 180, 180)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def layout_base_size(layout: Mapping[str, object]) -> tuple[int, int]:
    """Return the (width, height) the layout ROIs are expressed in."""
    meta = layout.get("metadata")
    if isinstance(meta, Mapping):
        inner = meta.get("layout")
        if isinstance(inner, Mapping):
            base_w, base_h = inner.get("base_width"), inner.get("base_height")
            if isinstance(base_w, int) and isinstance(base_h, int) and base_w > 0 and base_h > 0:
                return base_w, base_h
    width, height = layout.get("width"), layout.get("height")
    if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
        return width, height
    raise ValueError("layout annotation has no usable base width/height")


def draw_layout_regions(image: BgrImage, layout: Mapping[str, object]) -> BgrImage:
    """Draw every layout ROI rectangle (scaled to ``image``) labelled with its name.

    Mutates and returns ``image``.
    """
    height, width = image.shape[:2]
    base_w, base_h = layout_base_size(layout)
    scale_x, scale_y = width / base_w, height / base_h
    regions = layout.get("regions")
    if not isinstance(regions, Mapping):
        return image
    for group, items in regions.items():
        color = GROUP_COLORS.get(str(group), _DEFAULT_COLOR)
        if not isinstance(items, Sequence):
            continue
        for region in items:
            if not isinstance(region, Mapping):
                continue
            rect = region.get("rect")
            if not isinstance(rect, Mapping):
                continue
            try:
                x = round(int(rect["x"]) * scale_x)
                y = round(int(rect["y"]) * scale_y)
                rect_w = round(int(rect["width"]) * scale_x)
                rect_h = round(int(rect["height"]) * scale_y)
            except (KeyError, TypeError, ValueError):
                continue
            cv2.rectangle(image, (x, y), (x + rect_w, y + rect_h), color, 2)
            name = str(region.get("name") or group)
            cv2.putText(image, name, (x + 2, max(y - 4, 10)), _FONT, 0.4, color, 1, cv2.LINE_AA)
    return image


def draw_text_panel(
    image: BgrImage, lines: Sequence[str], *, origin: tuple[int, int] = (8, 8)
) -> BgrImage:
    """Draw ``lines`` of text on a darkened panel at ``origin``. Mutates ``image``."""
    if not lines:
        return image
    line_height, pad, char_w = 18, 6, 8
    longest = max((len(text) for text in lines), default=0)
    x0, y0 = origin
    x1 = min(x0 + longest * char_w + 2 * pad, image.shape[1] - 1)
    y1 = min(y0 + line_height * len(lines) + 2 * pad, image.shape[0] - 1)
    panel = image[y0:y1, x0:x1]
    if panel.size:
        # Darken the background so light text stays legible over any frame.
        cv2.addWeighted(panel, 0.35, np.zeros_like(panel), 0.65, 0.0, panel)
    for index, text in enumerate(lines):
        baseline = y0 + pad + line_height * (index + 1) - 4
        cv2.putText(image, text, (x0 + pad, baseline), _FONT, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    return image


def render_overlay(
    frame_bgr: BgrImage, layout: Mapping[str, object], lines: Sequence[str]
) -> BgrImage:
    """Return a copy of ``frame_bgr`` with layout ROIs and a text panel drawn on it."""
    out = frame_bgr.copy()
    draw_layout_regions(out, layout)
    draw_text_panel(out, lines)
    return out
