import numpy as np
import pytest
from holdem_bot.vision.perception_overlay import (
    draw_layout_regions,
    layout_base_size,
    render_overlay,
)


def _layout(base_w: int = 100, base_h: int = 100) -> dict[str, object]:
    return {
        "metadata": {"layout": {"base_width": base_w, "base_height": base_h}},
        "regions": {
            "buttons": [
                {"name": "primary_left", "rect": {"x": 10, "y": 10, "width": 20, "height": 20}}
            ],
            "board": [{"name": "board_0", "rect": {"x": 40, "y": 40, "width": 10, "height": 10}}],
        },
    }


def test_layout_base_size_prefers_metadata() -> None:
    assert layout_base_size(_layout(160, 98)) == (160, 98)


def test_layout_base_size_falls_back_to_width_height() -> None:
    assert layout_base_size({"width": 800, "height": 600}) == (800, 600)


def test_layout_base_size_raises_without_dimensions() -> None:
    with pytest.raises(ValueError, match="base width/height"):
        layout_base_size({})


def test_render_overlay_returns_new_array_without_mutating_input() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = render_overlay(frame, _layout(), ["hello", "world"])

    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, frame)  # something was drawn
    assert np.array_equal(frame, np.zeros((100, 100, 3), dtype=np.uint8))  # input untouched


def test_draw_layout_regions_scales_rois_to_frame() -> None:
    # base 100x100, button ROI (10,10,20,20) -> on a 200x200 frame it scales to (20,20,40,40).
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    draw_layout_regions(frame, _layout(100, 100))

    assert frame[20:60, 20:60].any()  # the scaled rectangle border is drawn here
    assert not frame[150:200, 150:200].any()  # nothing reaches the far corner


def test_render_overlay_tolerates_missing_regions() -> None:
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    out = render_overlay(frame, {"width": 40, "height": 40}, [])
    assert out.shape == frame.shape
