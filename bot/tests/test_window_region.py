"""Window-selection policy tests (pure; OS enumeration is host-only)."""

from __future__ import annotations

from holdem_bot.adapters.window_region import WindowInfo, select_window


def _w(title: str, *, x: int = 0, y: int = 0, width: int = 100, height: int = 100) -> WindowInfo:
    return WindowInfo(title=title, left=x, top=y, width=width, height=height)


def test_single_title_match_wins() -> None:
    windows = [_w("Finder"), _w("Poker Legends", x=10), _w("Safari", x=20)]
    selected = select_window(windows, None, title_substring="poker legends")
    assert selected is not None and selected.title == "Poker Legends"


def test_multiple_matches_prefer_foreground() -> None:
    a = _w("Poker Legends", x=0)
    b = _w("Poker Legends (2)", x=500)
    assert select_window([a, b], b, title_substring="poker") is b


def test_multiple_matches_largest_when_no_foreground_match() -> None:
    small = _w("Poker Legends", x=10, width=100, height=100)
    big = _w("Poker Legends big", x=20, width=800, height=600)
    other = _w("Finder", x=900, y=900)
    assert select_window([small, big], other, title_substring="poker") is big


def test_no_title_match_falls_back_to_foreground() -> None:
    foreground = _w("Some Game Window", x=300)
    windows = [_w("Finder"), _w("Safari", x=20)]
    selected = select_window(windows, foreground, title_substring="poker")
    assert selected is foreground


def test_no_match_and_no_foreground_returns_none() -> None:
    assert select_window([_w("Finder")], None, title_substring="poker") is None
