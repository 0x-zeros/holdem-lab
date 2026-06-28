"""Locate the game's OS window so capture can crop to it (no desktop leaks, never clips UI).

Cross-platform: macOS via ``osascript`` (System Events; needs Accessibility permission) and
Windows via the Win32 API through ``ctypes`` (no permission). Both are dependency-free. The
window-selection policy (``select_window``) is pure and unit-tested; the per-OS enumeration is
host-only. ``find_game_window`` returns the window rect in the OS's screen coordinates -- the
capture layer may still need a scale factor on HiDPI/Retina displays, so calibrate once.
"""

from __future__ import annotations

import platform
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WindowInfo:
    """One on-screen window in OS screen coordinates (points on macOS, pixels on Windows)."""

    title: str
    left: int
    top: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return max(self.width, 0) * max(self.height, 0)

    def region(self) -> dict[str, int]:
        """As an mss-style capture region dict."""
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


def select_window(
    windows: Sequence[WindowInfo],
    foreground: WindowInfo | None,
    *,
    title_substring: str,
) -> WindowInfo | None:
    """Hybrid title + foreground policy.

    Match by title substring (case-insensitive): exactly one match wins; several matches resolve
    to the foreground one if it is among them, else the largest; no match falls back to the
    foreground window (the one you clicked to the front).
    """
    needle = title_substring.strip().lower()
    matches = [window for window in windows if needle and needle in window.title.lower()]
    if len(matches) == 1:
        return matches[0]
    if matches:
        if foreground is not None:
            for window in matches:
                if _same_window(window, foreground):
                    return window
        return max(matches, key=lambda window: window.area)
    return foreground


def find_game_window(title_substring: str = "Poker Legends") -> WindowInfo | None:
    """Find the game window via the hybrid policy on the current OS (None if unavailable)."""
    return select_window(list_windows(), foreground_window(), title_substring=title_substring)


def list_windows() -> list[WindowInfo]:
    system = platform.system()
    if system == "Darwin":
        return _macos_list_windows()
    if system == "Windows":
        return _windows_list_windows()
    return []


def foreground_window() -> WindowInfo | None:
    system = platform.system()
    if system == "Darwin":
        return _macos_foreground_window()
    if system == "Windows":
        return _windows_foreground_window()
    return None


def _same_window(a: WindowInfo, b: WindowInfo) -> bool:
    return a.left == b.left and a.top == b.top and a.width == b.width and a.height == b.height


# --- macOS (osascript / System Events) -------------------------------------------------------

_MACOS_LIST_SCRIPT = (
    'tell application "System Events"\n'
    "  set rows to {}\n"
    "  repeat with proc in (application processes whose visible is true)\n"
    "    repeat with win in (windows of proc)\n"
    "      try\n"
    "        set p to position of win\n"
    "        set s to size of win\n"
    "        set end of rows to ((name of proc) & tab & (name of win) & tab & "
    "(item 1 of p) & tab & (item 2 of p) & tab & (item 1 of s) & tab & (item 2 of s))\n"
    "      end try\n"
    "    end repeat\n"
    "  end repeat\n"
    "  set AppleScript's text item delimiters to linefeed\n"
    "  return rows as text\n"
    "end tell"
)

_MACOS_FOREGROUND_SCRIPT = (
    'tell application "System Events"\n'
    "  set proc to first application process whose frontmost is true\n"
    "  set win to front window of proc\n"
    "  set p to position of win\n"
    "  set s to size of win\n"
    "  return ((name of proc) & tab & (name of win) & tab & (item 1 of p) & tab & "
    "(item 2 of p) & tab & (item 1 of s) & tab & (item 2 of s))\n"
    "end tell"
)


def _run_osascript(script: str) -> str:
    result = subprocess.run(  # noqa: S603 - fixed osascript invocation, no shell
        ["osascript", "-e", script],  # noqa: S607
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"osascript failed: {result.stderr.strip()}")
    return result.stdout


def _macos_window_from_row(row: str) -> WindowInfo | None:
    parts = row.split("\t")
    if len(parts) != 6:
        return None
    app, title, x, y, w, h = parts
    try:
        left, top, width, height = (int(round(float(v))) for v in (x, y, w, h))
    except ValueError:
        return None
    label = f"{app.strip()} - {title.strip()}".strip(" -")
    return WindowInfo(title=label, left=left, top=top, width=width, height=height)


def _macos_list_windows() -> list[WindowInfo]:
    windows: list[WindowInfo] = []
    for row in _run_osascript(_MACOS_LIST_SCRIPT).splitlines():
        window = _macos_window_from_row(row.strip())
        if window is not None and window.area > 0:
            windows.append(window)
    return windows


def _macos_foreground_window() -> WindowInfo | None:
    output = _run_osascript(_MACOS_FOREGROUND_SCRIPT).strip()
    return _macos_window_from_row(output) if output else None


# --- Windows (Win32 via ctypes) --------------------------------------------------------------


def _windows_list_windows() -> list[WindowInfo]:  # pragma: no cover - host-only
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    windows: list[WindowInfo] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)  # type: ignore[attr-defined, untyped-decorator]
    def _collect(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return True
        windows.append(
            WindowInfo(
                title=buffer.value,
                left=rect.left,
                top=rect.top,
                width=rect.right - rect.left,
                height=rect.bottom - rect.top,
            )
        )
        return True

    user32.EnumWindows(_collect, 0)
    return [window for window in windows if window.area > 0]


def _windows_foreground_window() -> WindowInfo | None:  # pragma: no cover - host-only
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    return WindowInfo(
        title=buffer.value,
        left=rect.left,
        top=rect.top,
        width=rect.right - rect.left,
        height=rect.bottom - rect.top,
    )


def find_window_main(argv: Sequence[str] | None = None) -> None:
    """Probe CLI: list/select the game window and optionally capture it for calibration."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Find the Poker Legends window (and optionally capture it) for calibration."
    )
    parser.add_argument("--title", default="Poker Legends", help="window-title substring to match")
    parser.add_argument("--list", action="store_true", help="print every visible window first")
    parser.add_argument("--capture", help="grab the selected window region with mss into this PNG")
    args = parser.parse_args(argv)

    windows = list_windows()
    foreground = foreground_window()
    if args.list:
        for window in windows:
            rect = f"{window.width}x{window.height} @ ({window.left},{window.top})"
            print(f"  {rect}  {window.title}")
        print(f"foreground: {foreground}")
    selected = select_window(windows, foreground, title_substring=args.title)
    payload = None if selected is None else {"title": selected.title, **selected.region()}
    print(json.dumps({"selected": payload}, ensure_ascii=False, indent=2))
    if args.capture and selected is not None:
        import mss
        import mss.tools

        with mss.mss() as sct:
            shot = sct.grab(selected.region())
            mss.tools.to_png(shot.rgb, shot.size, output=args.capture)
        print(f"captured region {selected.width}x{selected.height} -> {args.capture} "
              f"(mss returned {shot.size[0]}x{shot.size[1]})")
