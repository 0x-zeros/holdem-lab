"""Capture abstractions for bot input sources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    payload: object
    source: str
    metadata: Mapping[str, object] = field(default_factory=dict)


class Capture(Protocol):
    def capture(self) -> CapturedFrame:
        """Return one raw frame or state snapshot."""
        ...
