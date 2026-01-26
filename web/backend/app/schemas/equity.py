"""Equity calculation schemas."""

from typing import Literal
from pydantic import BaseModel, Field


class PlayerHandInput(BaseModel):
    """Input for a player's hand (specific cards, range, or random)."""

    cards: list[str] | None = None  # ["Ah", "Kh"]
    range: list[str] | None = None  # ["QQ+", "AKs"]
    random: bool = False  # True = random hand sampled each simulation


class EquityRequest(BaseModel):
    """Request to calculate equity."""

    players: list[PlayerHandInput] = Field(..., min_length=1, max_length=10)
    board: list[str] | None = None  # ["7h", "6c", "2d"]
    dead_cards: list[str] | None = None  # ["2h"]
    num_simulations: int = Field(default=10000, ge=100, le=1000000)
    mode: Literal["monte_carlo", "enumerate"] = "monte_carlo"


class PlayerEquityResult(BaseModel):
    """Result for a single player."""

    index: int
    hand_description: str  # "AhKh" or "QQ+, AKs (40 combos)"
    equity: float  # 0.0 - 1.0
    win_rate: float
    tie_rate: float
    combos: int


class EquityResponse(BaseModel):
    """Response with equity calculation results."""

    players: list[PlayerEquityResult]
    total_simulations: int
    elapsed_ms: float
