"""Draw analysis schemas."""

from typing import Literal
from pydantic import BaseModel, Field


class DrawsRequest(BaseModel):
    """Request to analyze draws."""

    hole_cards: list[str] = Field(..., min_length=2, max_length=2)  # ["9h", "8h"]
    board: list[str] = Field(..., min_length=3, max_length=5)  # ["7h", "6c", "2h"]
    dead_cards: list[str] | None = None


class FlushDrawInfo(BaseModel):
    """Information about a flush draw."""

    suit: str  # "h"
    suit_symbol: str  # "♥"
    cards_held: int  # 3 or 4
    outs: list[str]  # ["Ah", "Kh", ...]
    out_count: int
    is_nut: bool
    draw_type: Literal["flush_draw", "backdoor_flush"]


class StraightDrawInfo(BaseModel):
    """Information about a straight draw."""

    draw_type: Literal["open_ended", "gutshot", "double_gutshot", "backdoor_straight"]
    needed_ranks: list[int]  # [10, 5] for OESD
    outs: list[str]  # ["Ts", "Th", "Tc", "Td", ...]
    out_count: int
    high_card: int  # Highest card of completed straight
    is_nut: bool


class DrawsResponse(BaseModel):
    """Response with draw analysis."""

    has_flush: bool
    has_straight: bool
    flush_draws: list[FlushDrawInfo]
    straight_draws: list[StraightDrawInfo]
    total_outs: int
    all_outs: list[str]  # Unique outs
    is_combo_draw: bool
