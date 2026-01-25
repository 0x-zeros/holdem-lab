"""Card-related schemas."""

from pydantic import BaseModel


class CardInfo(BaseModel):
    """Information about a single card."""

    notation: str  # "Ah"
    rank: str  # "A"
    suit: str  # "h"
    suit_symbol: str  # "♥"


class ParseCardsRequest(BaseModel):
    """Request to parse card string."""

    input: str  # "AhKh" or "Ah Kh"


class ParseCardsResponse(BaseModel):
    """Response with parsed cards."""

    cards: list[CardInfo]
    formatted: str  # "Ah Kh"
    valid: bool
    error: str | None = None


class CanonicalHandInfo(BaseModel):
    """Information about a canonical hand."""

    notation: str  # "AA", "AKs", "AKo"
    high_rank: str  # "A"
    low_rank: str  # "K"
    suited: bool
    is_pair: bool
    num_combos: int  # 6, 4, or 12
    matrix_row: int  # 0-12
    matrix_col: int  # 0-12


class CanonicalHandsResponse(BaseModel):
    """Response with all canonical hands."""

    hands: list[CanonicalHandInfo]
    total: int  # Always 169
