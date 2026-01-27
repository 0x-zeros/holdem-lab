"""Schemas for hand evaluation endpoints."""

from pydantic import BaseModel, Field


class EvaluateRequest(BaseModel):
    """Request for hand evaluation."""

    cards: list[str] = Field(
        ...,
        min_length=5,
        max_length=7,
        description="5-7 cards to evaluate (e.g., ['Ah', 'Kh', 'Qh', 'Jh', 'Th'])",
    )


class EvaluateResponse(BaseModel):
    """Response from hand evaluation."""

    hand_type: str = Field(..., description="Hand type name (e.g., 'Royal Flush')")
    description: str = Field(
        ..., description="Detailed description (e.g., 'Royal Flush')"
    )
    primary_ranks: list[int] = Field(
        ..., description="Primary ranks that define the hand"
    )
    kickers: list[int] = Field(..., description="Kicker cards for tiebreaking")
