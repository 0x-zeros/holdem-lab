"""Card-related endpoints."""

from fastapi import APIRouter, HTTPException

from app.schemas.card import (
    ParseCardsRequest,
    ParseCardsResponse,
    CanonicalHandsResponse,
)
from app.services.holdem import HoldemService

router = APIRouter()


@router.get("/canonical", response_model=CanonicalHandsResponse)
async def get_canonical_hands() -> CanonicalHandsResponse:
    """Get all 169 canonical hands with matrix positions."""
    hands = HoldemService.get_all_canonical_hands()
    return CanonicalHandsResponse(hands=hands, total=len(hands))


@router.post("/parse-cards", response_model=ParseCardsResponse)
async def parse_cards(request: ParseCardsRequest) -> ParseCardsResponse:
    """Parse card string and return card information."""
    result = HoldemService.parse_cards_safe(request.input)
    if not result.valid:
        raise HTTPException(status_code=400, detail=result.error)
    return result
