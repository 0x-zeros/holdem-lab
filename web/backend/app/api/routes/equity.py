"""Equity calculation endpoint."""

from fastapi import APIRouter, HTTPException

from app.schemas.equity import EquityRequest, EquityResponse
from app.services.holdem import HoldemService

router = APIRouter()


@router.post("/equity", response_model=EquityResponse)
async def calculate_equity(request: EquityRequest) -> EquityResponse:
    """Calculate equity for given players."""
    try:
        result = HoldemService.calculate_equity(
            players=request.players,
            board=request.board,
            dead_cards=request.dead_cards,
            num_simulations=request.num_simulations,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculation error: {str(e)}")
