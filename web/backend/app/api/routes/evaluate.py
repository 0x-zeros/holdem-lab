"""Hand evaluation endpoint."""

from fastapi import APIRouter, HTTPException

from app.schemas.evaluate import EvaluateRequest, EvaluateResponse
from app.services.holdem import HoldemService

router = APIRouter()


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_hand(request: EvaluateRequest) -> EvaluateResponse:
    """Evaluate 5-7 cards and return the best hand."""
    try:
        result = HoldemService.evaluate_hand_cards(cards=request.cards)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")
