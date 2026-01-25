"""Draw analysis endpoint."""

from fastapi import APIRouter, HTTPException

from app.schemas.draws import DrawsRequest, DrawsResponse
from app.services.holdem import HoldemService

router = APIRouter()


@router.post("/draws", response_model=DrawsResponse)
async def analyze_draws(request: DrawsRequest) -> DrawsResponse:
    """Analyze draws for given hole cards and board."""
    try:
        result = HoldemService.analyze_draws(
            hole_cards=request.hole_cards,
            board=request.board,
            dead_cards=request.dead_cards,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
