"""Health check endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

import holdem_lab

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    holdem_lab_version: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        holdem_lab_version=holdem_lab.__version__,
    )
