"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, cards, equity, draws
from app.config import settings

app = FastAPI(
    title="Equity Calculator API",
    description="Texas Hold'em equity calculator powered by holdem-lab",
    version="0.1.0",
)

# CORS configuration - dynamic port from settings
frontend_port = settings.frontend_port
cors_origins = [
    f"http://localhost:{frontend_port}",
    f"http://127.0.0.1:{frontend_port}",
    "http://localhost:3000",  # Alternative dev port
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(cards.router, prefix="/api", tags=["cards"])
app.include_router(equity.router, prefix="/api", tags=["equity"])
app.include_router(draws.router, prefix="/api", tags=["draws"])


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Equity Calculator API",
        "version": "0.1.0",
        "docs": "/docs",
    }
