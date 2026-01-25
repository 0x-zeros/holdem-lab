"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, cards, equity, draws

app = FastAPI(
    title="Equity Calculator API",
    description="Texas Hold'em equity calculator powered by holdem-lab",
    version="0.1.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative dev port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
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
