#!/usr/bin/env python
"""Development server startup script.

Usage:
    python run.py                     # Use default port (8000)
    EQUITY_PORT=9000 python run.py    # Use custom port
"""

import uvicorn

from app.config import settings

if __name__ == "__main__":
    print(f"Starting server on {settings.host}:{settings.port}")
    print(f"Frontend CORS allowed on port: {settings.frontend_port}")

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
