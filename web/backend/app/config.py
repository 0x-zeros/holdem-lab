"""Application configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    debug: bool = False
    default_simulations: int = 10000
    max_simulations: int = 1000000

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # Frontend port (for CORS)
    frontend_port: int = 5173

    model_config = {"env_prefix": "EQUITY_"}


settings = Settings()
