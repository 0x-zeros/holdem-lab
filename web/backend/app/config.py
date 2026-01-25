"""Application configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    debug: bool = False
    default_simulations: int = 10000
    max_simulations: int = 1000000

    class Config:
        env_prefix = "EQUITY_"


settings = Settings()
