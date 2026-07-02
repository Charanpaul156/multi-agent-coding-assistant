"""Application configuration.

Uses python-dotenv and pydantic-settings for validation.
"""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


@lru_cache(maxsize=1)
def _load_env() -> None:
    load_dotenv()


class Settings(BaseSettings):
    """Strongly-typed configuration (placeholders)."""

    openai_api_key: str | None = None

    fastapi_host: str = "0.0.0.0"
    fastapi_port: int = 8000

    model_config = SettingsConfigDict(env_prefix="")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_env()
    return Settings()

