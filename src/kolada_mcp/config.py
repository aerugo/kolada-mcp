"""Configuration settings for Kolada MCP Server."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# API Constants
KOLADA_BASE_URL = "https://api.kolada.se/v2"
KOLADA_PAGE_SIZE = 5000
EMBEDDINGS_CACHE_FILE = "kpi_embeddings.npz"

# Model Constants
SENTENCE_MODEL_NAME = "KBLab/sentence-bert-swedish-cased"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MCP Transport configuration
    mcp_transport: Literal["stdio", "http"] = "stdio"
    port: int = 8001

    # Kolada API configuration
    kolada_base_url: str = KOLADA_BASE_URL
    kolada_page_size: int = KOLADA_PAGE_SIZE

    # Embeddings configuration
    embeddings_cache_path: Path = Path(__file__).parent / EMBEDDINGS_CACHE_FILE
    sentence_model_name: str = SENTENCE_MODEL_NAME

    # Retry configuration
    max_retries: int = 3
    retry_base_delay: float = 1.0

    # Logging
    log_level: str = "INFO"


settings = Settings()
