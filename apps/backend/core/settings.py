"""
Application settings, loaded from environment variables (and `.env`).

We keep everything in one place so the rest of the codebase can simply do
`from apps.backend.core.settings import settings`. Every value has a sensible
default so the app can boot even with an empty `.env` — features that need a
real key (e.g. OpenAI calls) will fail at the point of use with a clear error,
not at import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM providers --------------------------------------------------------
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # --- DeepEval / Confident AI ---------------------------------------------
    confident_api_key: str = ""

    # --- RAG ------------------------------------------------------------------
    chroma_path: Path = Field(default=Path("./.chroma"))
    # `chromadb` is the offline default (ONNX MiniLM, no key, no Ollama).
    # Switch to `ollama` or `openai` for higher-quality embeddings.
    embedding_provider: Literal["chromadb", "ollama", "openai"] = "chromadb"
    embedding_model: str = "nomic-embed-text"

    # --- App ------------------------------------------------------------------
    log_level: str = "INFO"
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000

    # --- CI safety knob -------------------------------------------------------
    smoke_max_cases: int = 15


settings = Settings()
