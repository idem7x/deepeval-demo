"""
Embedding providers.

Three back-ends, chosen via `EMBEDDING_PROVIDER` in `.env`:

- `chromadb` (default) — ChromaDB's built-in `DefaultEmbeddingFunction`,
  which uses an ONNX-quantised `all-MiniLM-L6-v2` model (~30 MB). Pure
  Python deps; no torch, no Ollama, no API key. Works offline and in CI.
  Vector dim: 384. Quality is "good enough for a demo".

- `ollama` — uses a local Ollama instance with `nomic-embed-text` (or
  whichever model is set via `EMBEDDING_MODEL`). Higher quality than the
  default; needs `ollama serve` running.
  Vector dim: 768 (for nomic-embed-text).

- `openai` — `text-embedding-3-small` by default. Costs money but is fast
  and high quality. The right choice for cost-bounded CI matrices.
  Vector dim: 1536.

Switching providers changes the *vector space*, so re-running ingestion is
mandatory after a switch — you cannot mix vectors from different
providers in the same collection. Phase 2.3 (ChromaStore) enforces this
by namespacing the collection by provider+model.
"""

from __future__ import annotations

from typing import Protocol

from apps.backend.core.settings import settings


class EmbeddingProvider(Protocol):
    """The single contract for everything that turns text into vectors."""

    name: str   # e.g. "chromadb:default", "ollama:nomic-embed-text"
    dim: int    # vector dimension; useful for sanity checks

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...


class ChromaDefaultEmbedder:
    """Wraps ChromaDB's bundled `DefaultEmbeddingFunction` (ONNX MiniLM)."""

    name = "chromadb:default"
    dim = 384

    def __init__(self) -> None:
        # Imported lazily so the module doesn't pay the cost unless used.
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Chroma's function returns numpy arrays — coerce to plain lists so
        # the downstream contract is JSON-serialisable.
        vectors = self._fn(texts)
        return [list(map(float, v)) for v in vectors]


class OllamaEmbedder:
    """Embeddings via a local Ollama server."""

    dim_by_model = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
    }

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        import ollama

        self._model = model or settings.embedding_model
        self._client = ollama.Client(host=base_url or settings.ollama_base_url)
        self.name = f"ollama:{self._model}"
        self.dim = self.dim_by_model.get(self._model, 0)  # 0 = unknown; we'll trust the server

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            # Ollama doesn't accept batches for /api/embeddings, so loop.
            resp = self._client.embeddings(model=self._model, prompt=t)
            out.append(list(resp["embedding"]))
        if out and self.dim == 0:
            self.dim = len(out[0])
        return out


class OpenAIEmbedder:
    """OpenAI text-embedding-3-small / -large."""

    dim_by_model = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        from openai import OpenAI

        self._model = model or "text-embedding-3-small"
        key = api_key or settings.openai_api_key
        if not key:
            raise RuntimeError(
                "OpenAIEmbedder requires OPENAI_API_KEY. "
                "Set it in .env or switch EMBEDDING_PROVIDER."
            )
        self._client = OpenAI(api_key=key)
        self.name = f"openai:{self._model}"
        self.dim = self.dim_by_model.get(self._model, 1536)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # OpenAI batches up to ~2048 inputs per request; keep it generous
        # but split very large jobs to stay well under the request limit.
        out: list[list[float]] = []
        for i in range(0, len(texts), 256):
            batch = texts[i : i + 256]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            out.extend(d.embedding for d in resp.data)
        return out


def get_embedder(provider: str | None = None, model: str | None = None) -> EmbeddingProvider:
    """Factory used by ChromaStore. Defaults come from settings (env)."""
    provider = (provider or settings.embedding_provider).lower()

    if provider == "chromadb":
        return ChromaDefaultEmbedder()
    if provider == "ollama":
        return OllamaEmbedder(model=model)
    if provider == "openai":
        return OpenAIEmbedder(model=model)
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER={provider!r}. "
        "Choose one of: chromadb, ollama, openai."
    )
