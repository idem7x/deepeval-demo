"""GET /models — which providers/models can the UI offer right now?"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from apps.backend.api.deps import get_registry
from apps.backend.llm.registry import ModelRegistry


router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
async def list_models(
    refresh: bool = Query(False, description="Bypass the 30s registry cache."),
    registry: ModelRegistry = Depends(get_registry),
) -> dict[str, object]:
    """Return per-provider availability + model list.

    Shape (intentionally stable for the UI to bind to):

        {
            "providers": [
                {
                    "provider": "openai",
                    "available": true,
                    "default_model": "gpt-4o-mini",
                    "models": [{"id": "...", "label": "..."}, ...],
                    "error": null
                },
                ...
            ]
        }
    """
    statuses = await registry.describe(force_refresh=refresh)
    return {"providers": [s.to_json() for s in statuses]}
