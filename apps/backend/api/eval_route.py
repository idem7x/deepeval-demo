"""
GET /eval/runs — placeholder.

Phase 4 just exposes the empty surface so the frontend can render the
dashboard page without 404s. Persistence and POST /eval/run land in
Phase 7 along with the eval matrix runner.
"""

from __future__ import annotations

from fastapi import APIRouter


router = APIRouter(prefix="/eval", tags=["eval"])


@router.get("/runs")
async def list_runs() -> dict[str, object]:
    """List recorded eval runs. Empty until Phase 7 plugs in storage."""
    return {"runs": [], "_note": "Persistence and POST /eval/run land in Phase 7."}
