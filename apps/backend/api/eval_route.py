"""
Eval routes.

GET /eval/runs                 — list runs, newest first, with summary counts
GET /eval/runs/{run_id}        — run metadata + every result row for drill-down
GET /eval/runs/{run_id}/by-cell — per-(sut, judge, suite, metric) aggregates,
                                  the shape the dashboard's matrix view binds to
"""

from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter, HTTPException, Query

from apps.backend.eval import store


router = APIRouter(prefix="/eval", tags=["eval"])


@router.get("/runs")
async def list_runs(limit: int = Query(50, ge=1, le=500)) -> dict[str, object]:
    runs = store.list_runs(limit=limit)
    return {"runs": runs}


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, object]:
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    results = store.list_results(run_id=run_id, limit=5000)
    return {"run": run, "results": results}


@router.get("/runs/{run_id}/by-cell")
async def get_run_aggregates(run_id: str) -> dict[str, object]:
    """Aggregate per (sut, judge, suite, metric).

    The frontend matrix view groups by SUT+model on rows and suite+metric
    on columns; the heatmap cell is the average score.
    """
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    rows = store.list_results(run_id=run_id, limit=5000)

    cells: dict[tuple, dict] = defaultdict(lambda: {"pass": 0, "total": 0, "scores": []})
    for r in rows:
        key = (
            f"{r['sut_provider']}/{r['sut_model']}",
            f"{r['judge_provider']}/{r['judge_model']}",
            r["suite"],
            r["metric"],
        )
        c = cells[key]
        c["total"] += 1
        if r["passed"]:
            c["pass"] += 1
        if r["score"] is not None:
            c["scores"].append(r["score"])

    out = []
    for (sut, judge, suite, metric), v in cells.items():
        avg = sum(v["scores"]) / len(v["scores"]) if v["scores"] else None
        out.append({
            "sut": sut,
            "judge": judge,
            "suite": suite,
            "metric": metric,
            "passed": v["pass"],
            "total": v["total"],
            "avg_score": avg,
        })
    out.sort(key=lambda c: (c["sut"], c["judge"], c["suite"], c["metric"]))
    return {"run": run, "cells": out}
