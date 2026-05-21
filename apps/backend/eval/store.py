"""
SQLite store for DeepEval matrix results.

Two tables:

- `runs`         — one row per matrix invocation (a "session": same git
                   commit + same set of providers exercised). Holds
                   timestamps, summary counts, and the optional CLI args.
- `results`      — one row per (run, sut, judge, suite, metric, case).
                   This is the analytic surface the dashboard reads.

Why plain sqlite3 over SQLModel/SQLAlchemy?
- Zero migration ceremony; one file, no Alembic.
- The schema is small and won't sprout joins.
- The dashboard only reads — a few SELECTs are clearer than ORM gymnastics.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from apps.backend.core.settings import settings


DB_PATH = Path("eval_results") / "runs.sqlite3"


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    started_at    REAL NOT NULL,
    finished_at   REAL,
    sut_combos    TEXT NOT NULL,         -- json: [[provider, model], ...]
    judge_combos  TEXT NOT NULL,         -- json: [[provider, model], ...]
    suites        TEXT NOT NULL,         -- json: ["rag", "single_turn", ...]
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS results (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL,
    ts             REAL NOT NULL,
    sut_provider   TEXT NOT NULL,
    sut_model      TEXT NOT NULL,
    judge_provider TEXT NOT NULL,
    judge_model    TEXT NOT NULL,
    suite          TEXT NOT NULL,
    metric         TEXT NOT NULL,
    case_id        TEXT NOT NULL,
    score          REAL,
    threshold      REAL,
    passed         INTEGER,              -- 0/1 (NULL when metric has no threshold)
    reason         TEXT,
    duration_ms    INTEGER,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_results_run    ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_results_sut    ON results(sut_provider, sut_model);
CREATE INDEX IF NOT EXISTS idx_results_suite  ON results(suite, metric);
"""


@dataclass(slots=True)
class Result:
    """One metric measurement on one case in one (SUT, judge) combination."""

    sut_provider: str
    sut_model: str
    judge_provider: str
    judge_model: str
    suite: str
    metric: str
    case_id: str
    score: float | None
    threshold: float | None
    passed: bool | None
    reason: str | None
    duration_ms: int


# ---------------------------------------------------------------------------


@contextmanager
def _connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    p = path or _resolved_db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _resolved_db_path() -> Path:
    # Honour an absolute override via env, else relative to repo root.
    # settings doesn't carry this knob today; we read directly so callers
    # can shadow it in tests without touching the global.
    import os
    override = os.environ.get("EVAL_DB_PATH")
    return Path(override) if override else DB_PATH


# --- writes ----------------------------------------------------------------


def start_run(
    sut_combos: list[tuple[str, str]],
    judge_combos: list[tuple[str, str]],
    suites: list[str],
    notes: str | None = None,
) -> str:
    """Create a new run row and return its run_id."""
    run_id = uuid.uuid4().hex[:12]
    with _connect() as conn:
        conn.execute(
            "INSERT INTO runs(run_id, started_at, sut_combos, judge_combos, suites, notes)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                time.time(),
                json.dumps([list(s) for s in sut_combos]),
                json.dumps([list(j) for j in judge_combos]),
                json.dumps(suites),
                notes,
            ),
        )
    return run_id


def finish_run(run_id: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE runs SET finished_at = ? WHERE run_id = ?",
            (time.time(), run_id),
        )


def record(run_id: str, r: Result) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO results("
            " run_id, ts, sut_provider, sut_model, judge_provider, judge_model,"
            " suite, metric, case_id, score, threshold, passed, reason, duration_ms"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                time.time(),
                r.sut_provider, r.sut_model,
                r.judge_provider, r.judge_model,
                r.suite, r.metric, r.case_id,
                r.score, r.threshold,
                int(r.passed) if r.passed is not None else None,
                r.reason, r.duration_ms,
            ),
        )


# --- reads -----------------------------------------------------------------


def list_runs(limit: int = 50) -> list[dict]:
    """Return runs with summary counts (newest first)."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                r.run_id, r.started_at, r.finished_at,
                r.sut_combos, r.judge_combos, r.suites, r.notes,
                COUNT(res.id)                     AS results_count,
                SUM(CASE WHEN res.passed=1 THEN 1 ELSE 0 END) AS passed_count,
                AVG(res.score)                    AS avg_score
            FROM runs r
            LEFT JOIN results res ON res.run_id = r.run_id
            GROUP BY r.run_id
            ORDER BY r.started_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    out: list[dict] = []
    for row in rows:
        d = dict(row)
        for k in ("sut_combos", "judge_combos", "suites"):
            d[k] = json.loads(d[k])
        out.append(d)
    return out


def get_run(run_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT run_id, started_at, finished_at, sut_combos, judge_combos, suites, notes"
            " FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    for k in ("sut_combos", "judge_combos", "suites"):
        d[k] = json.loads(d[k])
    return d


def list_results(
    run_id: str | None = None,
    sut_provider: str | None = None,
    sut_model: str | None = None,
    suite: str | None = None,
    limit: int = 2000,
) -> list[dict]:
    """Filtered fetch. Used by the dashboard's drill-down view."""
    clauses: list[str] = []
    args: list = []
    for col, val in (
        ("run_id", run_id),
        ("sut_provider", sut_provider),
        ("sut_model", sut_model),
        ("suite", suite),
    ):
        if val:
            clauses.append(f"{col} = ?")
            args.append(val)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    args.append(limit)
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM results {where} ORDER BY ts DESC LIMIT ?",
            args,
        ).fetchall()
    return [dict(r) for r in rows]
