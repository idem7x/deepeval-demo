"""
Shared pytest fixtures for the DeepEval Lab test suite.

This file is intentionally tiny in Phase 0 — it grows as we add real metrics.
Right now it only enforces the smoke-mode case cap so CI cannot blow the
budget if someone accidentally adds a huge dataset.
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Cap the number of `smoke` tests when running with `-m smoke`.

    Keeps CI cost predictable. The cap comes from $SMOKE_MAX_CASES (default 15).
    """
    marker_expr = config.getoption("-m") or ""
    if "smoke" not in marker_expr:
        return

    cap = int(os.environ.get("SMOKE_MAX_CASES", "15"))
    smoke_items = [i for i in items if i.get_closest_marker("smoke")]
    if len(smoke_items) <= cap:
        return

    skip = pytest.mark.skip(reason=f"capped by SMOKE_MAX_CASES={cap}")
    for item in smoke_items[cap:]:
        item.add_marker(skip)
