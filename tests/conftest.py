"""
Shared pytest fixtures + the smoke-mode case cap.

The cap (`$SMOKE_MAX_CASES`, default 50) protects CI cost from
**parametrized** eval tests that explode with data — e.g. Phase 6 will
parametrize one test over 30+ goldens. Without a cap, expanding the
dataset can silently 10× the bill.

Crucially, the cap is applied **per test function**, only to functions
that use `pytest.mark.parametrize`. Non-parametrized tests are never
touched, so adding a regular assertion test doesn't risk eating into
the cap.
"""

from __future__ import annotations

import os
from collections import defaultdict

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    marker_expr = config.getoption("-m") or ""
    if "smoke" not in marker_expr:
        return

    cap = int(os.environ.get("SMOKE_MAX_CASES", "50"))

    # Group by parent test function (one nodeid prefix). Non-parametrized
    # tests have exactly one item per group; they're never trimmed.
    groups: dict[str, list[pytest.Item]] = defaultdict(list)
    for item in items:
        if not item.get_closest_marker("smoke"):
            continue
        # Parametrized id is "tests/foo.py::test_name[case]"; the prefix
        # before "[" is the function's nodeid.
        key = item.nodeid.split("[", 1)[0]
        groups[key].append(item)

    skip = pytest.mark.skip(reason=f"capped by SMOKE_MAX_CASES={cap}")
    for key, group in groups.items():
        if len(group) <= cap:
            continue
        for item in group[cap:]:
            item.add_marker(skip)
