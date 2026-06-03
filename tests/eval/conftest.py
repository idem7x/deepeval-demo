"""
Shared fixtures for the DeepEval suite.

Every test that grades model output needs a *judge* model. We make that
one shared object (`judge`) so it's obvious what's doing the scoring
across all suites — and so cost-per-run is predictable. Default judge:
gpt-4o-mini, temperature=0 (cheap, deterministic-ish).

Tests that need a chat answer (RAG, conversational, custom) use the
`chat_adapter` fixture and the shared `service.answer()` helper. That
keeps the SUT identical to what the /chat HTTP endpoint produces, so
metric scores reflect production behaviour rather than test-specific
prompt drift.
"""

from __future__ import annotations

import os
from collections import defaultdict

import pytest

from apps.backend.llm.deepeval_wrap import DeepEvalLLM
from apps.backend.llm.openai import OpenAIAdapter


# Per-test list of metric outputs. Populated by the autouse fixture below
# that wraps BaseMetric.measure / a_measure, then drained and printed by
# the pytest_runtest_makereport hook. Visible both in `-s` stdout and in
# the pytest-html report's "Captured stdout call" section.
_metric_records: dict[str, list[dict]] = defaultdict(list)


def _all_metric_subclasses(root):
    """Walk every subclass of BaseMetric (including grand-children)."""
    seen: set[type] = set()
    stack = [root]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
                yield sub


def _shorten(s: str | None, limit: int = 500) -> str | None:
    if s is None:
        return None
    s = str(s).strip()
    if len(s) <= limit:
        return s
    return s[:limit].rstrip() + " …"


def _extract_io(test_case) -> dict:
    """Pull input/output from whatever TestCase variant was passed to measure()."""
    if test_case is None:
        return {}
    # Multi-turn: ConversationalTestCase has `turns: list[Turn]`.
    turns = getattr(test_case, "turns", None)
    if turns:
        user_turns = [t for t in turns if getattr(t, "role", None) == "user"]
        assistant_turns = [t for t in turns if getattr(t, "role", None) == "assistant"]
        return {
            "input": " | ".join(getattr(t, "content", "") for t in user_turns),
            "actual_output": " | ".join(getattr(t, "content", "") for t in assistant_turns),
        }
    # Arena: ArenaTestCase has `contestants` (list of Contestant with nested test_case).
    contestants = getattr(test_case, "contestants", None)
    if contestants:
        first = contestants[0]
        inner = getattr(first, "test_case", None)
        return _extract_io(inner)
    # Single-turn LLMTestCase
    return {
        "input": getattr(test_case, "input", None),
        "actual_output": getattr(test_case, "actual_output", None),
        "expected_output": getattr(test_case, "expected_output", None),
    }


def _snapshot(self, test_case=None) -> dict:
    try:
        success = self.is_successful()
    except Exception:
        success = None
    return {
        "metric": type(self).__name__,
        "score": getattr(self, "score", None),
        "threshold": getattr(self, "threshold", None),
        "reason": getattr(self, "reason", None),
        "success": success,
        **_extract_io(test_case),
    }


def _emit(snap: dict) -> None:
    score = snap["score"]
    score_s = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
    thr = snap["threshold"]
    thr_s = f" (thr {thr})" if thr is not None else ""
    success_s = "PASS" if snap["success"] else "FAIL" if snap["success"] is False else "?"
    print(f"\n[{success_s}] {snap['metric']} score={score_s}{thr_s}")
    if snap.get("input"):
        print(f"   input:  {_shorten(snap['input'])}")
    if snap.get("actual_output"):
        print(f"   output: {_shorten(snap['actual_output'])}")
    if snap.get("expected_output"):
        print(f"   expected: {_shorten(snap['expected_output'])}")
    if snap["reason"]:
        print(f"   reason: {_shorten(snap['reason'], 800)}")


# Single mutable cell so the wrappers can find the active test nodeid without
# every metric class needing a fresh closure per test.
_current_nodeid: list[str | None] = [None]

# Re-entrancy depth per measurement. DeepEval's sync `measure` often
# delegates to `a_measure` internally; we only want to emit on the
# outermost call so each `metric.measure(tc)` produces one record.
_measure_depth: list[int] = [0]


def _peek_test_case(args, kwargs):
    if args:
        return args[0]
    return kwargs.get("test_case")


def _wrap_sync(original):
    def patched(self, *args, **kwargs):
        tc = _peek_test_case(args, kwargs)
        _measure_depth[0] += 1
        try:
            return original(self, *args, **kwargs)
        finally:
            _measure_depth[0] -= 1
            if _measure_depth[0] == 0:
                snap = _snapshot(self, tc)
                nodeid = _current_nodeid[0]
                if nodeid:
                    _metric_records[nodeid].append(snap)
                _emit(snap)
    return patched


def _wrap_async(original):
    async def patched(self, *args, **kwargs):
        tc = _peek_test_case(args, kwargs)
        _measure_depth[0] += 1
        try:
            return await original(self, *args, **kwargs)
        finally:
            _measure_depth[0] -= 1
            if _measure_depth[0] == 0:
                snap = _snapshot(self, tc)
                nodeid = _current_nodeid[0]
                if nodeid:
                    _metric_records[nodeid].append(snap)
                _emit(snap)
    return patched


@pytest.fixture(autouse=True)
def _record_metric_results(request, monkeypatch):
    """Patch every BaseMetric subclass's `measure` / `a_measure` to record."""
    from deepeval.metrics import BaseMetric

    _current_nodeid[0] = request.node.nodeid

    # Patch concrete subclasses only — patching BaseMetric too would
    # double-print whenever a subclass calls `super().measure(...)`.
    for cls in _all_metric_subclasses(BaseMetric):
        if "measure" in cls.__dict__:
            monkeypatch.setattr(cls, "measure", _wrap_sync(cls.measure), raising=False)
        if "a_measure" in cls.__dict__:
            monkeypatch.setattr(cls, "a_measure", _wrap_async(cls.a_measure), raising=False)
    yield
    _current_nodeid[0] = None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when != "call":
        return
    records = _metric_records.pop(item.nodeid, [])
    if not records:
        return
    lines = ["", "── metric results ──"]
    for r in records:
        score = r["score"]
        score_s = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
        threshold = r["threshold"]
        thr_s = f" (thr {threshold})" if threshold is not None else ""
        success_s = "PASS" if r["success"] else "FAIL" if r["success"] is False else "?"
        lines.append(f"  [{success_s}] {r['metric']:<28} score={score_s}{thr_s}")
        if r.get("input"):
            lines.append(f"         input:    {_shorten(r['input'])}")
        if r.get("actual_output"):
            lines.append(f"         output:   {_shorten(r['actual_output'])}")
        if r.get("expected_output"):
            lines.append(f"         expected: {_shorten(r['expected_output'])}")
        if r["reason"]:
            lines.append(f"         reason:   {_shorten(r['reason'], 800)}")
    rep.sections.append(("captured metrics", "\n".join(lines)))


JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
SUT_MODEL = os.environ.get("SUT_MODEL", "gpt-4o-mini")


def _require_openai_key() -> None:
    """Skip the suite cleanly if no OpenAI key is configured.

    These tests cost money — we never want them silently passing in CI
    without the budget being approved via the OPENAI_API_KEY secret.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        # Fall back to the .env loader (settings reads it on import).
        from apps.backend.core.settings import settings  # noqa: WPS433
        if not settings.openai_api_key:
            pytest.skip("OPENAI_API_KEY not set; skipping live eval suite.")


@pytest.fixture(scope="session")
def chat_adapter() -> OpenAIAdapter:
    _require_openai_key()
    return OpenAIAdapter()


@pytest.fixture(scope="session")
def judge(chat_adapter: OpenAIAdapter) -> DeepEvalLLM:
    """The model that grades. We reuse `chat_adapter`'s underlying client
    to keep connection pooling tight — but bind a different (cheaper,
    fixed-temperature) ChatOptions via the wrapper."""
    return DeepEvalLLM(chat_adapter, model=JUDGE_MODEL, temperature=0.0)


@pytest.fixture(scope="session")
def sut_model() -> str:
    """Model name used as the system-under-test in eval tests."""
    return SUT_MODEL
