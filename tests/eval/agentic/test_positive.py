"""
Agentic metric suite — for grading tool-using agents.

We don't run a real LangChain/MCP agent loop in smoke; instead we
hand-craft `tools_called` lists on `LLMTestCase` to demonstrate the
metrics deterministically and cheaply. The same metric API plugs into
a real agent later (Phase 8+) without changes.

Metrics covered:
- TaskCompletionMetric — did the agent finish the user's task?
- ToolCorrectnessMetric — were the right tools picked vs. expected?
- ArgumentCorrectnessMetric — were tool arguments well-formed?
"""

from __future__ import annotations

import pytest
from deepeval.metrics import (
    ArgumentCorrectnessMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from deepeval.test_case import LLMTestCase, ToolCall


# ---------------------------------------------------------------------------
# A small toolbox the simulated agent has access to.
# These match what a property-buying assistant would plausibly carry.
# ---------------------------------------------------------------------------


AVAILABLE_TOOLS = [
    ToolCall(name="search_listings", description="Search property listings by region and price band."),
    ToolCall(name="mortgage_calc", description="Compute monthly mortgage payment from price, LTV, term."),
    ToolCall(name="lookup_tax_rate", description="Look up ITP / IBI / IRNR rate for a given autonomous community."),
    ToolCall(name="region_info", description="Return market summary and average price/m² for a region."),
]


# ---------------------------------------------------------------------------
# Task: "Find me a flat in Madrid under 400k and tell me my monthly mortgage
# at 70% LTV over 25 years." — a good agent calls 2 tools in sequence.
# ---------------------------------------------------------------------------


GOOD_CALLS = [
    ToolCall(
        name="search_listings",
        input_parameters={"region": "Madrid", "max_price_eur": 400_000, "type": "piso"},
        output=[
            {"id": "MAD-1", "price": 385_000, "area": "Chamberí"},
            {"id": "MAD-2", "price": 390_000, "area": "Salamanca"},
        ],
    ),
    ToolCall(
        name="mortgage_calc",
        input_parameters={"price_eur": 385_000, "ltv_percent": 70, "term_years": 25, "rate_percent": 3.4},
        output={"monthly_eur": 1336, "total_interest_eur": 131_700},
    ),
]

BAD_CALLS = [
    ToolCall(
        name="region_info",   # wrong tool — answers the wrong question
        input_parameters={"region": "Madrid"},
        output={"avg_price_per_m2": 4200, "trend": "stable"},
    ),
]


GOOD_ANSWER = (
    "Two flats in Madrid under €400k: a 385k flat in Chamberí and a 390k flat "
    "in Salamanca. On the cheaper one (385k) at 70% LTV over 25 years, your "
    "monthly mortgage at a 3.4% fixed rate is about €1,336."
)

BAD_ANSWER = (
    "Madrid market: average price per m² is €4,200; trend is stable. "
    "I don't have a mortgage figure for you."
)


# ---------------------------------------------------------------------------
# TaskCompletion — judge decides "did the agent answer the task?"
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.agentic
def test_task_completion_good_path(judge):
    tc = LLMTestCase(
        input=(
            "Find me a flat in Madrid under €400,000 and tell me my monthly "
            "mortgage at 70% LTV over 25 years."
        ),
        actual_output=GOOD_ANSWER,
        tools_called=GOOD_CALLS,
    )
    m = TaskCompletionMetric(threshold=0.5, model=judge, include_reason=True)
    m.measure(tc)
    assert m.is_successful(), f"TaskCompletion {m.score:.2f}: {m.reason}"


@pytest.mark.smoke
@pytest.mark.agentic
def test_task_completion_bad_path_is_flagged(judge):
    """Negative case: the wrong tool is called, the answer ducks the mortgage
    part. TaskCompletion should flag this — assert FAILURE on purpose."""
    tc = LLMTestCase(
        input=(
            "Find me a flat in Madrid under €400,000 and tell me my monthly "
            "mortgage at 70% LTV over 25 years."
        ),
        actual_output=BAD_ANSWER,
        tools_called=BAD_CALLS,
    )
    m = TaskCompletionMetric(threshold=0.5, model=judge, include_reason=True)
    m.measure(tc)
    assert not m.is_successful(), (
        f"TaskCompletion was meant to fail on this case but scored "
        f"{m.score:.2f}: {m.reason}"
    )


# ---------------------------------------------------------------------------
# Tool correctness — the right tools were called
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.agentic
def test_tool_correctness_matches_expected(judge):
    tc = LLMTestCase(
        input="search for Madrid flats under 400k and compute the mortgage",
        actual_output=GOOD_ANSWER,
        tools_called=GOOD_CALLS,
        expected_tools=[
            ToolCall(name="search_listings"),
            ToolCall(name="mortgage_calc"),
        ],
    )
    # ToolCorrectness is a deterministic check — it does NOT call an LLM,
    # so passing `model=judge` is optional. We pass it anyway for symmetry
    # with the metric matrix runner in Phase 7.
    m = ToolCorrectnessMetric(
        available_tools=AVAILABLE_TOOLS,
        threshold=1.0,
        should_exact_match=False,
        should_consider_ordering=False,
    )
    m.measure(tc)
    assert m.is_successful(), f"ToolCorrectness {m.score:.2f}: {m.reason}"


# ---------------------------------------------------------------------------
# Argument correctness — were the tool arguments well-formed?
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.agentic
def test_argument_correctness_well_formed(judge):
    tc = LLMTestCase(
        input=(
            "Find me a flat in Madrid under €400,000 and tell me my monthly "
            "mortgage at 70% LTV over 25 years."
        ),
        actual_output=GOOD_ANSWER,
        tools_called=GOOD_CALLS,
    )
    m = ArgumentCorrectnessMetric(threshold=0.5, model=judge, include_reason=True)
    m.measure(tc)
    assert m.is_successful(), f"ArgumentCorrectness {m.score:.2f}: {m.reason}"
