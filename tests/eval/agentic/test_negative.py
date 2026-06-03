"""
Agentic NEGATIVE controls — hand-craft failing tool-use traces and assert
each agentic metric flags them.

Mirror of agentic/test_positive.py. We build `tools_called` / `expected_tools`
by hand (no live agent loop), so the run is deterministic and cheap.

All three metrics are HIGHER-is-better, so a broken trace must come out
UNSUCCESSFUL → `not m.is_successful()`.
"""

from __future__ import annotations

import pytest
from deepeval.metrics import (
    ArgumentCorrectnessMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from deepeval.test_case import LLMTestCase, ToolCall


AVAILABLE_TOOLS = [
    ToolCall(name="search_listings", description="Search property listings by region and price band."),
    ToolCall(name="mortgage_calc", description="Compute monthly mortgage payment from price, LTV, term."),
    ToolCall(name="lookup_tax_rate", description="Look up ITP / IBI / IRNR rate for a given autonomous community."),
    ToolCall(name="region_info", description="Return market summary and average price/m² for a region."),
]


TASK = (
    "Find me a flat in Madrid under €400,000 and tell me my monthly "
    "mortgage at 70% LTV over 25 years."
)


# A trace that calls the WRONG tool and ducks half the task.
WRONG_CALLS = [
    ToolCall(
        name="region_info",
        input_parameters={"region": "Madrid"},
        output={"avg_price_per_m2": 4200, "trend": "stable"},
    ),
]

DUCKING_ANSWER = (
    "Madrid market: average price per m² is €4,200; trend is stable. "
    "I don't have a mortgage figure for you."
)


# ---------------------------------------------------------------------------
# TaskCompletion — wrong tool + half-answered task must be flagged
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.agentic
def test_task_completion_catches_incomplete(judge):
    tc = LLMTestCase(input=TASK, actual_output=DUCKING_ANSWER, tools_called=WRONG_CALLS)
    m = TaskCompletionMetric(threshold=0.5, model=judge, include_reason=True)
    m.measure(tc)
    assert not m.is_successful(), (
        f"TaskCompletion failed to flag an incomplete task "
        f"(score={m.score:.2f} >= threshold): {m.reason}"
    )


# ---------------------------------------------------------------------------
# ToolCorrectness — wrong tools vs expected must be flagged (deterministic)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.agentic
def test_tool_correctness_catches_wrong_tools(judge):
    tc = LLMTestCase(
        input=TASK,
        actual_output=DUCKING_ANSWER,
        tools_called=[ToolCall(name="region_info")],
        expected_tools=[
            ToolCall(name="search_listings"),
            ToolCall(name="mortgage_calc"),
        ],
    )
    # Deterministic check — no LLM. None of the expected tools were called,
    # so the matched fraction is 0.0, well below threshold=1.0.
    m = ToolCorrectnessMetric(
        available_tools=AVAILABLE_TOOLS,
        threshold=1.0,
        should_exact_match=False,
        should_consider_ordering=False,
    )
    m.measure(tc)
    assert not m.is_successful(), (
        f"ToolCorrectness failed to flag wrong tool selection "
        f"(score={m.score:.2f} >= threshold): {m.reason}"
    )


# ---------------------------------------------------------------------------
# ArgumentCorrectness — malformed / nonsensical tool arguments must be flagged
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.agentic
def test_argument_correctness_catches_bad_args(judge):
    # Right tools, but arguments contradict the task: wrong region, a price
    # cap 25x over budget, an absurd 900% LTV and a negative term.
    bad_calls = [
        ToolCall(
            name="search_listings",
            input_parameters={"region": "Barcelona", "max_price_eur": 10_000_000, "type": "castle"},
            output=[{"id": "BCN-1", "price": 9_500_000, "area": "Eixample"}],
        ),
        ToolCall(
            name="mortgage_calc",
            input_parameters={"price_eur": 9_500_000, "ltv_percent": 900, "term_years": -25, "rate_percent": 3.4},
            output={"monthly_eur": -42, "total_interest_eur": 0},
        ),
    ]
    tc = LLMTestCase(
        input=TASK,
        actual_output="Here is a €9.5M castle in Barcelona at 900% LTV over -25 years.",
        tools_called=bad_calls,
    )
    m = ArgumentCorrectnessMetric(threshold=0.5, model=judge, include_reason=True)
    m.measure(tc)
    assert not m.is_successful(), (
        f"ArgumentCorrectness failed to flag malformed arguments "
        f"(score={m.score:.2f} >= threshold): {m.reason}"
    )
