"""
Custom metrics — `GEval`, `DAGMetric`, `ArenaGEval`.

These are DeepEval's escape hatches: when no built-in metric expresses
what you want to grade, you write your own criterion in natural language
(GEval), wire up a decision tree (DAG), or compare candidates pairwise
(ArenaGEval).

The cases here are all domain-specific — they grade things the built-in
metrics can't: "does the answer carry a tax-figure disclaimer", "is the
answer structured as a numbered procedure", "of two answers, which is
more concrete about article numbers".
"""

from __future__ import annotations

import asyncio

import pytest
from deepeval.metrics import ArenaGEval, DAGMetric, GEval
from deepeval.metrics.dag import (
    BinaryJudgementNode,
    DeepAcyclicGraph,
    TaskNode,
    VerdictNode,
)
from deepeval.test_case import (
    ArenaTestCase,
    LLMTestCase,
    LLMTestCaseParams,
)
from deepeval.test_case.arena_test_case import Contestant

from apps.backend.chat.service import answer
from apps.backend.llm.base import ChatMessage, ChatOptions


# ---------------------------------------------------------------------------
# GEval — domain-specific criterion in natural language
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.custom
def test_geval_includes_disclaimer_on_tax_advice(chat_adapter, sut_model, judge):
    """When asked for tax figures, the assistant should remind that rates
    can change and the user should verify with current sources."""
    question = "What's the ITP rate I'll pay buying a flat in Madrid?"
    result = asyncio.run(
        answer(
            chat_adapter,
            question,
            model=sut_model,
            use_rag=True,
            temperature=0.0,
            max_tokens=300,
            system_prompt=(
                "You are a Spanish real-estate assistant. When you quote "
                "any tax rate or article number, always add a short "
                "sentence reminding the user to verify the current rate "
                "with the autonomous community before relying on it."
            ),
        )
    )
    tc = LLMTestCase(input=question, actual_output=result.text)

    m = GEval(
        name="TaxDisclaimerPresent",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        criteria=(
            "Does the answer include a short caveat that tax rates can "
            "change and should be verified with current official sources? "
            "It should be present whenever any tax figure is quoted."
        ),
        model=judge,
        threshold=0.5,
    )
    m.measure(tc)
    assert m.is_successful(), f"GEval {m.score:.2f}: {m.reason}"


# ---------------------------------------------------------------------------
# DAGMetric — decision tree of binary checks
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.custom
def test_dag_procedural_check(chat_adapter, sut_model, judge):
    """One TaskNode → one BinaryJudgementNode → terminal verdicts.

    A DAG forces the judge to evaluate one question at a time, which is
    both cheaper (smaller prompts) and easier to debug than one giant
    rubric. The deeper 3-level variant lives in test_dag_layered_full
    (@local) — VerdictNode terminals must not be shared across paths,
    so deeper graphs need a separate leaf per branch.
    """
    question = "Walk me through the steps to buy a resale flat in Spain."
    result = asyncio.run(
        answer(
            chat_adapter, question, model=sut_model, use_rag=True,
            temperature=0.0, max_tokens=500,
        )
    )
    tc = LLMTestCase(input=question, actual_output=result.text)

    procedural = BinaryJudgementNode(
        criteria=(
            "Is the answer structured as an ordered or numbered sequence "
            "of distinct steps, rather than a single paragraph of prose?"
        ),
        children=[
            VerdictNode(verdict=True, score=10),
            VerdictNode(verdict=False, score=0),
        ],
    )
    root = TaskNode(
        instructions=(
            "Read the assistant answer below; downstream nodes will judge "
            "its structure."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        output_label="answer",
        children=[procedural],
    )
    dag = DeepAcyclicGraph(root_nodes=[root])

    m = DAGMetric(name="ProceduralAnswer", dag=dag, model=judge, threshold=0.5)
    m.measure(tc)
    assert m.is_successful(), f"DAG {m.score:.2f}: {m.reason}"


@pytest.mark.local
@pytest.mark.custom
def test_dag_layered_full(chat_adapter, sut_model, judge):
    """Three-level DAG: procedural → arras → NIE.

    Each branch has its OWN fail leaf — shared VerdictNodes silently
    corrupt parent tracking and the metric returns score=None.
    """
    question = "Walk me through the steps to buy a resale flat in Spain."
    result = asyncio.run(
        answer(
            chat_adapter, question, model=sut_model, use_rag=True,
            temperature=0.0, max_tokens=500,
        )
    )
    tc = LLMTestCase(input=question, actual_output=result.text)

    def fail() -> VerdictNode:
        # One fresh leaf per usage — never share.
        return VerdictNode(verdict=False, score=0)

    mentions_nie = BinaryJudgementNode(
        criteria="Does the answer reference obtaining a NIE?",
        children=[VerdictNode(verdict=True, score=10), VerdictNode(verdict=False, score=0)],
    )
    mentions_arras = BinaryJudgementNode(
        criteria="Does the answer mention the deposit contract (arras / contrato de arras)?",
        children=[
            VerdictNode(verdict=True, child=mentions_nie),
            VerdictNode(verdict=False, child=fail()),
        ],
    )
    looks_procedural = BinaryJudgementNode(
        criteria="Is the answer an ordered, numbered sequence of distinct steps?",
        children=[
            VerdictNode(verdict=True, child=mentions_arras),
            VerdictNode(verdict=False, child=fail()),
        ],
    )
    root = TaskNode(
        instructions="Read the assistant answer; downstream nodes judge it.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        output_label="answer",
        children=[looks_procedural],
    )
    dag = DeepAcyclicGraph(root_nodes=[root])

    m = DAGMetric(name="BuyingProcessLayered", dag=dag, model=judge, threshold=0.5)
    m.measure(tc)
    assert m.is_successful(), f"DAG {m.score:.2f}: {m.reason}"


# ---------------------------------------------------------------------------
# ArenaGEval — pairwise comparison of two answers
# ---------------------------------------------------------------------------


# NOTE: ArenaGEval is marked @local rather than @smoke.
# DeepEval 4.0.3 remaps real contestant names to internal "fake names"
# (Iris, Henry, …) before sending the comparison prompt, and the judge
# occasionally hallucinates a name that isn't in the dummy-to-real map —
# which then raises KeyError on lookup. The metric works most of the time
# but the failure mode is irreducibly flaky, so we don't gate CI on it.
@pytest.mark.local
@pytest.mark.custom
def test_arena_geval_concrete_vs_vague(chat_adapter, sut_model, judge):
    """Two answers to the same question — one terse, one detailed.

    ArenaGEval picks the better answer per a stated criterion. We use
    different prompts to deliberately produce a concrete vs. vague
    answer; ArenaGEval should prefer the concrete one.
    """
    question = "What rate of ITP applies in Andalucía and Madrid?"

    async def one(system_extra: str) -> str:
        resp = await chat_adapter.chat(
            [
                ChatMessage(role="system", content=system_extra),
                ChatMessage(role="user", content=question),
            ],
            ChatOptions(model=sut_model, temperature=0.0, max_tokens=200),
        )
        return resp.text

    concrete, vague = asyncio.run(
        _gather_two(
            one("You are precise. Always quote exact figures and the legal source."),
            one("You are vague. Avoid giving specific figures; speak in generalities."),
        )
    )

    tc = ArenaTestCase(
        contestants=[
            # NOTE: short stable names — the metric prompt asks the judge to
            # echo the name verbatim, and longer/expressive names occasionally
            # cause the judge to hallucinate a related word as the winner.
            Contestant(name="A", test_case=LLMTestCase(input=question, actual_output=concrete)),
            Contestant(name="B", test_case=LLMTestCase(input=question, actual_output=vague)),
        ],
    )
    m = ArenaGEval(
        name="MoreConcreteWins",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        criteria=(
            "Pick the contestant that provides more specific, verifiable "
            "tax figures and clearer citations of the autonomous communities."
        ),
        model=judge,
    )
    m.measure(tc)
    # ArenaGEval doesn't have is_successful (it picks a winner, not a score).
    # We assert the concrete one won.
    print(f"\nArenaGEval winner = {m.winner!r}  reason: {m.reason}")
    assert m.winner == "A", (
        f"expected concrete answer (A) to win, got {m.winner!r}: {m.reason}"
    )


async def _gather_two(a, b):
    import asyncio as _a
    return await _a.gather(a, b)
