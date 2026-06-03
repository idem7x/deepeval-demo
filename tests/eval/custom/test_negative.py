"""
Custom-metric NEGATIVE controls — feed GEval and DAGMetric output that
violates their criterion and assert they flag it.

Mirror of custom/test_positive.py. Output is hand-crafted (no SUT), so the
violation is unambiguous and the run is judge-only/cheap.

Both metrics are HIGHER-is-better → a violating answer must be
UNSUCCESSFUL. (ArenaGEval is intentionally omitted: it's @local-only and
irreducibly flaky in DeepEval 4.0.3 — see custom/test_positive.py.)
"""

from __future__ import annotations

import pytest
from deepeval.metrics import DAGMetric, GEval
from deepeval.metrics.dag import (
    BinaryJudgementNode,
    DeepAcyclicGraph,
    TaskNode,
    VerdictNode,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


# ---------------------------------------------------------------------------
# GEval — answer quotes a tax figure but omits the required disclaimer
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.custom
def test_geval_catches_missing_disclaimer(judge):
    question = "What's the ITP rate I'll pay buying a flat in Madrid?"
    # Quotes a concrete figure with NO "verify the current rate" caveat —
    # exactly what the criterion requires to be present.
    bad_output = "The ITP rate in Madrid is 6%. You'll pay that on the purchase price."
    tc = LLMTestCase(input=question, actual_output=bad_output)

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
    assert not m.is_successful(), (
        f"GEval failed to flag a missing disclaimer "
        f"(score={m.score:.2f} >= threshold): {m.reason}"
    )


# ---------------------------------------------------------------------------
# DAGMetric — answer is a prose blob, not the required numbered procedure
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.custom
def test_dag_catches_non_procedural_answer(judge):
    question = "Walk me through the steps to buy a resale flat in Spain."
    # A single run-on paragraph — the BinaryJudgementNode should route this
    # to the verdict=False leaf (score 0), failing the metric.
    bad_output = (
        "Buying a flat in Spain involves getting your finances together and "
        "finding a place you like and then sorting out the paperwork with a "
        "notary while also dealing with taxes and a deposit at some point, "
        "and eventually you sign and it's all done more or less."
    )
    tc = LLMTestCase(input=question, actual_output=bad_output)

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
    assert not m.is_successful(), (
        f"DAG failed to flag a non-procedural answer "
        f"(score={m.score:.2f} >= threshold): {m.reason}"
    )
