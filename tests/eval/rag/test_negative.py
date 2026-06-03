"""
RAG NEGATIVE controls — feed each RAG metric a deliberately broken
(input, output, retrieval_context, expected_output) tuple and assert the
metric flags it.

No SUT / no real retrieval here: every field is hand-crafted so the
failure mode is unambiguous and the run is cheap (judge-only). This is the
mirror image of rag/test_positive.py — there we prove good RAG passes;
here we prove the metrics actually catch each distinct RAG failure.

All five RAG metrics are HIGHER-is-better, so a broken case must come out
UNSUCCESSFUL → `not m.is_successful()`.
"""

from __future__ import annotations

import pytest
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase


# ---------------------------------------------------------------------------
# Faithfulness — answer makes claims absent from / contradicting the context
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.rag
def test_faithfulness_catches_unsupported_claims(judge):
    tc = LLMTestCase(
        input="What is the ITP rate for resale homes in Madrid?",
        # Context says 6%; the answer invents 21% and an unrelated benefit.
        actual_output=(
            "The ITP in Madrid is 21% and every foreign buyer automatically "
            "receives Spanish citizenship after the purchase."
        ),
        retrieval_context=[
            "In the Community of Madrid the ITP (transfer tax) on resale "
            "homes is 6% of the declared purchase price.",
        ],
    )
    m = FaithfulnessMetric(threshold=0.6, model=judge, include_reason=True)
    m.measure(tc)
    assert not m.is_successful(), (
        f"Faithfulness failed to flag unsupported claims "
        f"(score={m.score:.2f} >= threshold): {m.reason}"
    )


# ---------------------------------------------------------------------------
# Answer Relevancy — answer is fluent but does not address the question
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.rag
def test_answer_relevancy_catches_off_topic(judge):
    tc = LLMTestCase(
        input="What is the ITP rate for resale homes in Madrid?",
        # Coherent, but about the weather — not the question.
        actual_output=(
            "Madrid enjoys a warm continental climate with hot summers and "
            "mild winters, and the city is famous for its museums and tapas."
        ),
        retrieval_context=[
            "In the Community of Madrid the ITP on resale homes is 6%.",
        ],
    )
    m = AnswerRelevancyMetric(threshold=0.5, model=judge, include_reason=True)
    m.measure(tc)
    assert not m.is_successful(), (
        f"AnswerRelevancy failed to flag an off-topic answer "
        f"(score={m.score:.2f} >= threshold): {m.reason}"
    )


# ---------------------------------------------------------------------------
# Contextual Relevancy — retrieved chunks are irrelevant to the question
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.rag
def test_contextual_relevancy_catches_irrelevant_chunks(judge):
    tc = LLMTestCase(
        input="What is the ITP rate for resale homes in Madrid?",
        actual_output="The ITP in Madrid is 6%.",
        # None of these chunks are about ITP rates.
        retrieval_context=[
            "Spanish tortilla is made with eggs, potatoes and onion.",
            "The AVE high-speed train connects Madrid and Seville in 2.5 hours.",
            "La Liga is the top division of Spanish football.",
        ],
    )
    m = ContextualRelevancyMetric(threshold=0.3, model=judge, include_reason=True)
    m.measure(tc)
    assert not m.is_successful(), (
        f"ContextualRelevancy failed to flag irrelevant context "
        f"(score={m.score:.2f} >= threshold): {m.reason}"
    )


# ---------------------------------------------------------------------------
# Contextual Recall — context lacks the info needed for the expected answer
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.rag
def test_contextual_recall_catches_missing_info(judge):
    tc = LLMTestCase(
        input="What is the ITP rate for resale homes in Madrid?",
        actual_output="The ITP in Madrid is 6%.",
        expected_output="The ITP on resale homes in the Community of Madrid is 6%.",
        # The retrieved context is entirely off-topic — NONE of the specifics
        # in the expected answer (ITP, Madrid, 6%) can be attributed to it, so
        # recall must be near zero.
        retrieval_context=[
            "Spanish tortilla is made with eggs, potatoes and onion.",
            "The AVE high-speed train connects Madrid and Seville in 2.5 hours.",
            "La Liga is the top division of Spanish football.",
        ],
    )
    m = ContextualRecallMetric(threshold=0.7, model=judge, include_reason=True)
    m.measure(tc)
    assert not m.is_successful(), (
        f"ContextualRecall failed to flag missing supporting info "
        f"(score={m.score:.2f} >= threshold): {m.reason}"
    )


# ---------------------------------------------------------------------------
# Contextual Precision — the relevant chunk is ranked BELOW noise
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.rag
def test_contextual_precision_catches_bad_ranking(judge):
    tc = LLMTestCase(
        input="What is the ITP rate for resale homes in Madrid?",
        actual_output="The ITP in Madrid is 6%.",
        expected_output="The ITP on resale homes in the Community of Madrid is 6%.",
        # The one relevant chunk is LAST; irrelevant noise is ranked first.
        retrieval_context=[
            "Spanish tortilla is made with eggs, potatoes and onion.",
            "La Liga is the top division of Spanish football.",
            "In the Community of Madrid the ITP on resale homes is 6%.",
        ],
    )
    m = ContextualPrecisionMetric(threshold=0.5, model=judge, include_reason=True)
    m.measure(tc)
    assert not m.is_successful(), (
        f"ContextualPrecision failed to flag bad ranking "
        f"(score={m.score:.2f} >= threshold): {m.reason}"
    )
