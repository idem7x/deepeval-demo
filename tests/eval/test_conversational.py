"""
Multi-turn conversational metric suite.

Pattern: roll a real conversation forward by calling our chat service for
each user turn (with the running history), then assemble a
ConversationalTestCase from the resulting Turn list and grade.

Metrics covered:
- KnowledgeRetentionMetric — does the assistant remember earlier facts?
- ConversationCompletenessMetric — did the assistant address every goal?
- RoleAdherenceMetric — did the assistant stay in the declared role?

The dialog mirrors a real buyer flow: research a region, ask about
specific taxes, then follow up on visa implications.
"""

from __future__ import annotations

import asyncio

import pytest
from deepeval.metrics import (
    ConversationCompletenessMetric,
    KnowledgeRetentionMetric,
    RoleAdherenceMetric,
)
from deepeval.test_case import ConversationalTestCase, Turn

from apps.backend.chat.service import answer
from apps.backend.llm.base import ChatMessage


CHATBOT_ROLE = (
    "You are a knowledgeable assistant for foreign buyers of Spanish real "
    "estate. You explain taxes, legal procedures, and visa rules. You never "
    "invent figures or article numbers."
)


def _run_dialog(adapter, sut_model, user_messages: list[str]) -> list[Turn]:
    """Roll the dialog one turn at a time, threading history through `answer`."""
    history: list[ChatMessage] = []
    turns: list[Turn] = []
    for user_msg in user_messages:
        result = asyncio.run(
            answer(
                adapter,
                user_msg,
                model=sut_model,
                use_rag=True,
                rag_k=4,
                history=history,
                temperature=0.0,
                max_tokens=350,
            )
        )
        turns.append(Turn(role="user", content=user_msg))
        turns.append(Turn(
            role="assistant",
            content=result.text,
            retrieval_context=result.retrieval_context,
        ))
        history.append(ChatMessage(role="user", content=user_msg))
        history.append(ChatMessage(role="assistant", content=result.text))
    return turns


@pytest.fixture(scope="module")
def buyer_dialog(chat_adapter, sut_model):
    """A 4-turn buyer's flow about a Madrid resale apartment."""
    return _run_dialog(
        chat_adapter,
        sut_model,
        user_messages=[
            "I'm thinking about buying a resale apartment in Madrid. What's the main tax I'll pay on the purchase?",
            "And what's the rate in Madrid specifically?",
            "Got it. As a UK citizen, can I get residency by making this purchase?",
            "If I want to rent it out short-term, do I need a special licence?",
        ],
    )


def _ctc(turns: list[Turn], scenario: str, expected_outcome: str) -> ConversationalTestCase:
    return ConversationalTestCase(
        turns=turns,
        scenario=scenario,
        chatbot_role=CHATBOT_ROLE,
        expected_outcome=expected_outcome,
    )


# ---------------------------------------------------------------------------
# Knowledge retention: did the assistant remember earlier facts?
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.conversational
def test_knowledge_retention_smoke(buyer_dialog, judge):
    tc = _ctc(
        buyer_dialog,
        scenario="UK buyer asking sequential questions about a Madrid resale.",
        expected_outcome="Assistant keeps track of 'Madrid' and 'UK citizen' across turns.",
    )
    # Lower-is-better metric (counts attritions of remembered facts).
    m = KnowledgeRetentionMetric(threshold=0.5, model=judge, include_reason=True)
    m.measure(tc)
    assert m.is_successful(), f"KnowledgeRetention {m.score:.2f}: {m.reason}"


# ---------------------------------------------------------------------------
# Conversation completeness: did the assistant address every goal?
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.conversational
def test_conversation_completeness_smoke(buyer_dialog, judge):
    tc = _ctc(
        buyer_dialog,
        scenario="UK buyer asking sequential questions about a Madrid resale.",
        expected_outcome=(
            "Assistant has covered: purchase tax kind + Madrid rate, UK-citizen "
            "residency outcome, and short-term-rental licensing implication."
        ),
    )
    m = ConversationCompletenessMetric(threshold=0.5, model=judge, include_reason=True)
    m.measure(tc)
    assert m.is_successful(), f"ConversationCompleteness {m.score:.2f}: {m.reason}"


# ---------------------------------------------------------------------------
# Role adherence: did the assistant stay in the declared role?
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.conversational
def test_role_adherence_smoke(buyer_dialog, judge):
    tc = _ctc(
        buyer_dialog,
        scenario="UK buyer asking sequential questions about a Madrid resale.",
        expected_outcome="Assistant stays in the Spanish real estate advisor role throughout.",
    )
    m = RoleAdherenceMetric(threshold=0.5, model=judge, include_reason=True)
    m.measure(tc)
    assert m.is_successful(), f"RoleAdherence {m.score:.2f}: {m.reason}"
