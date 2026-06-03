"""
Conversational NEGATIVE controls — hand-craft broken multi-turn dialogs
and assert the conversational metrics flag them.

No SUT: each Turn is written by hand so the failure is unambiguous and the
run is cheap (judge-only). Mirror of conversational/test_positive.py.

Direction:
- KnowledgeRetention → LOWER is better (counts forgotten facts); a dialog
  where the assistant forgets facts scores HIGH → unsuccessful.
- ConversationCompleteness / RoleAdherence → HIGHER is better; a dialog
  that ducks goals / breaks role scores LOW → unsuccessful.

Either way: `not m.is_successful()`.
"""

from __future__ import annotations

import pytest
from deepeval.metrics import (
    ConversationCompletenessMetric,
    KnowledgeRetentionMetric,
    RoleAdherenceMetric,
)
from deepeval.test_case import ConversationalTestCase, Turn


CHATBOT_ROLE = (
    "You are a knowledgeable assistant for foreign buyers of Spanish real "
    "estate. You explain taxes, legal procedures, and visa rules. You never "
    "invent figures or article numbers."
)


def _ctc(turns: list[Turn], scenario: str, expected_outcome: str) -> ConversationalTestCase:
    return ConversationalTestCase(
        turns=turns,
        scenario=scenario,
        chatbot_role=CHATBOT_ROLE,
        expected_outcome=expected_outcome,
    )


# ---------------------------------------------------------------------------
# Knowledge retention — assistant forgets a fact the user already gave
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.conversational
def test_knowledge_retention_catches_forgotten_fact(judge):
    # The user says "Madrid" and "UK citizen" up front; the assistant later
    # asks for both again and even guesses a different city — clear attrition.
    turns = [
        Turn(role="user", content="I'm a UK citizen buying a resale flat in Madrid."),
        Turn(role="assistant", content="Great, happy to help with your purchase."),
        Turn(role="user", content="What's the main purchase tax I'll pay?"),
        Turn(role="assistant", content="That's ITP. By the way, which city are you buying in?"),
        Turn(role="user", content="I told you — Madrid."),
        Turn(role="assistant", content="Understood. And what nationality are you, so I can check visa rules?"),
    ]
    tc = _ctc(
        turns,
        scenario="Buyer gives city and nationality, assistant keeps forgetting them.",
        expected_outcome="Assistant should retain 'Madrid' and 'UK citizen' across turns.",
    )
    m = KnowledgeRetentionMetric(threshold=0.5, model=judge, include_reason=True)
    m.measure(tc)
    assert not m.is_successful(), (
        f"KnowledgeRetention failed to flag forgotten facts "
        f"(score={m.score:.2f}): {m.reason}"
    )


# ---------------------------------------------------------------------------
# Conversation completeness — assistant never addresses the user's goals
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.conversational
def test_conversation_completeness_catches_unmet_goals(judge):
    # User asks three concrete things; assistant deflects every one.
    turns = [
        Turn(role="user", content="What purchase tax applies to a Madrid resale, and at what rate?"),
        Turn(role="assistant", content="Taxes are complicated, you should really talk to a lawyer."),
        Turn(role="user", content="As a UK citizen, does buying give me residency?"),
        Turn(role="assistant", content="I'm not sure, immigration is tricky."),
        Turn(role="user", content="Do I need a licence to rent it short-term?"),
        Turn(role="assistant", content="Maybe, it depends. Anyway, good luck with everything!"),
    ]
    tc = _ctc(
        turns,
        scenario="UK buyer asks three concrete questions about a Madrid resale.",
        expected_outcome=(
            "Assistant should cover: purchase tax kind + Madrid rate, UK-citizen "
            "residency outcome, and short-term-rental licensing."
        ),
    )
    m = ConversationCompletenessMetric(threshold=0.5, model=judge, include_reason=True)
    m.measure(tc)
    assert not m.is_successful(), (
        f"ConversationCompleteness failed to flag unmet goals "
        f"(score={m.score:.2f} >= threshold): {m.reason}"
    )


# ---------------------------------------------------------------------------
# Role adherence — assistant abandons its declared role
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.conversational
def test_role_adherence_catches_out_of_role(judge):
    # Role is a Spanish real-estate advisor; the assistant veers into
    # medical advice and stock tips, clearly out of role.
    turns = [
        Turn(role="user", content="What tax do I pay buying a flat in Madrid?"),
        Turn(role="assistant", content="Honestly, forget property. Let me give you medical advice: take vitamin D daily."),
        Turn(role="user", content="That's not what I asked. What about the purchase tax?"),
        Turn(role="assistant", content="You should buy Tesla stock instead, it's going to the moon. Real estate is boring."),
    ]
    tc = _ctc(
        turns,
        scenario="Buyer asks about Spanish property tax; assistant gives medical and stock advice.",
        expected_outcome="Assistant stays in the Spanish real-estate advisor role throughout.",
    )
    m = RoleAdherenceMetric(threshold=0.5, model=judge, include_reason=True)
    m.measure(tc)
    assert not m.is_successful(), (
        f"RoleAdherence failed to flag out-of-role behaviour "
        f"(score={m.score:.2f} >= threshold): {m.reason}"
    )
