"""
TestClient smoke tests for every Phase 4 endpoint.

No real LLM calls — FakeAdapter and FakeRetriever are injected via
FastAPI's dependency_overrides. These tests verify the wiring (shape,
status codes, RAG-context plumbing, session persistence, SSE event
ordering), not provider behaviour.
"""

from __future__ import annotations

import pytest


# --- /health ----------------------------------------------------------------


@pytest.mark.smoke
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "providers" in body and "embeddings" in body


# --- /models ----------------------------------------------------------------


@pytest.mark.smoke
def test_models_lists_injected_providers(client):
    r = client.get("/models")
    assert r.status_code == 200
    names = {p["provider"] for p in r.json()["providers"]}
    assert names == {"openai", "anthropic"}
    for p in r.json()["providers"]:
        assert p["available"] is True
        assert p["default_model"]
        assert p["models"]


# --- /chat (non-streaming) --------------------------------------------------


@pytest.mark.smoke
def test_chat_non_stream_returns_full_response(client):
    r = client.post(
        "/chat",
        json={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "message": "What is ITP rate in Madrid?",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "GPT answers ITP is 6%."
    assert body["provider"] == "openai"
    assert body["model"] == "gpt-4o-mini"
    assert body["session_id"]
    assert body["prompt_tokens"] == 3 and body["completion_tokens"] == 4
    # RAG default is on → must include the chunks our FakeRetriever returns.
    assert len(body["retrieved_context"]) == 2
    titles = [c["title"] for c in body["retrieved_context"]]
    assert "ITP — Transfer tax" in titles


@pytest.mark.smoke
def test_chat_persists_messages_to_session(client):
    r1 = client.post("/chat", json={
        "provider": "openai", "model": "gpt-4o-mini",
        "message": "first", "use_rag": False,
    })
    sid = r1.json()["session_id"]
    r2 = client.post("/chat", json={
        "provider": "openai", "model": "gpt-4o-mini",
        "message": "second", "use_rag": False,
        "session_id": sid,
    })
    assert r2.json()["session_id"] == sid

    s = client.get(f"/chat/sessions/{sid}").json()
    roles = [m["role"] for m in s["messages"]]
    contents = [m["content"] for m in s["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert contents[0] == "first" and contents[2] == "second"


@pytest.mark.smoke
def test_chat_rag_disabled_means_no_retrieval(client):
    r = client.post("/chat", json={
        "provider": "openai", "model": "gpt-4o-mini",
        "message": "anything", "use_rag": False,
    })
    assert r.json()["retrieved_context"] == []
    assert client.fake_retriever.last_query is None


@pytest.mark.smoke
def test_chat_passes_rag_filters(client):
    client.post("/chat", json={
        "provider": "openai", "model": "gpt-4o-mini",
        "message": "Spanish taxes",
        "rag_k": 3,
        "rag_filters": {"topic": "tax"},
    })
    assert client.fake_retriever.last_kwargs == {"k": 3, "topic": "tax"}


@pytest.mark.smoke
def test_chat_rejects_unknown_provider(client):
    r = client.post("/chat", json={
        "provider": "mistral", "model": "x",
        "message": "hi", "use_rag": False,
    })
    assert r.status_code == 400


@pytest.mark.smoke
def test_chat_delete_session(client):
    r1 = client.post("/chat", json={
        "provider": "openai", "model": "gpt-4o-mini",
        "message": "first", "use_rag": False,
    })
    sid = r1.json()["session_id"]
    assert client.delete(f"/chat/sessions/{sid}").status_code == 204
    assert client.get(f"/chat/sessions/{sid}").status_code == 404


# --- /chat (streaming) ------------------------------------------------------


@pytest.mark.smoke
def test_chat_stream_emits_expected_events(client):
    with client.stream("POST", "/chat", json={
        "provider": "openai", "model": "gpt-4o-mini",
        "message": "Hello",
        "stream": True,
    }) as r:
        assert r.status_code == 200
        text = "".join(r.iter_text())

    # Event ordering: session → context → deltas → done
    sess_at = text.find("event: session")
    ctx_at = text.find("event: context")
    first_delta_at = text.find("event: delta")
    done_at = text.find("event: done")
    assert 0 <= sess_at < ctx_at < first_delta_at < done_at
    # All canned words must show up across delta events.
    for word in "GPT answers ITP is 6%.".split():
        assert word in text


# --- /arena -----------------------------------------------------------------


@pytest.mark.smoke
def test_arena_runs_all_candidates_in_parallel(client):
    r = client.post("/arena/compare", json={
        "prompt": "What is ITP in Madrid?",
        "candidates": [
            {"provider": "openai", "model": "gpt-4o-mini"},
            {"provider": "anthropic", "model": "claude-haiku-4-5"},
        ],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prompt"] == "What is ITP in Madrid?"
    assert len(body["retrieved_context"]) == 2
    providers = {res["provider"] for res in body["results"]}
    assert providers == {"openai", "anthropic"}
    for res in body["results"]:
        assert res["text"]
        assert res["error"] is None
        assert res["latency_ms"] >= 0


@pytest.mark.smoke
def test_arena_rejects_unknown_provider(client):
    r = client.post("/arena/compare", json={
        "prompt": "x",
        "candidates": [
            {"provider": "openai", "model": "gpt-4o-mini"},
            {"provider": "ghost", "model": "y"},
        ],
    })
    assert r.status_code == 400


# --- /eval/runs -------------------------------------------------------------


@pytest.mark.smoke
def test_eval_runs_stub(client):
    r = client.get("/eval/runs")
    assert r.status_code == 200
    assert r.json()["runs"] == []
