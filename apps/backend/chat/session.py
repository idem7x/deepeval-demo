"""
Process-local chat session store.

Keeps the conversation history (a list of ChatMessage) per `session_id`,
plus a `last_seen` timestamp for TTL eviction. The interface is small on
purpose — Phase 4 ships with the in-memory implementation; a Redis-backed
one can swap in later behind the same surface.

We rely on this for two things:
1. The /chat endpoint appends and reads from it on every turn.
2. DeepEval's multi-turn metrics (KnowledgeRetention, ConversationCompleteness)
   in Phase 6 need the full transcript — we read it from here as well.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from apps.backend.llm.base import ChatMessage


@dataclass(slots=True)
class Session:
    id: str
    messages: list[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    # Per-session sticky settings — what the UI selected for this thread.
    provider: str = ""
    model: str = ""

    def touch(self) -> None:
        self.last_seen = time.time()


class SessionStore:
    """In-memory dict with TTL eviction. Process-local — fine for one
    backend instance; switch to Redis when we want horizontal scaling."""

    def __init__(self, ttl_seconds: float = 60 * 60 * 2) -> None:  # 2h
        self._sessions: dict[str, Session] = {}
        self._ttl = ttl_seconds

    def get_or_create(self, session_id: str | None = None) -> Session:
        self._evict_expired()
        if session_id and session_id in self._sessions:
            s = self._sessions[session_id]
            s.touch()
            return s
        new_id = session_id or uuid.uuid4().hex
        s = Session(id=new_id)
        self._sessions[new_id] = s
        return s

    def append(self, session_id: str, message: ChatMessage) -> None:
        s = self.get_or_create(session_id)
        s.messages.append(message)
        s.touch()

    def get(self, session_id: str) -> Session | None:
        self._evict_expired()
        s = self._sessions.get(session_id)
        if s:
            s.touch()
        return s

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _evict_expired(self) -> None:
        cutoff = time.time() - self._ttl
        expired = [k for k, s in self._sessions.items() if s.last_seen < cutoff]
        for k in expired:
            del self._sessions[k]

    # Diagnostics
    def stats(self) -> dict[str, int]:
        return {"sessions": len(self._sessions)}


# Single process-wide store. /chat routes import this directly; tests can
# clear it between cases via store.clear() or instantiate a new one.
store = SessionStore()
