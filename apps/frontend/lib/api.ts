// Typed client for the FastAPI backend.
//
// All endpoints go through `request()` so we have one place for headers,
// error handling, and base URL resolution. Streaming /chat is special-cased
// because it returns an SSE stream rather than JSON.

import { parseSSE } from "./sse";
import type {
  ArenaRequest,
  ArenaResponse,
  ChatRequest,
  ChatResponse,
  ModelsResponse,
  SessionDetail,
  StreamEvent,
} from "./types";

const BASE_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const r = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...(init.headers || {}) },
    ...init,
  });
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try {
      const body = await r.json();
      if (body?.detail) detail = `${detail} — ${body.detail}`;
    } catch {
      /* no JSON body */
    }
    throw new ApiError(r.status, detail);
  }
  return r.json() as Promise<T>;
}

// --- public surface --------------------------------------------------------

export const api = {
  async health(): Promise<unknown> {
    return request("/health");
  },

  async models(refresh = false): Promise<ModelsResponse> {
    return request<ModelsResponse>(`/models${refresh ? "?refresh=true" : ""}`);
  },

  async chat(req: ChatRequest): Promise<ChatResponse> {
    return request<ChatResponse>("/chat", {
      method: "POST",
      body: JSON.stringify({ ...req, stream: false }),
    });
  },

  /**
   * Streaming /chat. Pass an `onEvent` callback that receives each parsed
   * SSE event as it arrives. Returns the final session_id after `done`.
   */
  async chatStream(
    req: ChatRequest,
    onEvent: (e: StreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<string | null> {
    const r = await fetch(`${BASE_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...req, stream: true }),
      signal,
    });
    if (!r.ok) {
      throw new ApiError(r.status, `${r.status} ${r.statusText}`);
    }
    let sessionId: string | null = null;
    for await (const evt of parseSSE(r)) {
      if (evt.type === "session") sessionId = evt.data.session_id;
      onEvent(evt);
      if (evt.type === "done" || evt.type === "error") break;
    }
    return sessionId;
  },

  async getSession(sessionId: string): Promise<SessionDetail> {
    return request<SessionDetail>(`/chat/sessions/${sessionId}`);
  },

  async deleteSession(sessionId: string): Promise<void> {
    await fetch(`${BASE_URL}/chat/sessions/${sessionId}`, { method: "DELETE" });
  },

  async arena(req: ArenaRequest): Promise<ArenaResponse> {
    return request<ArenaResponse>("/arena/compare", {
      method: "POST",
      body: JSON.stringify(req),
    });
  },
};

export { ApiError };
