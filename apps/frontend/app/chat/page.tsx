"use client";

import { useEffect, useRef, useState } from "react";
import { ModelPicker } from "@/components/model-picker";
import { SourcesPanel } from "@/components/sources-panel";
import { api } from "@/lib/api";
import type {
  Provider,
  RetrievedChunk,
} from "@/lib/types";

interface Message {
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
}

const RAG_FILTER_TOPICS = [
  { value: "", label: "any topic" },
  { value: "tax", label: "tax" },
  { value: "region", label: "region" },
  { value: "process", label: "process" },
  { value: "visa", label: "visa" },
  { value: "property-type", label: "property type" },
  { value: "law", label: "law" },
  { value: "market", label: "market" },
];

export default function ChatPage() {
  const [picked, setPicked] = useState<{ provider: Provider; model: string } | null>(null);
  const [useRag, setUseRag] = useState(true);
  const [topic, setTopic] = useState<string>("");
  const [region, setRegion] = useState<string>("");
  const [temperature, setTemperature] = useState(0.7);

  const [messages, setMessages] = useState<Message[]>([]);
  const [retrieved, setRetrieved] = useState<RetrievedChunk[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Keep the message list scrolled to the bottom as new content lands.
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  async function send() {
    if (!input.trim() || !picked || busy) return;
    setError(null);
    const userMsg: Message = { role: "user", content: input.trim() };
    const assistantMsg: Message = { role: "assistant", content: "", streaming: true };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setInput("");
    setBusy(true);

    const filters: Record<string, string> = {};
    if (topic) filters.topic = topic;
    if (region) filters.region = region;

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await api.chatStream(
        {
          provider: picked.provider,
          model: picked.model,
          message: userMsg.content,
          session_id: sessionId,
          use_rag: useRag,
          rag_filters: filters,
          temperature,
        },
        (evt) => {
          if (evt.type === "session") setSessionId(evt.data.session_id);
          if (evt.type === "context") setRetrieved(evt.data.chunks);
          if (evt.type === "delta") {
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last?.role === "assistant") {
                next[next.length - 1] = { ...last, content: last.content + evt.data.text };
              }
              return next;
            });
          }
          if (evt.type === "done") {
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last?.role === "assistant") {
                next[next.length - 1] = { ...last, streaming: false };
              }
              return next;
            });
          }
          if (evt.type === "error") {
            setError(evt.data.message);
          }
        },
        controller.signal,
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      // strip the half-built assistant bubble on error so the user can retry
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  async function reset() {
    if (sessionId) {
      try {
        await api.deleteSession(sessionId);
      } catch {
        /* best effort */
      }
    }
    setSessionId(null);
    setMessages([]);
    setRetrieved([]);
    setError(null);
  }

  function stop() {
    abortRef.current?.abort();
  }

  return (
    <div className="mx-auto grid h-[calc(100vh-65px)] max-w-[1400px] grid-cols-[280px_1fr_360px] gap-4 p-4">
      {/* --- left sidebar --- */}
      <aside className="space-y-5 overflow-y-auto rounded-xl border bg-white p-4 shadow-sm">
        <div>
          <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">Model</div>
          <ModelPicker value={picked} onChange={setPicked} />
        </div>

        <div>
          <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">RAG</div>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={useRag}
              onChange={(e) => setUseRag(e.target.checked)}
            />
            Retrieve from knowledge base
          </label>
          {useRag && (
            <div className="mt-2 space-y-2">
              <div>
                <label className="text-xs text-slate-700 mb-0.5 block">Topic filter</label>
                <select
                  className="w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                >
                  {RAG_FILTER_TOPICS.map((t) => (
                    <option key={t.value} value={t.value}>{t.label}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-xs text-slate-700 mb-0.5 block">Region filter</label>
                <input
                  className="w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm"
                  placeholder='e.g. "Madrid" (optional)'
                  value={region}
                  onChange={(e) => setRegion(e.target.value)}
                />
              </div>
            </div>
          )}
        </div>

        <div>
          <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">
            Sampling
          </div>
          <label className="text-xs text-slate-700">
            Temperature: {temperature.toFixed(1)}
          </label>
          <input
            type="range"
            min={0}
            max={2}
            step={0.1}
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
            className="w-full"
          />
        </div>

        <div className="pt-3 border-t">
          <button
            onClick={reset}
            className="w-full rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm hover:bg-slate-50"
          >
            New conversation
          </button>
          {sessionId && (
            <div className="mt-2 text-[10px] font-mono text-slate-400 break-all">
              session {sessionId.slice(0, 12)}…
            </div>
          )}
        </div>
      </aside>

      {/* --- center: conversation --- */}
      <section className="flex flex-col rounded-xl border bg-white shadow-sm overflow-hidden">
        <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-5 space-y-4">
          {messages.length === 0 && (
            <div className="text-center text-slate-500 mt-12">
              <p className="text-lg font-medium">Ask anything about Spanish real estate</p>
              <p className="mt-2 text-sm">
                Try “What is ITP rate in Madrid?” or “Is the Golden Visa still available?”
              </p>
            </div>
          )}
          {messages.map((m, i) => (
            <MessageBubble key={i} message={m} />
          ))}
          {error && (
            <div className="rounded-md border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700">
              {error}
            </div>
          )}
        </div>

        <div className="border-t bg-slate-50 px-4 py-3">
          <div className="flex gap-2">
            <textarea
              className="flex-1 resize-none rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 placeholder:text-slate-400 shadow-sm focus:border-slate-500 focus:outline-none"
              rows={2}
              value={input}
              placeholder="Type a message…"
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              disabled={busy}
            />
            <div className="flex flex-col gap-1">
              <button
                onClick={send}
                disabled={busy || !input.trim() || !picked}
                className="rounded-md bg-slate-900 px-4 py-2 text-sm text-white hover:bg-slate-700 disabled:bg-slate-300"
              >
                Send
              </button>
              {busy && (
                <button
                  onClick={stop}
                  className="rounded-md border border-slate-300 bg-white px-4 py-1 text-xs hover:bg-slate-50"
                >
                  Stop
                </button>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* --- right: sources --- */}
      <aside className="overflow-y-auto rounded-xl border bg-slate-100 p-4 shadow-sm">
        <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">
          Retrieved sources
        </div>
        <SourcesPanel chunks={retrieved} />
      </aside>
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-2 text-sm whitespace-pre-wrap ${
          isUser
            ? "bg-slate-900 text-white"
            : "bg-slate-100 text-slate-900 border border-slate-200"
        }`}
      >
        {message.content}
        {message.streaming && (
          <span className="ml-0.5 inline-block h-3 w-1.5 animate-pulse bg-current align-middle" />
        )}
      </div>
    </div>
  );
}
