"use client";

import { useEffect, useState } from "react";
import { ModelPicker } from "@/components/model-picker";
import { SourcesPanel } from "@/components/sources-panel";
import { api } from "@/lib/api";
import type {
  ArenaResponse,
  CandidateResult,
  Provider,
} from "@/lib/types";

type Slot = { provider: Provider; model: string } | null;

const STARTING_PROMPT =
  "Is the Spanish Golden Visa still available for new applicants in 2026?";

export default function ArenaPage() {
  const [slots, setSlots] = useState<Slot[]>([null, null]);
  const [prompt, setPrompt] = useState(STARTING_PROMPT);
  const [useRag, setUseRag] = useState(true);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<ArenaResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  function addSlot() {
    if (slots.length >= 5) return;
    setSlots([...slots, null]);
  }
  function removeSlot(i: number) {
    if (slots.length <= 2) return;
    setSlots(slots.filter((_, idx) => idx !== i));
  }
  function setSlot(i: number, value: NonNullable<Slot>) {
    const next = [...slots];
    next[i] = value;
    setSlots(next);
  }

  async function compare() {
    setError(null);
    const candidates = slots.filter((s): s is NonNullable<Slot> => s !== null);
    if (candidates.length < 2 || !prompt.trim()) {
      setError("Pick at least two models and enter a prompt.");
      return;
    }
    setBusy(true);
    setResult(null);
    try {
      const r = await api.arena({
        prompt: prompt.trim(),
        candidates,
        use_rag: useRag,
      });
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-6 space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Arena</h1>
        <p className="text-sm text-slate-600 mt-1">
          Send one prompt to 2–5 models in parallel. RAG context is retrieved
          <em> once</em> and shared across them so you measure the model, not
          retrieval variance.
        </p>
      </header>

      <section className="rounded-xl border bg-white p-5 shadow-sm space-y-4">
        <div>
          <label className="text-xs font-medium text-slate-700 mb-1 block">
            Prompt
          </label>
          <textarea
            className="w-full resize-none rounded-md border border-slate-300 bg-white px-3 py-2 text-sm shadow-sm"
            rows={3}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </div>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {slots.map((s, i) => (
            <div key={i} className="rounded-md border border-slate-200 p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-slate-700">
                  Candidate {i + 1}
                </span>
                {slots.length > 2 && (
                  <button
                    onClick={() => removeSlot(i)}
                    className="text-xs text-slate-500 hover:text-red-600"
                  >
                    remove
                  </button>
                )}
              </div>
              <ModelPicker
                value={s}
                onChange={(v) => setSlot(i, v)}
                compact
              />
            </div>
          ))}
          {slots.length < 5 && (
            <button
              onClick={addSlot}
              className="rounded-md border border-dashed border-slate-300 px-3 py-6 text-sm text-slate-600 hover:bg-slate-50"
            >
              + Add candidate
            </button>
          )}
        </div>

        <div className="flex items-center justify-between border-t pt-3">
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={useRag}
              onChange={(e) => setUseRag(e.target.checked)}
            />
            Use RAG (retrieve from knowledge base)
          </label>
          <button
            onClick={compare}
            disabled={busy}
            className="rounded-md bg-slate-900 px-5 py-2 text-sm text-white hover:bg-slate-700 disabled:bg-slate-300"
          >
            {busy ? "Running…" : "Compare"}
          </button>
        </div>

        {error && (
          <div className="rounded-md border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700">
            {error}
          </div>
        )}
      </section>

      {result && (
        <>
          <section>
            <h2 className="mb-3 text-sm font-medium text-slate-700">
              Results ({result.results.length})
            </h2>
            <div className="grid gap-4 lg:grid-cols-2 xl:grid-cols-3">
              {result.results.map((r, i) => (
                <ResultCard key={i} r={r} />
              ))}
            </div>
          </section>

          <section className="rounded-xl border bg-slate-100 p-4 shadow-sm">
            <h2 className="mb-3 text-xs uppercase tracking-wide text-slate-600">
              Shared retrieved context
            </h2>
            <SourcesPanel
              chunks={result.retrieved_context}
              emptyHint="RAG was off — candidates answered without retrieved context."
            />
          </section>
        </>
      )}
    </div>
  );
}

function ResultCard({ r }: { r: CandidateResult }) {
  return (
    <div className="rounded-xl border bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between gap-2">
        <div className="text-sm font-semibold">
          <span className="text-slate-600">{r.provider}</span>
          <span className="mx-1">/</span>
          <span>{r.model}</span>
        </div>
        <div className="text-[10px] tabular-nums text-slate-500">
          {r.latency_ms}ms · {r.prompt_tokens + r.completion_tokens}t
        </div>
      </div>
      {r.error ? (
        <div className="mt-2 rounded-md border border-red-300 bg-red-50 px-2 py-1 text-xs text-red-700">
          {r.error}
        </div>
      ) : (
        <div className="mt-3 whitespace-pre-wrap text-sm text-slate-800">
          {r.text}
        </div>
      )}
    </div>
  );
}
