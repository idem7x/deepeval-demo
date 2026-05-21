"use client";

import { useEffect, useState } from "react";

const BASE_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";

interface RunsResp {
  runs: unknown[];
  _note?: string;
}

export default function EvalPage() {
  const [data, setData] = useState<RunsResp | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${BASE_URL}/eval/runs`)
      .then((r) => r.json())
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <div className="mx-auto max-w-5xl px-4 py-6 space-y-5">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Eval runs</h1>
        <p className="text-sm text-slate-600 mt-1">
          DeepEval matrix runs (models × suites × judges) appear here once
          Phase 7 lands. Until then, this page just confirms the backend
          endpoint responds.
        </p>
      </header>

      <div className="rounded-xl border bg-white p-6 shadow-sm">
        {error && (
          <div className="text-sm text-red-700">Could not load: {error}</div>
        )}
        {!error && !data && (
          <div className="text-sm text-slate-500">Loading…</div>
        )}
        {data && data.runs.length === 0 && (
          <div className="text-sm text-slate-500">
            No runs recorded yet.
            {data._note && (
              <div className="mt-2 rounded-md bg-slate-50 px-3 py-2 text-xs text-slate-600">
                {data._note}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
