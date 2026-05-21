"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { Cell, ResultRow, RunSummary } from "@/lib/types";

export default function EvalPage() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    api
      .evalRuns()
      .then((r) => {
        setRuns(r.runs);
        if (r.runs[0]) setSelected(r.runs[0].run_id);
      })
      .catch((e) => setError(e.message));
  }, []);

  return (
    <div className="mx-auto max-w-7xl px-4 py-6 space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Eval runs</h1>
        <p className="text-sm text-slate-600 mt-1">
          DeepEval matrix runs (SUT × judge × suite). Populate via{" "}
          <code className="rounded bg-slate-100 px-1.5 py-0.5 text-xs">
            python -m scripts.run_eval_matrix
          </code>
          .
        </p>
      </header>

      {error && (
        <div className="rounded-md border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700">
          Could not load: {error}
        </div>
      )}

      {!runs && !error && (
        <div className="text-sm text-slate-500">Loading…</div>
      )}

      {runs && runs.length === 0 && (
        <div className="rounded-xl border bg-white p-6 shadow-sm text-sm text-slate-500">
          No runs recorded yet. Run{" "}
          <code className="rounded bg-slate-100 px-1.5 py-0.5">
            make eval-matrix
          </code>{" "}
          to populate.
        </div>
      )}

      {runs && runs.length > 0 && (
        <>
          <RunsTable runs={runs} selected={selected} onSelect={setSelected} />
          {selected && <RunDetail runId={selected} />}
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------

function RunsTable({
  runs,
  selected,
  onSelect,
}: {
  runs: RunSummary[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <section className="rounded-xl border bg-white shadow-sm overflow-hidden">
      <div className="border-b px-4 py-2 text-xs uppercase tracking-wide text-slate-500">
        Runs ({runs.length})
      </div>
      <table className="w-full text-sm">
        <thead className="bg-slate-50 text-xs uppercase text-slate-500">
          <tr>
            <th className="px-4 py-2 text-left">Run</th>
            <th className="px-4 py-2 text-left">When</th>
            <th className="px-4 py-2 text-left">SUTs</th>
            <th className="px-4 py-2 text-left">Judges</th>
            <th className="px-4 py-2 text-left">Suites</th>
            <th className="px-4 py-2 text-right">Pass / Total</th>
            <th className="px-4 py-2 text-right">Avg score</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => {
            const active = r.run_id === selected;
            return (
              <tr
                key={r.run_id}
                onClick={() => onSelect(r.run_id)}
                className={`cursor-pointer border-t ${
                  active ? "bg-slate-100" : "hover:bg-slate-50"
                }`}
              >
                <td className="px-4 py-2 font-mono text-xs">{r.run_id}</td>
                <td className="px-4 py-2 text-xs text-slate-600">
                  {fmtTs(r.started_at)}
                </td>
                <td className="px-4 py-2 text-xs">
                  {r.sut_combos.map((s) => s.join("/")).join(", ")}
                </td>
                <td className="px-4 py-2 text-xs">
                  {r.judge_combos.map((j) => j.join("/")).join(", ")}
                </td>
                <td className="px-4 py-2 text-xs">{r.suites.join(", ")}</td>
                <td className="px-4 py-2 text-right tabular-nums">
                  {r.passed_count}/{r.results_count}
                </td>
                <td className="px-4 py-2 text-right tabular-nums">
                  {r.avg_score != null ? r.avg_score.toFixed(2) : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}

// ---------------------------------------------------------------------------

function RunDetail({ runId }: { runId: string }) {
  const [cells, setCells] = useState<Cell[] | null>(null);
  const [results, setResults] = useState<ResultRow[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    setCells(null);
    setResults(null);
    setErr(null);
    Promise.all([api.evalRunCells(runId), api.evalRun(runId)])
      .then(([c, d]) => {
        setCells(c.cells);
        setResults(d.results);
      })
      .catch((e) => setErr(e.message));
  }, [runId]);

  if (err) {
    return (
      <div className="rounded-md border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700">
        {err}
      </div>
    );
  }
  if (!cells || !results) {
    return <div className="text-sm text-slate-500">Loading run…</div>;
  }

  return (
    <div className="space-y-6">
      <Heatmap cells={cells} />
      <ResultsTable rows={results} />
    </div>
  );
}

// ---------------------------------------------------------------------------

function Heatmap({ cells }: { cells: Cell[] }) {
  // Rows = SUT|judge|suite ; columns = metric.
  const rowKeys = uniq(cells.map((c) => `${c.sut} ‖ judge:${c.judge} ‖ ${c.suite}`));
  const colKeys = uniq(cells.map((c) => c.metric));
  const byCell = new Map<string, Cell>();
  for (const c of cells) {
    byCell.set(`${c.sut} ‖ judge:${c.judge} ‖ ${c.suite}||${c.metric}`, c);
  }

  return (
    <section className="rounded-xl border bg-white shadow-sm overflow-hidden">
      <div className="border-b px-4 py-2 text-xs uppercase tracking-wide text-slate-500">
        Heatmap — avg score per (SUT, judge, suite) × metric
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="bg-slate-50">
            <tr>
              <th className="sticky left-0 z-10 bg-slate-50 px-3 py-2 text-left">
                SUT ‖ judge ‖ suite
              </th>
              {colKeys.map((m) => (
                <th key={m} className="px-3 py-2 text-center align-bottom">
                  <div className="whitespace-nowrap">{shortMetric(m)}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rowKeys.map((rk) => (
              <tr key={rk} className="border-t">
                <td className="sticky left-0 z-10 bg-white px-3 py-2 font-mono">
                  {rk}
                </td>
                {colKeys.map((m) => {
                  const c = byCell.get(`${rk}||${m}`);
                  if (!c) {
                    return (
                      <td key={m} className="px-3 py-2 text-center text-slate-300">
                        —
                      </td>
                    );
                  }
                  return (
                    <td
                      key={m}
                      className="px-3 py-2 text-center tabular-nums"
                      style={{ background: scoreColor(c.avg_score) }}
                      title={`${c.passed}/${c.total} passed`}
                    >
                      <div className="font-medium">
                        {c.avg_score != null ? c.avg_score.toFixed(2) : "—"}
                      </div>
                      <div className="text-[10px] text-slate-600">
                        {c.passed}/{c.total}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ResultsTable({ rows }: { rows: ResultRow[] }) {
  return (
    <section className="rounded-xl border bg-white shadow-sm overflow-hidden">
      <div className="border-b px-4 py-2 text-xs uppercase tracking-wide text-slate-500">
        All results ({rows.length})
      </div>
      <div className="max-h-[600px] overflow-y-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-slate-50 text-[10px] uppercase text-slate-500">
            <tr>
              <th className="px-3 py-2 text-left">SUT</th>
              <th className="px-3 py-2 text-left">Judge</th>
              <th className="px-3 py-2 text-left">Suite</th>
              <th className="px-3 py-2 text-left">Metric</th>
              <th className="px-3 py-2 text-left">Case</th>
              <th className="px-3 py-2 text-right">Score</th>
              <th className="px-3 py-2 text-right">Thr.</th>
              <th className="px-3 py-2 text-center">Pass</th>
              <th className="px-3 py-2 text-left">Reason</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className="border-t">
                <td className="px-3 py-2 whitespace-nowrap">
                  {r.sut_provider}/{r.sut_model}
                </td>
                <td className="px-3 py-2 whitespace-nowrap">
                  {r.judge_provider}/{r.judge_model}
                </td>
                <td className="px-3 py-2">{r.suite}</td>
                <td className="px-3 py-2">{shortMetric(r.metric)}</td>
                <td className="px-3 py-2 font-mono text-[11px]">{r.case_id}</td>
                <td className="px-3 py-2 text-right tabular-nums">
                  {r.score != null ? r.score.toFixed(2) : "—"}
                </td>
                <td className="px-3 py-2 text-right text-slate-500 tabular-nums">
                  {r.threshold != null ? r.threshold.toFixed(2) : "—"}
                </td>
                <td className="px-3 py-2 text-center">
                  {r.passed === 1 ? (
                    <span className="text-emerald-600">✓</span>
                  ) : r.passed === 0 ? (
                    <span className="text-red-600">✗</span>
                  ) : (
                    <span className="text-slate-400">—</span>
                  )}
                </td>
                <td className="px-3 py-2 max-w-[480px] text-slate-700">
                  <span className="line-clamp-2" title={r.reason || ""}>
                    {r.reason || "—"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------

function uniq<T>(xs: T[]): T[] {
  return Array.from(new Set(xs));
}

function fmtTs(ts: number): string {
  return new Date(ts * 1000).toLocaleString();
}

function shortMetric(name: string): string {
  return name.replace(/Metric$/, "");
}

function scoreColor(score: number | null): string {
  if (score == null) return "transparent";
  // 0 → red-100, 0.5 → amber-100, 1 → emerald-100
  if (score < 0.4) return "rgb(254 226 226)";   // red-100
  if (score < 0.7) return "rgb(254 243 199)";   // amber-100
  return "rgb(209 250 229)";                    // emerald-100
}
