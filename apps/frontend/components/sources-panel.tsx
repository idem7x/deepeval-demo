"use client";

import type { RetrievedChunk } from "@/lib/types";

interface Props {
  chunks: RetrievedChunk[];
  emptyHint?: string;
}

/**
 * Right-rail panel listing the chunks the RAG retriever returned for the
 * current turn. Distance is shown so users learn to read it as relevance
 * (lower = better with cosine distance).
 */
export function SourcesPanel({ chunks, emptyHint }: Props) {
  if (!chunks.length) {
    return (
      <div className="text-sm text-slate-500 italic">
        {emptyHint || "No sources yet — ask a question."}
      </div>
    );
  }
  return (
    <div className="space-y-3">
      {chunks.map((c) => (
        <div
          key={c.chunk_id}
          className="rounded-lg border bg-white p-3 text-xs shadow-sm"
        >
          <div className="flex items-center justify-between gap-2">
            <div className="font-semibold text-slate-800 line-clamp-1">
              {c.title}
            </div>
            <div className="shrink-0 text-[10px] tabular-nums text-slate-500">
              d={c.distance.toFixed(3)}
            </div>
          </div>
          <div className="mt-1 flex flex-wrap gap-1 text-[10px] text-slate-500">
            <span className="rounded bg-slate-100 px-1.5 py-0.5">{c.source}</span>
            {c.topic && (
              <span className="rounded bg-slate-100 px-1.5 py-0.5">
                {c.topic}
              </span>
            )}
            {c.region && (
              <span className="rounded bg-slate-100 px-1.5 py-0.5">
                {c.region}
              </span>
            )}
          </div>
          <details className="mt-2">
            <summary className="cursor-pointer text-slate-600 hover:text-slate-900">
              Show snippet
            </summary>
            <pre className="mt-1.5 max-h-48 overflow-auto whitespace-pre-wrap break-words text-slate-700">
              {c.text}
            </pre>
          </details>
          {c.url && (
            <a
              href={c.url}
              target="_blank"
              rel="noreferrer"
              className="mt-1 inline-block text-blue-600 hover:underline"
            >
              source ↗
            </a>
          )}
        </div>
      ))}
    </div>
  );
}
