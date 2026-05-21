"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { ModelsResponse, Provider, ProviderStatus } from "@/lib/types";

interface Props {
  value: { provider: Provider; model: string } | null;
  onChange: (next: { provider: Provider; model: string }) => void;
  compact?: boolean;
}

/**
 * Two cascading selects: provider, then model from that provider's list.
 * Polls /models on mount; greys out providers that aren't configured.
 */
export function ModelPicker({ value, onChange, compact }: Props) {
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .models()
      .then(setModels)
      .catch((e) => setError(e.message));
  }, []);

  // Auto-pick the first available provider+model when the data lands.
  useEffect(() => {
    if (value || !models) return;
    const firstAvail = models.providers.find((p) => p.available && p.models.length);
    if (firstAvail) {
      onChange({
        provider: firstAvail.provider,
        model: firstAvail.default_model || firstAvail.models[0].id,
      });
    }
  }, [models, value, onChange]);

  if (error) {
    return (
      <div className="text-xs text-red-600">
        Could not load /models: {error}
      </div>
    );
  }
  if (!models) {
    return <div className="text-xs text-slate-500">Loading models…</div>;
  }

  const selected = models.providers.find((p) => p.provider === value?.provider);

  const labelCls = "text-xs font-medium text-slate-700 mb-1 block";
  const selectCls =
    "w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm shadow-sm focus:border-slate-500 focus:outline-none disabled:opacity-50";

  return (
    <div className={compact ? "grid grid-cols-2 gap-2" : "space-y-3"}>
      <div>
        <label className={labelCls}>Provider</label>
        <select
          className={selectCls}
          value={value?.provider || ""}
          onChange={(e) => {
            const p = models.providers.find((x) => x.provider === e.target.value);
            if (!p) return;
            const model = p.default_model || p.models[0]?.id || "";
            onChange({ provider: p.provider, model });
          }}
        >
          {models.providers.map((p) => (
            <option key={p.provider} value={p.provider} disabled={!p.available}>
              {p.provider}
              {!p.available ? " (configure key)" : ""}
            </option>
          ))}
        </select>
      </div>
      <div>
        <label className={labelCls}>Model</label>
        <select
          className={selectCls}
          value={value?.model || ""}
          onChange={(e) => {
            if (!value) return;
            onChange({ provider: value.provider, model: e.target.value });
          }}
          disabled={!selected?.available}
        >
          {selected?.models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          )) || <option>—</option>}
        </select>
      </div>
      {selected?.error && (
        <div className="text-xs text-amber-600 col-span-2">{selected.error}</div>
      )}
    </div>
  );
}

export type { ProviderStatus };
