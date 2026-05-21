// Types mirror the Pydantic models on the FastAPI side. Keep them in sync
// when the backend evolves — this is hand-maintained, not generated, to
// avoid pulling in an OpenAPI codegen step for a learning project.
//
// One rule of thumb: if a field is optional on the backend (`x: T | None`),
// make it `x?: T` here; if it has a default, leave it required and document
// the default near the call site.

export type Provider = "openai" | "anthropic" | "ollama";

export interface ModelInfo {
  id: string;
  label: string;
}

export interface ProviderStatus {
  provider: Provider;
  available: boolean;
  default_model: string;
  models: ModelInfo[];
  error: string | null;
}

export interface ModelsResponse {
  providers: ProviderStatus[];
}

// --- /chat -----------------------------------------------------------------

export interface RetrievedChunk {
  text: string;
  chunk_id: string;
  source: string;
  title: string;
  distance: number;
  url?: string | null;
  region?: string | null;
  topic?: string | null;
}

export interface ChatRequest {
  provider: Provider;
  model: string;
  message: string;
  session_id?: string | null;
  use_rag?: boolean;
  rag_k?: number;
  rag_filters?: Record<string, string>;
  system_prompt?: string | null;
  temperature?: number;
  max_tokens?: number | null;
  stream?: boolean;
}

export interface ChatResponse {
  session_id: string;
  text: string;
  provider: string;
  model: string;
  finish_reason: string;
  prompt_tokens: number;
  completion_tokens: number;
  retrieved_context: RetrievedChunk[];
  latency_ms: number;
}

export interface SessionDetail {
  id: string;
  provider: string;
  model: string;
  messages: { role: string; content: string }[];
}

// --- /arena ----------------------------------------------------------------

export interface ArenaCandidate {
  provider: Provider;
  model: string;
}

export interface ArenaRequest {
  prompt: string;
  candidates: ArenaCandidate[];
  use_rag?: boolean;
  rag_k?: number;
  rag_filters?: Record<string, string>;
  system_prompt?: string | null;
  temperature?: number;
  max_tokens?: number | null;
}

export interface CandidateResult {
  provider: string;
  model: string;
  text: string;
  finish_reason: string;
  prompt_tokens: number;
  completion_tokens: number;
  latency_ms: number;
  error: string | null;
}

export interface ArenaResponse {
  prompt: string;
  retrieved_context: RetrievedChunk[];
  results: CandidateResult[];
}

// --- /eval -----------------------------------------------------------------

export interface RunSummary {
  run_id: string;
  started_at: number;
  finished_at: number | null;
  sut_combos: [string, string][];
  judge_combos: [string, string][];
  suites: string[];
  notes: string | null;
  results_count: number;
  passed_count: number;
  avg_score: number | null;
}

export interface ResultRow {
  id: number;
  run_id: string;
  ts: number;
  sut_provider: string;
  sut_model: string;
  judge_provider: string;
  judge_model: string;
  suite: string;
  metric: string;
  case_id: string;
  score: number | null;
  threshold: number | null;
  passed: number | null;          // 0 / 1 / null
  reason: string | null;
  duration_ms: number;
}

export interface RunDetail {
  run: Omit<RunSummary, "results_count" | "passed_count" | "avg_score">;
  results: ResultRow[];
}

export interface Cell {
  sut: string;
  judge: string;
  suite: string;
  metric: string;
  passed: number;
  total: number;
  avg_score: number | null;
}

export interface RunCells {
  run: Omit<RunSummary, "results_count" | "passed_count" | "avg_score">;
  cells: Cell[];
}

// --- /chat stream events ---------------------------------------------------

export type StreamEvent =
  | { type: "session"; data: { session_id: string } }
  | { type: "context"; data: { chunks: RetrievedChunk[] } }
  | { type: "delta"; data: { text: string } }
  | { type: "done"; data: { finish_reason: string; latency_ms: number; model: string; provider: string } }
  | { type: "error"; data: { message: string } };
