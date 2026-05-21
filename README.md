# DeepEval Lab

A hands-on playground that demonstrates **every metric category** in
[DeepEval](https://github.com/confident-ai/deepeval) on a single, realistic
domain: **Spanish real estate**.

It is built as a learning project — the goal is to make the eval flow visible
end-to-end: from raw documents → RAG retrieval → multi-provider chat → judge
models → metric scores → side-by-side model comparison → CI gate.

---

## What's inside

- **Multi-provider chat**: OpenAI, Anthropic, and local Ollama models behind a
  single adapter. The same adapter is used for the chatbot (SUT) **and** for
  the DeepEval judge — so you can grade GPT with Claude, Claude with a local
  Llama, etc.
- **RAG pipeline** on top of a local **ChromaDB** index, seeded from
  hand-curated docs, Wikipedia, and official Spanish government PDFs.
- **Full DeepEval metric coverage** across seven categories: single-turn,
  conversational, RAG, agentic, safety / red-teaming, multimodal,
  and custom (`G-Eval`, `DAG`, `ArenaGEval`).
- **Web app**: FastAPI backend + Next.js frontend with a chat page (SSE
  streaming + retrieved sources), an arena (side-by-side) page, and an
  eval dashboard with a heatmap of model × judge × metric scores.
- **Eval matrix runner**: cartesian product of SUTs × judges × suites,
  persisted to SQLite, surfaced through `/eval` in the UI.
- **CI**: smoke gate on every PR (no LLM cost), full live-LLM smoke
  gate on push to main (~$0.05 per run).

---

## Architecture at a glance

```
                    ┌──────────────┐
        user ─────► │  Next.js UI  │
                    └──────┬───────┘
                           │ SSE
                    ┌──────▼───────┐    ┌────────────────┐
                    │   FastAPI    │◄──►│  LLM adapters  │── OpenAI / Anthropic / Ollama
                    │   backend    │    └────────────────┘
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼─────┐      ┌─────▼─────┐      ┌─────▼──────┐
   │ ChromaDB │      │  DeepEval │      │ Eval runs  │
   │  (RAG)   │      │  metrics  │      │  (SQLite)  │
   └──────────┘      └───────────┘      └────────────┘
```

---

## Phases

The repo is built up incrementally — each phase is a single commit and
ships something working.

| Phase | Commit | Goal |
|-------|--------|------|
| 0 | `c8dbdf9` | Repo scaffold, deps, `.env`, `Makefile`, FastAPI skeleton |
| 1 | `433b380` | Knowledge: 11 curated + 14 Wikipedia + 2 BOE PDFs |
| 2 | `7ddd675` | RAG: 3 embeddings, chunker, ChromaStore, retriever |
| 3 | `5b0920a` | LLM adapters (OpenAI/Anthropic/Ollama) + DeepEval LLM wrapper |
| 4 | `98982f8` | FastAPI endpoints (`/chat` SSE, `/arena`, `/models`, `/eval`) |
| 5 | `089b389` | Next.js frontend (chat, arena, eval pages) |
| 6 | `4b12b63` | DeepEval suites — every metric category |
| 7 | `953ea68` | Eval matrix runner + dashboard + Anthropic TLS/system fixes |
| 8 | _this_  | GitHub Actions smoke CI |

---

## Quickstart

```bash
# 1. Clone and install
git clone <your-fork-url> deepeval-lab && cd deepeval-lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[llm,rag,eval]"

# 2. Configure secrets
cp .env.example .env
# edit .env — at minimum set OPENAI_API_KEY for the eval suite

# 3. (Optional) install Ollama for local models
#    https://ollama.com
ollama pull llama3.1:8b
ollama pull nomic-embed-text   # only if you want Ollama-side embeddings

# 4. Seed the RAG index (one-off, ~30s; downloads a ~80MB ONNX model
#    on first run for the default ChromaDB embedder)
make ingest

# 5. Run the backend
make dev   # → http://127.0.0.1:8000/health

# 6. (Other terminal) run the frontend
cd apps/frontend
cp .env.local.example .env.local
npm install
npm run dev   # → http://localhost:3000
```

The chat is at `/chat`, the side-by-side arena at `/arena`, and the
eval dashboard at `/eval`.

---

## Common commands

```bash
# Tests
make test-smoke               # fast, cheap, what CI runs (PRs)
make test-local               # full local suite (Ollama, strict thresholds)

# Eval matrix (real LLM calls; persists to eval_results/runs.sqlite3)
python -m scripts.run_eval_matrix
python -m scripts.run_eval_matrix \
  --sut openai/gpt-4o-mini \
  --judge anthropic/claude-haiku-4-5 \
  --suite rag --suite single_turn

# Knowledge utilities
python -m scripts.list_knowledge --by topic
python -m knowledge.scripts.ingest_wikipedia --skip-existing
python -m knowledge.scripts.ingest_pdfs

# Clean
make clean                    # caches + .chroma + eval_results
```

---

## Continuous integration

`.github/workflows/smoke.yml` runs in **two tiers**:

1. **`tests` job** — runs on every pull request and every push.
   - No secrets, no LLM calls, ~1 minute.
   - Covers the knowledge loader, RAG pipeline (offline ONNX embedder),
     LLM adapter shape, and the FastAPI endpoints (via TestClient with
     fakes).

2. **`eval` job** — runs only on push to `main`, or manual
   `workflow_dispatch`.
   - Requires `OPENAI_API_KEY` (and optionally `ANTHROPIC_API_KEY`) in
     Settings → Secrets → Actions.
   - Calls real LLMs. Budget: roughly **$0.05 per run** with
     `gpt-4o-mini` as both SUT and judge.
   - Uploads `eval_results/` and `.deepeval/` as build artifacts so a
     failure can be inspected without re-running.

The `concurrency` block cancels in-flight runs when a new commit lands
on the same branch — useful when iterating on a PR.

### Adding the OpenAI secret

1. Repo → Settings → Secrets and variables → Actions → New repository secret
2. Name: `OPENAI_API_KEY`, value: your key (`sk-proj-…`)
3. Optional: add `ANTHROPIC_API_KEY` for Anthropic models in the suite

Without `OPENAI_API_KEY`, the eval job will fail-fast at the
`Verify OPENAI_API_KEY is set` step with a clear error message.

---

## Why Spanish real estate?

It hits a useful sweet spot for evaluation:

- **Factual but not trivial** — taxes (IRNR, ITP, IBI), regions, residency
  rules. Good for `Faithfulness` and `Hallucination` metrics.
- **Conversational** — buying a property is a multi-step dialog with
  follow-up questions. Good for `KnowledgeRetention` and
  `ConversationCompleteness`.
- **Has tools** — search listings, mortgage calculator. Good for agentic
  metrics.
- **Has images** — property photos. Good for multimodal metrics.
- **Sensitive** — legal/tax advice without disclaimers, bias by region or
  nationality. Good for safety / red-teaming.
- **Current** — the Golden Visa real-estate route was abolished on
  3 April 2025; sees if the model gets that right.

---

## Things worth noticing while reading the code

A few non-obvious quirks of the third-party stack that are documented
inline at their fix sites — pointers here for the curious:

- `apps/backend/llm/anthropic.py` pins httpx to TLS 1.2 max on macOS +
  Python 3.12 + OpenSSL 3.6, because TLS 1.3 ClientHello to
  `api.anthropic.com` is RST-ed in that environment.
- Anthropic Messages API rejects both `system=null` and bare-string
  systems; we always wrap as `[{"type":"text","text":...}]` and skip the
  kwarg entirely when there's nothing to send.
- DeepEval's `ContextualRelevancyMetric` is sentence-granular and our
  curated docs are intentionally rich, so it scores 0.05–0.15 even on
  well-retrieved questions. The smoke variant just checks the metric
  runs; the strict bar lives in `test_contextual_relevancy_full`
  (`@local`).
- DeepEval 4.0.3's `ArenaGEval` occasionally has the judge hallucinate
  a contestant name that isn't in its internal dummy mapping → KeyError.
  Flaky-by-design; moved to `@local`.
- DAG metrics: `VerdictNode` terminals must NOT be shared across branches
  — sharing silently sets `score=None`. Use a small `fail()` factory.

---

## License

MIT.
