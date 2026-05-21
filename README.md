# DeepEval Lab

A hands-on playground that demonstrates **every metric category** in
[DeepEval](https://github.com/confident-ai/deepeval) on a single, realistic
domain: **Spanish real estate**.

It is built as a learning project — the goal is to make the eval flow visible
end-to-end: from raw documents → RAG retrieval → multi-provider chat → judge
models → metric scores → side-by-side model comparison.

---

## What's inside

- **Multi-provider chat**: OpenAI, Anthropic, and local Ollama models behind a
  single adapter. The same adapter is used for the chatbot (SUT) **and** for
  the DeepEval judge — so you can grade GPT with Claude, Claude with a local
  Llama, etc.
- **RAG pipeline** on top of a local **ChromaDB** index, seeded from
  hand-curated docs, Wikipedia, and official Spanish gov. PDFs.
- **Full DeepEval metric coverage** organised by category:
  single-turn, conversational, RAG, agentic, safety / red-teaming, multimodal,
  and custom (`G-Eval`, `DAG`, `ArenaGEval`).
- **Web app**: FastAPI backend + Next.js frontend with a chat page, an arena
  (side-by-side) page, and an eval dashboard.
- **CI**: smoke-level test suite runs on every pull request with a tight
  budget cap; full matrix runs locally on demand.

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
   │ ChromaDB │      │  DeepEval │      │  Eval runs │
   │  (RAG)   │      │  metrics  │      │  (SQLite)  │
   └──────────┘      └───────────┘      └────────────┘
```

---

## Phases

The repo is built up incrementally — each phase produces something working.

| Phase | Goal | Status |
|-------|------|--------|
| 0 | Repo scaffold, deps, `.env`, `Makefile`, FastAPI skeleton | in progress |
| 1 | Knowledge base ingestion (curated + Wikipedia + scraped PDFs) | todo |
| 2 | RAG pipeline (ChromaDB + embeddings) | todo |
| 3 | LLM adapters (OpenAI, Anthropic, Ollama) + DeepEval LLM wrapper | todo |
| 4 | FastAPI endpoints (`/chat`, `/models`, `/arena`, `/eval`) | todo |
| 5 | Next.js frontend (chat, arena, eval dashboard) | todo |
| 6 | DeepEval test suites — every metric category | todo |
| 7 | Eval matrix runner (models × suites × judges) → results store | todo |
| 8 | GitHub Actions smoke CI | todo |

---

## Quickstart

```bash
# 1. Clone and install (Phase 0+)
git clone <your-fork-url> deepeval-lab && cd deepeval-lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# 2. Configure secrets
cp .env.example .env
# edit .env — at minimum set OPENAI_API_KEY

# 3. (Optional, for local models) install Ollama and pull models
#    https://ollama.com
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull nomic-embed-text   # used for RAG embeddings

# 4. Run the backend
make dev
```

More commands live in the `Makefile`.

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

---

## License

MIT.
