# DeepEval Lab — common commands
#
# Run `make help` to see what's available.

.PHONY: help install install-dev dev lint format test-smoke test-local \
        ingest eval-matrix clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --- Install ------------------------------------------------------------------

install: ## Install runtime deps only (core + llm + rag + eval)
	pip install -e ".[llm,rag,eval]"

install-dev: ## Install everything including dev tools
	pip install -e ".[all]"

# --- Run ----------------------------------------------------------------------

dev: ## Start the FastAPI backend with reload
	uvicorn apps.backend.main:app --reload \
	  --host $${BACKEND_HOST:-127.0.0.1} \
	  --port $${BACKEND_PORT:-8000}

# --- Lint / format ------------------------------------------------------------

lint: ## Run ruff + mypy
	ruff check .
	mypy apps

format: ## Auto-format code
	ruff format .
	ruff check --fix .

# --- Tests --------------------------------------------------------------------

test-smoke: ## Fast cheap tests, same as CI (gpt-4o-mini as SUT + judge)
	pytest -m smoke -v

test-local: ## Heavy tests (Ollama, full red-teaming, etc.) — local only
	pytest -m "local or smoke" -v

# --- Data ---------------------------------------------------------------------

ingest: ## Build / rebuild ChromaDB index from knowledge/
	python -m scripts.seed_chroma

# --- Eval matrix --------------------------------------------------------------

eval-matrix: ## Run the full eval matrix (models x suites x judges) locally
	python -m scripts.run_eval_matrix

# --- Cleanup ------------------------------------------------------------------

clean: ## Remove caches and local indexes
	rm -rf .pytest_cache .mypy_cache .ruff_cache .deepeval-cache
	rm -rf .chroma eval_results
	find . -type d -name __pycache__ -exec rm -rf {} +
