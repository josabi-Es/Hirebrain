# Hirebrain

CV screening with RAG: generate synthetic candidate CVs and query them through a
retrieval-augmented pipeline backed by Ollama.

## Setup (Ollama already running on `localhost:11434`)

```bash
# 1. Install deps
uv sync --extra rag --extra ui --extra dev

# 2. Config
cp .env.template .env

# 3. Vector store (Qdrant only — Ollama runs natively)
docker compose up -d qdrant
curl http://127.0.0.1:6333/readyz    # ready check
```

## Run

```bash
uv run cv-creator -n 10 --seed 42                          # generate CVs -> data/cvs/
uv run cv-extract --input-dir data/cvs --export-chunks-dir  # extract + chunk
uv run cv-index                                             # index into Qdrant

uv run cv-chat "Summarize the profile of Timothy Thompson" --thread-id demo --verbose
uv run cv-ask "Who has experience with Python?"             # retrieval only, no agent

uvicorn rag.api.app:app --reload                             # REST API on :8000
chainlit run src/rag/frontend/chainlit_app.py -w             # web UI on :8000
```

## Tests

```bash
uv run pytest
```
