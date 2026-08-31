# From local demo to a recruiter's daily tool

The scenario behind this project is simple. An HR recruiter posts a job on LinkedIn and gets
about 100 CVs back. Reading all of them by hand, and remembering who said what, does not work
once you have more than one open posting.

## What the local setup already does

Point `cv-extract` and `cv-index` at that posting's 100 PDFs. Then the recruiter can ask plain
questions like "Who has experience with Python?", "Summarize Jane Doe's profile", or "Which
candidates graduated from a target university?". The answer is grounded in the actual CV text
and comes with the source chunks attached, not a guess. That is the whole idea, and it already
works end to end on a laptop.

## What has to change at scale

More postings, more candidates, and more recruiters using it at the same time all add cost.
None of this is built yet, but the codebase is already shaped for these changes:

- **Ollama to a real inference backend.** All LLM calls go through a few small functions
  (`rag/agent/llm.py::chat` and `chat_json`, `cv_creator/integration/llm_service.py::_call_ollama`)
  instead of being scattered across the code. Swapping Ollama for a GPU-backed model server, or
  a hosted LLM API, means changing one or two files, not rewriting the project. This matters as
  soon as query volume goes past what one laptop GPU can handle at once.

- **One Qdrant collection to per-posting or filtered multi-tenancy.** Right now everything sits
  in one `cv_chunks` collection. At scale, a recruiter should only search their own postings.
  That means either one collection per posting, or a shared collection filtered by a
  `posting_id` field. `rag/retrieval/qdrant_payload.py` already stores structured metadata per
  chunk, so that field would live there.

- **Manual ingestion to an async queue.** Today, indexing a new batch of CVs is a manual step:
  run `cv-extract`, then `cv-index`. A real deployment needs new CVs indexed as they arrive, for
  example through a queue consumer built around `rag/ingest/pipeline.py`, not a batch job
  someone has to remember to run.

- **An open API to an authenticated one.** `rag/api/app.py` has no auth layer today. That is
  fine for local use, but not once a second recruiter's data needs to stay separate from the
  first one's.

- **Basic logs to real observability.** The request logs you already see locally are a starting
  point, not a monitoring system. A production setup needs latency and failure tracking per
  agent step (router, rewrite, retrieve, generate), since each one is a separate LLM call that
  can fail or drift on its own.

None of this is implemented in this repo. It is an honest answer to "what would it take to run
this for real", not a roadmap with a deadline. There is no Terraform, no Kubernetes, no new code
here. This is a design note, written because the architecture already supports these changes
without a rewrite, not because the infrastructure exists yet.
