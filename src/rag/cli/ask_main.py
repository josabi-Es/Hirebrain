"""CLI to test RAG retrieval and optional Ollama answer generation."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from typing import Any

from dotenv import load_dotenv

from rag.agent.llm import answer_temperature
from rag.cli.intent import SearchPlan, detect_search_plan, example_queries
from rag.retrieval.router import retrieve
from rag.shared.schemas import RetrievedChunk, SearchMode

load_dotenv()

_ANSWER_PROMPT = """You are a CV screening assistant. Answer ONLY using the context chunks below.
If the context is insufficient, say you do not have enough evidence.
Cite candidate names and sections when making claims.
Do not invent skills, employers, or degrees not present in the context.

Question:
{question}

Context chunks:
{context}

Answer in clear English. End with a "Sources:" line listing doc_id and section for each chunk used.
"""


def _parse_filters(raw: list[str] | None) -> dict[str, Any] | None:
    if not raw:
        return None
    filters: dict[str, Any] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Invalid filter (expected key=value): {item!r}")
        key, value = item.split("=", 1)
        filters[key.strip()] = value.strip()
    return filters


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        payload = chunk.payload
        blocks.append(
            f"[{index}] candidate={payload.candidate_name!r} "
            f"doc_id={payload.doc_id!r} section={payload.section!r} "
            f"score={chunk.score:.4f}\n{payload.text.strip()}"
        )
    return "\n\n".join(blocks)


def _print_hits(chunks: list[RetrievedChunk]) -> None:
    if not chunks:
        print("No chunks retrieved.")
        return

    print(f"\n--- Retrieved ({len(chunks)} chunks) ---")
    for index, chunk in enumerate(chunks, start=1):
        payload = chunk.payload
        print(
            f"\n[{index}] score={chunk.score:.4f} | "
            f"{payload.candidate_name} | {payload.section} | {payload.chunk_id}"
        )
        preview = payload.text.strip().replace("\n", " ")
        if len(preview) > 220:
            preview = preview[:217] + "..."
        print(preview)


def _generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError("ollama package is required for --generate") from exc

    model = os.getenv("OLLAMA_MODEL", "llama3:8b")
    host = os.getenv("OLLAMA_HOST")
    client_kwargs: dict[str, str] = {}
    if host:
        client_kwargs["host"] = host
    client = ollama.Client(**client_kwargs) if client_kwargs else ollama

    prompt = _ANSWER_PROMPT.format(
        question=question.strip(),
        context=_format_context(chunks) if chunks else "(no context retrieved)",
    )
    response = client.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Answer only from provided CV context. Be concise and factual.",
            },
            {"role": "user", "content": prompt},
        ],
        options={"temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))},
    )
    content = response.get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("Ollama returned an empty response")
    return content.strip()


def _run_plan(
    plan: SearchPlan,
    *,
    top_k: int | None,
    rerank_results: bool,
    generate: bool,
) -> int:
    print(
        f"Mode: {plan.mode.value}"
        + (f" | candidate={plan.candidate_name!r}" if plan.candidate_name else "")
        + (f" | keyword={plan.keyword!r}" if plan.keyword else "")
        + (f" | filters={plan.filters!r}" if plan.filters else "")
    )

    try:
        chunks = retrieve(
            plan.query,
            mode=plan.mode,
            filters=plan.filters,
            keyword=plan.keyword,
            candidate_name=plan.candidate_name,
            doc_id=plan.doc_id,
            top_k=top_k,
            rerank_results=rerank_results,
        )
    except Exception as exc:
        print(f"Retrieval error: {exc}", file=sys.stderr)
        return 1

    _print_hits(chunks)

    if generate:
        print("\n--- Answer (Ollama) ---")
        try:
            answer = _generate_answer(plan.query, chunks)
        except Exception as exc:
            print(f"Generation error: {exc}", file=sys.stderr)
            return 1
        print(answer)

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test RAG retrieval against Qdrant (optional Ollama answer).",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Natural-language question (optional if --candidate or --keyword is set).",
    )
    parser.add_argument(
        "--mode",
        choices=[m.value for m in SearchMode] + ["auto"],
        default="auto",
        help="Retrieval mode (default: auto-detect from question).",
    )
    parser.add_argument(
        "--candidate",
        help="Candidate full name (profile mode).",
    )
    parser.add_argument(
        "--keyword",
        help="Exact keyword for lexical mode (e.g. university name).",
    )
    parser.add_argument(
        "--filter",
        action="append",
        dest="filters",
        metavar="KEY=VALUE",
        help="Payload filter, e.g. --filter section=EDUCATION",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Max chunks to retrieve (hybrid/lexical).",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder rerank in hybrid mode.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate an answer with Ollama from retrieved chunks.",
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Run three built-in example queries against the corpus.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Alias for default behaviour (show chunks, no LLM).",
    )
    return parser


def _default_query(args: argparse.Namespace) -> str:
    """Build a display/LLM query when the positional question is omitted."""
    if args.question:
        return args.question.strip()
    if args.keyword:
        return f"Which candidate graduated from {args.keyword}?"
    if args.candidate:
        return f"Summarize the profile of {args.candidate}"
    raise ValueError(
        "Provide a question, use --examples, or pass --candidate / --keyword."
    )


def _resolve_plan(args: argparse.Namespace) -> SearchPlan:
    # _default_query raises the "provide a question/--candidate/--keyword" error
    # when nothing usable was passed, so no separate guard is needed here.
    query = _default_query(args)
    filters = _parse_filters(args.filters)

    base = (
        detect_search_plan(query)
        if args.mode == "auto"
        else SearchPlan(mode=SearchMode(args.mode), query=query)
    )

    # Explicit --candidate / --keyword flags override the auto-detected plan while
    # keeping every other field. dataclasses.replace avoids re-listing all fields.
    if args.candidate:
        mode = SearchMode.PROFILE if args.mode in ("auto", "profile") else base.mode
        return replace(
            base,
            mode=mode,
            candidate_name=args.candidate,
            filters=filters or base.filters,
        )
    if args.keyword:
        mode = SearchMode.LEXICAL if args.mode in ("auto", "lexical") else base.mode
        return replace(
            base,
            mode=mode,
            keyword=args.keyword,
            filters=filters or base.filters or {"section": "EDUCATION"},
        )
    if filters:
        return replace(base, filters=filters)

    return base


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.retrieval_only and args.generate:
        print("Use either --retrieval-only or --generate, not both.", file=sys.stderr)
        return 2

    rerank_results = not args.no_rerank

    if args.examples:
        exit_code = 0
        for question, plan in example_queries():
            print("\n" + "=" * 72)
            print(f"Question: {question}")
            code = _run_plan(
                plan,
                top_k=args.top_k,
                rerank_results=rerank_results,
                generate=args.generate,
            )
            exit_code = max(exit_code, code)
        return exit_code

    try:
        plan = _resolve_plan(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return _run_plan(
        plan,
        top_k=args.top_k,
        rerank_results=rerank_results,
        generate=args.generate,
    )


if __name__ == "__main__":
    raise SystemExit(main())
