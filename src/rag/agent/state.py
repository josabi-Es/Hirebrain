"""LangGraph agent state."""

from __future__ import annotations

from typing import Any, TypedDict

from rag.shared.schemas import SearchMode


class AgentState(TypedDict, total=False):
    """Shared state for the CV screening agent graph."""

    query: str
    rewritten_query: str
    in_domain: bool
    route_reason: str
    search_mode: SearchMode
    keyword: str | None
    candidate_name: str | None
    doc_id: str | None
    filters: dict[str, Any] | None
    retrieved_chunks: list[dict[str, Any]]
    reranked_chunks: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    thread_id: str
    answer: str
