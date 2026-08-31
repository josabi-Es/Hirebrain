"""Route retrieval requests to profile, lexical, or hybrid search."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

from rag.retrieval.hybrid_search import search as hybrid_search
from rag.retrieval.lexical_search import search as lexical_search
from rag.retrieval.vector_store import get_qdrant_client, scroll_by_candidate
from rag.shared.schemas import RetrievedChunk, SearchMode
from rag.shared.settings import RagSettings, get_settings


def retrieve(
    query: str,
    *,
    mode: SearchMode,
    filters: dict[str, Any] | None = None,
    keyword: str | None = None,
    candidate_name: str | None = None,
    doc_id: str | None = None,
    top_k: int | None = None,
    rerank_results: bool = True,
    rerank_top_k: int | None = None,
    client: QdrantClient | None = None,
    settings: RagSettings | None = None,
) -> list[RetrievedChunk]:
    """
    Unified retrieval entry point for the LangGraph agent.

    Dispatches by ``mode``:
    - PROFILE: scroll all chunks for one candidate (no vector search)
    - LEXICAL: sparse BM25 with optional keyword and filters
    - HYBRID: dense+sparse RRF with optional rerank
    """
    active = settings or get_settings()
    qdrant = client or get_qdrant_client(active)

    if mode == SearchMode.PROFILE:
        return scroll_by_candidate(
            qdrant,
            candidate_name=candidate_name,
            doc_id=doc_id,
            settings=active,
        )

    if mode == SearchMode.LEXICAL:
        return lexical_search(
            query,
            keyword=keyword,
            filters=filters,
            top_k=top_k,
            client=qdrant,
            settings=active,
        )

    if mode == SearchMode.HYBRID:
        return hybrid_search(
            query,
            filters=filters,
            top_k=top_k,
            rerank_results=rerank_results,
            rerank_top_k=rerank_top_k,
            client=qdrant,
            settings=active,
        )

    raise ValueError(f"Unsupported search mode: {mode!r}")
