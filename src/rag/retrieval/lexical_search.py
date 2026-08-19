"""Lexical / BM25 sparse search over indexed CV chunks."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from rag.retrieval.embeddings import SparseEmbedder, sparse_to_qdrant
from rag.retrieval.qdrant_hits import hit_to_retrieved
from rag.retrieval.search_context import build_search_context
from rag.shared.schemas import RetrievedChunk
from rag.shared.settings import RagSettings

_EXACT_METADATA_FIELDS = ("institution", "company", "degree")


def _has_exact_metadata(filters: dict[str, Any] | None) -> bool:
    return bool(filters) and any(filters.get(field) for field in _EXACT_METADATA_FIELDS)


def search(
    query: str,
    *,
    keyword: str | None = None,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    client: QdrantClient | None = None,
    settings: RagSettings | None = None,
) -> list[RetrievedChunk]:
    """
    Sparse BM25 search (Case 2: exact keyword in education, company, etc.).

    When ``filters`` pins an exact metadata field (institution/company/degree),
    every matching chunk is returned via a payload scroll so that aggregate
    questions ("which candidates graduated from X") list all matches instead of a
    BM25-truncated subset. Otherwise falls back to sparse BM25 over ``keyword`` or
    ``query``.
    """
    ctx = build_search_context(
        filters=filters,
        top_k=top_k,
        client=client,
        settings=settings,
    )

    if _has_exact_metadata(filters):
        records, _ = ctx.client.scroll(
            collection_name=ctx.settings.qdrant_collection,
            scroll_filter=ctx.query_filter,
            limit=ctx.settings.profile_scroll_limit,
            with_payload=True,
            with_vectors=False,
        )
        return [hit_to_retrieved(record) for record in records]

    search_text = (keyword or query).strip()
    if not search_text:
        raise ValueError("query or keyword must be non-empty for lexical search")

    sparse = SparseEmbedder(ctx.settings).embed(search_text)

    response = ctx.client.query_points(
        collection_name=ctx.settings.qdrant_collection,
        query=qmodels.SparseVector(**sparse_to_qdrant(sparse)),
        using=ctx.settings.sparse_vector_name,
        query_filter=ctx.query_filter,
        limit=ctx.limit,
        with_payload=True,
    )
    return [hit_to_retrieved(point) for point in response.points]
