"""Hybrid retrieval (dense + sparse fusion with optional reranking)."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from rag.retrieval.embeddings import DenseEmbedder, SparseEmbedder, sparse_to_qdrant
from rag.retrieval.qdrant_hits import hit_to_retrieved
from rag.retrieval.reranker import rerank as rerank_candidates
from rag.retrieval.search_context import build_search_context
from rag.shared.schemas import RetrievedChunk
from rag.shared.settings import RagSettings


def search(
    query: str,
    *,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    rerank_results: bool = True,
    rerank_top_k: int | None = None,
    client: QdrantClient | None = None,
    settings: RagSettings | None = None,
) -> list[RetrievedChunk]:
    """
    Hybrid search with RRF fusion (Case 3: open semantic skill queries).

    When ``rerank_results`` is True, results are reordered with the cross-encoder.
    """
    ctx = build_search_context(
        filters=filters,
        top_k=top_k,
        client=client,
        settings=settings,
    )

    dense = DenseEmbedder(ctx.settings).embed(query)
    sparse = SparseEmbedder(ctx.settings).embed(query)

    response = ctx.client.query_points(
        collection_name=ctx.settings.qdrant_collection,
        prefetch=[
            qmodels.Prefetch(
                query=dense,
                using=ctx.settings.dense_vector_name,
                filter=ctx.query_filter,
                limit=ctx.limit,
            ),
            qmodels.Prefetch(
                query=qmodels.SparseVector(**sparse_to_qdrant(sparse)),
                using=ctx.settings.sparse_vector_name,
                filter=ctx.query_filter,
                limit=ctx.limit,
            ),
        ],
        query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
        query_filter=ctx.query_filter,
        limit=ctx.limit,
        with_payload=True,
    )

    results = [hit_to_retrieved(point) for point in response.points]
    if rerank_results and results:
        return rerank_candidates(
            query,
            results,
            top_k=rerank_top_k or ctx.settings.rerank_top_k,
            settings=ctx.settings,
        )
    return results
