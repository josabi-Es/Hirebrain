"""Cross-encoder reranking over retrieved candidates."""

from __future__ import annotations

from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from rag.shared.schemas import RetrievedChunk
from rag.shared.settings import RagSettings, get_settings


@lru_cache(maxsize=2)
def _get_reranker(model_name: str) -> TextCrossEncoder:
    return TextCrossEncoder(model_name=model_name)


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    *,
    top_k: int | None = None,
    settings: RagSettings | None = None,
) -> list[RetrievedChunk]:
    """Re-score and reorder retrieval candidates for a query."""
    if not candidates:
        return []

    active = settings or get_settings()
    limit = top_k or active.rerank_top_k
    reranker = _get_reranker(active.reranker_model)

    documents = [candidate.payload.text for candidate in candidates]
    scores = list(reranker.rerank(query, documents))

    ranked = sorted(
        zip(candidates, scores, strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [
        candidate.model_copy(update={"score": float(score)})
        for candidate, score in ranked[:limit]
    ]
