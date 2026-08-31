"""Vector store adapter (Qdrant) — collection, upsert, and search by mode."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from rag.retrieval.embeddings import sparse_to_qdrant
from rag.retrieval.filters import build_query_filter
from rag.retrieval.search_context import build_search_context
from rag.retrieval.qdrant_hits import hit_to_retrieved
from rag.retrieval.qdrant_payload import (
    attach_hybrid_vectors,
    build_hybrid_upsert_batch,
)
from rag.shared.models import CVChunk
from rag.shared.schemas import (
    QdrantPointDraft,
    QdrantUpsertBatch,
    RetrievedChunk,
    SparseVectorData,
)
from rag.shared.settings import (
    RagSettings,
    get_settings,
)

_REQUIRED_PAYLOAD_KEYS = frozenset(
    {
        "chunk_id",
        "doc_id",
        "candidate_name",
        "section",
        "chunk_type",
        "chunk_index",
        "text",
    }
)

_PAYLOAD_INDEX_FIELDS = (
    "candidate_name",
    "doc_id",
    "section",
    "chunk_type",
    "institution",
    "company",
    "degree",
)


def get_qdrant_client(settings: RagSettings | None = None) -> QdrantClient:
    """Create a Qdrant client from settings."""
    active = settings or get_settings()
    return QdrantClient(url=active.qdrant_url)


def embedding_text_for_chunk(chunk: CVChunk) -> str:
    """Text to send to the embedding model (never mutates chunk.text)."""
    return chunk.embedding_text()


def validate_point_draft(point: QdrantPointDraft, *, hybrid: bool = True) -> None:
    """Validate a point draft before upsert."""
    if not point.chunk_id.strip():
        raise ValueError("chunk_id must be non-empty")
    if not point.embed_text.strip():
        raise ValueError(f"embed_text must be non-empty for {point.chunk_id!r}")
    if not point.payload.text.strip():
        raise ValueError(f"payload.text must be non-empty for {point.chunk_id!r}")
    payload_keys = set(point.payload.model_dump().keys())
    missing = _REQUIRED_PAYLOAD_KEYS - payload_keys
    if missing:
        raise ValueError(f"payload missing required keys: {sorted(missing)}")

    if point.dense_vector is None:
        raise ValueError(f"Dense vector is required for {point.chunk_id!r}")
    if hybrid and point.sparse_vector is None:
        raise ValueError(f"sparse_vector is required for hybrid upsert ({point.chunk_id!r})")
    if not point.qdrant_point_id.strip():
        raise ValueError(f"qdrant_point_id must be non-empty for {point.chunk_id!r}")


def validate_upsert_batch(batch: QdrantUpsertBatch, *, hybrid: bool = True) -> None:
    """Validate all points in a batch."""
    if not batch.points:
        raise ValueError("Upsert batch must contain at least one point")
    seen_ids: set[str] = set()
    for point in batch.points:
        validate_point_draft(point, hybrid=hybrid)
        if point.chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk_id in batch: {point.chunk_id!r}")
        seen_ids.add(point.chunk_id)


def ensure_collection(
    client: QdrantClient,
    settings: RagSettings | None = None,
) -> None:
    """Create hybrid Qdrant collection and payload indexes if missing."""
    active = settings or get_settings()
    if client.collection_exists(active.qdrant_collection):
        return

    client.create_collection(
        collection_name=active.qdrant_collection,
        vectors_config={
            active.dense_vector_name: qmodels.VectorParams(
                size=active.dense_vector_size,
                distance=qmodels.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            active.sparse_vector_name: qmodels.SparseVectorParams()
        },
    )

    for field in _PAYLOAD_INDEX_FIELDS:
        client.create_payload_index(
            collection_name=active.qdrant_collection,
            field_name=field,
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )


def _point_struct_from_draft(
    point: QdrantPointDraft,
    settings: RagSettings,
) -> qmodels.PointStruct:
    if point.dense_vector is None or point.sparse_vector is None:
        raise ValueError(f"Hybrid vectors required for upsert ({point.chunk_id!r})")

    return qmodels.PointStruct(
        id=point.qdrant_point_id,
        vector={
            settings.dense_vector_name: point.dense_vector,
            settings.sparse_vector_name: qmodels.SparseVector(
                **sparse_to_qdrant(point.sparse_vector)
            ),
        },
        payload=point.payload.model_dump(),
    )



def prepare_hybrid_upsert_batch_from_drafts(
    drafts: Sequence[QdrantPointDraft],
    dense_vectors: Sequence[Sequence[float]],
    sparse_vectors: Sequence[SparseVectorData],
) -> QdrantUpsertBatch:
    """Attach hybrid vectors to existing drafts and validate the batch."""
    points = attach_hybrid_vectors(drafts, dense_vectors, sparse_vectors)
    batch = QdrantUpsertBatch(points=points)
    validate_upsert_batch(batch, hybrid=True)
    return batch


def prepare_hybrid_upsert_batch(
    chunks: Iterable[CVChunk],
    dense_vectors: Sequence[Sequence[float]],
    sparse_vectors: Sequence[SparseVectorData],
    *,
    contextualize: bool = True,
) -> QdrantUpsertBatch:
    """Build and validate a hybrid batch ready for Qdrant upsert."""
    batch = build_hybrid_upsert_batch(
        chunks,
        dense_vectors,
        sparse_vectors,
        contextualize=contextualize,
    )
    validate_upsert_batch(batch, hybrid=True)
    return batch


def upsert_chunks(
    batch: QdrantUpsertBatch | Iterable[QdrantPointDraft],
    *,
    client: QdrantClient | None = None,
    collection_name: str | None = None,
    settings: RagSettings | None = None,
) -> None:
    """Upsert validated hybrid points into Qdrant."""
    active = settings or get_settings()
    if isinstance(batch, QdrantUpsertBatch):
        points = batch.points
    else:
        points = list(batch)

    upsert_batch = QdrantUpsertBatch(points=points)
    validate_upsert_batch(upsert_batch, hybrid=True)

    if client is None:
        raise NotImplementedError(
            "Qdrant client not configured; use prepare_hybrid_upsert_batch() and pass a "
            "QdrantClient to upsert_chunks()"
        )

    target_collection = collection_name or active.qdrant_collection
    if not target_collection:
        raise ValueError("collection_name is required for upsert")

    qdrant_points = [
        _point_struct_from_draft(point, active) for point in upsert_batch.points
    ]
    client.upsert(collection_name=target_collection, points=qdrant_points, wait=True)


def resolve_candidate_name(
    client: QdrantClient,
    candidate_name: str,
    *,
    settings: RagSettings | None = None,
) -> str:
    """Resolve a partial/lowercase candidate name to its canonical payload value.

    Qdrant keyword filters match exactly, so "timothy" never matches the stored
    "Timothy Thompson". This scans known candidate names and resolves the query to
    a canonical name via case-insensitive, token-subset, or substring matching.
    """
    active = settings or get_settings()
    target = " ".join(candidate_name.lower().split())
    if not target:
        return candidate_name

    records, _ = client.scroll(
        collection_name=active.qdrant_collection,
        with_payload=["candidate_name"],
        with_vectors=False,
        limit=10000,
    )
    names = {
        str(record.payload.get("candidate_name", "")).strip()
        for record in records
        if record.payload
    }
    names.discard("")

    for name in names:
        if name.lower() == target:
            return name

    target_tokens = set(target.split())
    token_matches = sorted(
        name for name in names if target_tokens <= set(name.lower().split())
    )
    if token_matches:
        return token_matches[0]

    substring_matches = sorted(name for name in names if target in name.lower())
    if substring_matches:
        return substring_matches[0]

    return candidate_name


def scroll_by_candidate(
    client: QdrantClient,
    *,
    candidate_name: str | None = None,
    doc_id: str | None = None,
    sections: Sequence[str] | None = None,
    settings: RagSettings | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve all chunks for a candidate profile without vector search (Case 1).

    Returns every indexed chunk for the candidate across all CV sections. A section
    filter is applied only when ``sections`` is explicitly provided.
    """
    active = settings or get_settings()
    if not candidate_name and not doc_id:
        raise ValueError("candidate_name or doc_id is required for profile scroll")

    filter_parts: dict[str, str] = {}
    if doc_id:
        filter_parts["doc_id"] = doc_id
    elif candidate_name:
        filter_parts["candidate_name"] = resolve_candidate_name(
            client, candidate_name, settings=active
        )

    query_filter = build_query_filter(
        filter_parts,
        sections=list(sections) if sections else None,
    )

    records, _ = client.scroll(
        collection_name=active.qdrant_collection,
        scroll_filter=query_filter,
        limit=active.profile_scroll_limit,
        with_payload=True,
        with_vectors=False,
    )

    retrieved = [hit_to_retrieved(record) for record in records]
    retrieved.sort(key=lambda item: (item.payload.section, item.payload.chunk_index))
    return retrieved


def similarity_search(
    client: QdrantClient,
    query: str,
    *,
    query_vector: Sequence[float] | None = None,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    settings: RagSettings | None = None,
) -> list[RetrievedChunk]:
    """Dense vector search only (semantic fallback)."""
    from rag.retrieval.embeddings import DenseEmbedder

    ctx = build_search_context(
        filters=filters,
        top_k=top_k,
        client=client,
        settings=settings,
    )
    dense = (
        list(query_vector)
        if query_vector is not None
        else DenseEmbedder(ctx.settings).embed(query)
    )

    response = ctx.client.query_points(
        collection_name=ctx.settings.qdrant_collection,
        query=dense,
        using=ctx.settings.dense_vector_name,
        query_filter=ctx.query_filter,
        limit=ctx.limit,
        with_payload=True,
    )
    return [hit_to_retrieved(point) for point in response.points]


__all__ = [
    "embedding_text_for_chunk",
    "ensure_collection",
    "get_qdrant_client",
    "prepare_hybrid_upsert_batch",
    "prepare_hybrid_upsert_batch_from_drafts",
    "resolve_candidate_name",
    "scroll_by_candidate",
    "similarity_search",
    "upsert_chunks",
    "validate_point_draft",
    "validate_upsert_batch",
]
