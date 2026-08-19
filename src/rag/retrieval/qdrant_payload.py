"""Build consolidated Qdrant payloads and upsert-ready point drafts."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence

from rag.ingest.contextualizer import ChunkContextualizer
from rag.shared.models import CVChunk
from rag.shared.schemas import (
    QdrantChunkPayload,
    QdrantPointDraft,
    QdrantUpsertBatch,
    SparseVectorData,
)

_QDRANT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def qdrant_point_id(chunk_id: str) -> str:
    """Deterministic UUID for Qdrant point id from chunk_id."""
    return str(uuid.uuid5(_QDRANT_NAMESPACE, chunk_id))


def lexical_text_for_chunk(chunk: CVChunk) -> str:
    """Text for sparse BM25 indexing (keywords in metadata + body)."""
    parts = [
        chunk.institution or "",
        chunk.degree or "",
        chunk.company or "",
        chunk.job_title or "",
        chunk.text,
    ]
    return " ".join(part.strip() for part in parts if part and part.strip())


def build_qdrant_payload(chunk: CVChunk) -> QdrantChunkPayload:
    """
    Consolidated payload for Qdrant: filterable metadata + original text.

    Does not duplicate nested vector_metadata; optional fields use empty strings.
    """
    if not isinstance(chunk, CVChunk):
        raise TypeError(f"Expected CVChunk, got {type(chunk).__name__}")
    if not chunk.text.strip():
        raise ValueError(f"Chunk {chunk.chunk_id!r} has empty text")

    meta = chunk.to_vector_metadata()
    return QdrantChunkPayload(
        chunk_id=str(meta["chunk_id"]),
        doc_id=str(meta["doc_id"]),
        candidate_name=str(meta["candidate_name"]),
        section=str(meta["section"]),
        chunk_type=str(meta["chunk_type"]),
        chunk_index=int(meta["chunk_index"]),
        text=chunk.text,
        job_title=str(meta["job_title"]),
        company=str(meta["company"]),
        date_range=str(meta["date_range"]),
        degree=str(meta["degree"]),
        institution=str(meta["institution"]),
    )


def build_qdrant_point(
    chunk: CVChunk,
    vector: Sequence[float],
    *,
    text_to_embed: str | None = None,
    contextualizer: ChunkContextualizer | None = None,
    sparse_vector: SparseVectorData | None = None,
) -> QdrantPointDraft:
    """
    Build a Qdrant upsert point: chunk_id, vector, consolidated payload.

    ``embed_text`` comes from text_to_embed argument, chunk.text_to_embed, or
    contextualizer output. Payload always carries original ``chunk.text``.
    """
    if not vector:
        raise ValueError(f"Vector must be non-empty for chunk {chunk.chunk_id!r}")

    embed_text = text_to_embed or chunk.text_to_embed
    if not embed_text or not embed_text.strip():
        active = contextualizer or ChunkContextualizer()
        embed_text = active.build_text_to_embed(chunk)

    dense = list(vector)
    return QdrantPointDraft(
        chunk_id=chunk.chunk_id,
        qdrant_point_id=qdrant_point_id(chunk.chunk_id),
        embed_text=embed_text,
        payload=build_qdrant_payload(chunk),
        dense_vector=dense,
        sparse_vector=sparse_vector,
    )


def prepare_point_drafts(
    chunks: Iterable[CVChunk],
    *,
    contextualize: bool = True,
    contextualizer: ChunkContextualizer | None = None,
) -> list[QdrantPointDraft]:
    """
    Prepare point drafts without vectors (for embed-then-upsert pipelines).

    Vectors are attached later via attach_hybrid_vectors.
    """
    active = contextualizer or ChunkContextualizer()
    prepared: list[CVChunk] = (
        active.contextualize_chunks(list(chunks))
        if contextualize
        else list(chunks)
    )
    return [
        QdrantPointDraft(
            chunk_id=chunk.chunk_id,
            qdrant_point_id=qdrant_point_id(chunk.chunk_id),
            embed_text=chunk.embedding_text(),
            payload=build_qdrant_payload(chunk),
            dense_vector=None,
            sparse_vector=None,
        )
        for chunk in prepared
    ]


def attach_hybrid_vectors(
    drafts: Sequence[QdrantPointDraft],
    dense_vectors: Sequence[Sequence[float]],
    sparse_vectors: Sequence[SparseVectorData],
) -> list[QdrantPointDraft]:
    """Attach dense and sparse vectors to drafts in order."""
    draft_list = list(drafts)
    if len(draft_list) != len(dense_vectors):
        raise ValueError(
            f"Draft count ({len(draft_list)}) must match dense vector count "
            f"({len(dense_vectors)})"
        )
    if len(draft_list) != len(sparse_vectors):
        raise ValueError(
            f"Draft count ({len(draft_list)}) must match sparse vector count "
            f"({len(sparse_vectors)})"
        )
    result: list[QdrantPointDraft] = []
    for draft, dense, sparse in zip(
        draft_list, dense_vectors, sparse_vectors, strict=True
    ):
        if not dense:
            raise ValueError(f"Empty dense vector for chunk {draft.chunk_id!r}")
        if not sparse.indices:
            raise ValueError(f"Empty sparse vector for chunk {draft.chunk_id!r}")
        dense_list = list(dense)
        result.append(
            draft.model_copy(
                update={
                    "dense_vector": dense_list,
                    "sparse_vector": sparse,
                }
            )
        )
    return result


def build_hybrid_upsert_batch(
    chunks: Iterable[CVChunk],
    dense_vectors: Sequence[Sequence[float]],
    sparse_vectors: Sequence[SparseVectorData],
    *,
    contextualize: bool = True,
    contextualizer: ChunkContextualizer | None = None,
) -> QdrantUpsertBatch:
    """Build a hybrid upsert batch with dense + sparse vectors."""
    drafts = prepare_point_drafts(
        chunks, contextualize=contextualize, contextualizer=contextualizer
    )
    points = attach_hybrid_vectors(drafts, dense_vectors, sparse_vectors)
    return QdrantUpsertBatch(points=points)
