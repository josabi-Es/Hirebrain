"""Index CV chunks from JSON artifacts into Qdrant."""

from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient

from rag.ingest.pipeline import load_chunks_from_directory, load_chunks_from_json
from rag.retrieval.embeddings import DenseEmbedder, SparseEmbedder
from rag.retrieval.qdrant_payload import lexical_text_for_chunk, prepare_point_drafts
from rag.retrieval.vector_store import (
    ensure_collection,
    get_qdrant_client,
    prepare_hybrid_upsert_batch_from_drafts,
    upsert_chunks,
)
from rag.shared.models import CVChunk
from rag.shared.settings import RagSettings, get_settings


def index_chunks(
    chunks: list[CVChunk],
    *,
    client: QdrantClient | None = None,
    settings: RagSettings | None = None,
    contextualize: bool = False,
) -> int:
    """
    Embed and upsert CV chunks into Qdrant.

    Returns the number of points indexed.
    """
    if not chunks:
        return 0

    active = settings or get_settings()
    qdrant = client or get_qdrant_client(active)
    ensure_collection(qdrant, active)

    drafts = prepare_point_drafts(chunks, contextualize=contextualize)
    dense_embedder = DenseEmbedder(active)
    sparse_embedder = SparseEmbedder(active)

    dense_texts = [draft.embed_text for draft in drafts]
    sparse_texts = [lexical_text_for_chunk(chunk) for chunk in chunks]

    dense_vectors = dense_embedder.embed_batch(dense_texts)
    sparse_vectors = sparse_embedder.embed_batch(sparse_texts)

    batch = prepare_hybrid_upsert_batch_from_drafts(
        drafts,
        dense_vectors,
        sparse_vectors,
    )
    upsert_chunks(batch, client=qdrant, settings=active)
    return len(batch.points)


def index_chunks_from_json(
    path: Path,
    *,
    client: QdrantClient | None = None,
    settings: RagSettings | None = None,
) -> int:
    """Load one chunks JSON file and index into Qdrant."""
    chunks = load_chunks_from_json(path)
    return index_chunks(chunks, client=client, settings=settings)


def index_chunks_from_directory(
    directory: Path,
    *,
    client: QdrantClient | None = None,
    settings: RagSettings | None = None,
) -> int:
    """Load all *.chunks.json files in a directory and index into Qdrant."""
    chunks = load_chunks_from_directory(directory)
    return index_chunks(chunks, client=client, settings=settings)
