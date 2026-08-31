"""Retrieval layer: Qdrant payloads, vector store, search modes."""

from rag.retrieval.indexer import (
    index_chunks,
    index_chunks_from_directory,
    index_chunks_from_json,
)
from rag.retrieval.qdrant_payload import (
    attach_hybrid_vectors,
    build_hybrid_upsert_batch,
    build_qdrant_payload,
    build_qdrant_point,
    lexical_text_for_chunk,
    prepare_point_drafts,
    qdrant_point_id,
)
from rag.retrieval.router import retrieve
from rag.retrieval.vector_store import (
    embedding_text_for_chunk,
    ensure_collection,
    get_qdrant_client,
    prepare_hybrid_upsert_batch,
    prepare_hybrid_upsert_batch_from_drafts,
    scroll_by_candidate,
    similarity_search,
    upsert_chunks,
    validate_point_draft,
    validate_upsert_batch,
)
from rag.shared.schemas import RetrievedChunk, SearchMode

__all__ = [
    "RetrievedChunk",
    "SearchMode",
    "attach_hybrid_vectors",
    "build_hybrid_upsert_batch",
    "build_qdrant_payload",
    "build_qdrant_point",
    "embedding_text_for_chunk",
    "ensure_collection",
    "get_qdrant_client",
    "index_chunks",
    "index_chunks_from_directory",
    "index_chunks_from_json",
    "lexical_text_for_chunk",
    "prepare_hybrid_upsert_batch",
    "prepare_hybrid_upsert_batch_from_drafts",
    "prepare_point_drafts",
    "qdrant_point_id",
    "retrieve",
    "scroll_by_candidate",
    "similarity_search",
    "upsert_chunks",
    "validate_point_draft",
    "validate_upsert_batch",
]
