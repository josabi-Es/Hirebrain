"""API and cross-layer request/response schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SearchMode(str, Enum):
    """Retrieval strategy selected by the LangGraph intent detector."""

    PROFILE = "profile"
    LEXICAL = "lexical"
    HYBRID = "hybrid"


class RouterDecision(BaseModel):
    """Structured decision emitted by the LangGraph router node."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    in_domain: bool
    mode: SearchMode = SearchMode.HYBRID
    candidate_name: str | None = None
    keyword: str | None = None
    doc_id: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class QdrantChunkPayload(BaseModel):
    """Consolidated Qdrant payload for filtering, lexical search, and LLM citations."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    chunk_id: str
    doc_id: str
    candidate_name: str
    section: str
    chunk_type: str
    chunk_index: int
    text: str
    job_title: str = ""
    company: str = ""
    date_range: str = ""
    degree: str = ""
    institution: str = ""


class SparseVectorData(BaseModel):
    """Sparse vector components for Qdrant hybrid indexing."""

    model_config = ConfigDict(extra="forbid")

    indices: list[int] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)


class QdrantPointDraft(BaseModel):
    """Point ready for embedding + upsert into Qdrant."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    embed_text: str
    payload: QdrantChunkPayload
    qdrant_point_id: str = ""
    dense_vector: list[float] | None = None
    sparse_vector: SparseVectorData | None = None

    def to_upsert_dict(self) -> dict[str, Any]:
        """Serialize to Qdrant upsert shape: id, vector, payload."""
        if self.dense_vector is None:
            raise ValueError(
                f"dense_vector is required for upsert (chunk_id={self.chunk_id!r})"
            )
        if self.sparse_vector is None:
            raise ValueError(
                f"sparse_vector is required for hybrid upsert (chunk_id={self.chunk_id!r})"
            )
        point_id = self.qdrant_point_id or self.chunk_id
        return {
            "id": point_id,
            "vector": {
                "dense": self.dense_vector,
                "sparse": {
                    "indices": self.sparse_vector.indices,
                    "values": self.sparse_vector.values,
                },
            },
            "payload": self.payload.model_dump(),
        }


class QdrantUpsertBatch(BaseModel):
    """Batch of points prepared for vector_store.upsert_chunks."""

    model_config = ConfigDict(extra="forbid")

    points: list[QdrantPointDraft] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    """Single chunk returned by a retrieval query."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    score: float
    payload: QdrantChunkPayload
