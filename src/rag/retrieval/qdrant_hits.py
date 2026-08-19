"""Convert Qdrant query/scroll hits into RetrievedChunk objects."""

from __future__ import annotations

from qdrant_client.http import models as qmodels

from rag.shared.schemas import QdrantChunkPayload, RetrievedChunk


def hit_to_retrieved(point: qmodels.ScoredPoint | qmodels.Record) -> RetrievedChunk:
    """Map a Qdrant scored point or scroll record to RetrievedChunk."""
    payload_raw = point.payload or {}
    payload = QdrantChunkPayload.model_validate(payload_raw)
    score = getattr(point, "score", 0.0) or 0.0
    return RetrievedChunk(
        chunk_id=payload.chunk_id,
        score=float(score),
        payload=payload,
    )
