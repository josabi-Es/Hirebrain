from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag.retrieval.qdrant_payload import (
    attach_hybrid_vectors,
    lexical_text_for_chunk,
    prepare_point_drafts,
    qdrant_point_id,
)
from rag.retrieval.reranker import rerank
from rag.shared.models import CVChunk
from rag.shared.schemas import QdrantChunkPayload, RetrievedChunk, SparseVectorData
from rag.shared.settings import RagSettings


def _education_chunk() -> CVChunk:
    return CVChunk(
        chunk_id="cv_test_education_item_0",
        doc_id="cv_test",
        candidate_name="Jane Doe",
        section="EDUCATION",
        chunk_type="education_item",
        chunk_index=0,
        text="BSc Computer Science\nUniversitat Politecnica de Catalunya (UPC)\n2018 – 2022",
        degree="BSc Computer Science",
        institution="Universitat Politecnica de Catalunya (UPC)",
    )


def test_qdrant_point_id_is_deterministic() -> None:
    first = qdrant_point_id("cv_test_experience_item_0")
    second = qdrant_point_id("cv_test_experience_item_0")
    assert first == second
    assert first != "cv_test_experience_item_0"


def test_lexical_text_for_chunk_includes_institution_and_degree() -> None:
    chunk = _education_chunk()
    lexical = lexical_text_for_chunk(chunk)
    assert "UPC" in lexical
    assert "BSc Computer Science" in lexical
    assert chunk.text in lexical


def test_attach_hybrid_vectors_sets_dense_and_sparse() -> None:
    drafts = prepare_point_drafts([_education_chunk()], contextualize=False)
    dense = [[0.1, 0.2, 0.3]]
    sparse = [SparseVectorData(indices=[1, 2], values=[0.5, 0.8])]
    points = attach_hybrid_vectors(drafts, dense, sparse)
    assert points[0].dense_vector == [0.1, 0.2, 0.3]
    assert points[0].sparse_vector == sparse[0]
    assert points[0].qdrant_point_id


def test_validate_hybrid_upsert_batch_requires_sparse() -> None:
    from rag.retrieval.vector_store import prepare_hybrid_upsert_batch, validate_upsert_batch

    batch = prepare_hybrid_upsert_batch(
        [_education_chunk()],
        [[0.1, 0.2]],
        [SparseVectorData(indices=[0], values=[1.0])],
        contextualize=False,
    )
    validate_upsert_batch(batch, hybrid=True)
    assert len(batch.points) == 1


def test_rerank_orders_by_score_and_truncates_top_k() -> None:
    payload_a = QdrantChunkPayload(
        chunk_id="a",
        doc_id="cv_a",
        candidate_name="A",
        section="SKILLS",
        chunk_type="skills_list",
        chunk_index=0,
        text="Python expert",
    )
    payload_b = QdrantChunkPayload(
        chunk_id="b",
        doc_id="cv_b",
        candidate_name="B",
        section="SKILLS",
        chunk_type="skills_list",
        chunk_index=0,
        text="Java expert",
    )
    candidates = [
        RetrievedChunk(chunk_id="a", score=0.1, payload=payload_a),
        RetrievedChunk(chunk_id="b", score=0.9, payload=payload_b),
    ]
    settings = RagSettings(
        qdrant_url="http://127.0.0.1:6333",
        qdrant_collection="test",
        reranker_model="mock-reranker",
        rerank_top_k=1,
    )

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [0.2, 0.95]

    with patch("rag.retrieval.reranker._get_reranker", return_value=mock_reranker):
        result = rerank("Python", candidates, top_k=1, settings=settings)

    assert len(result) == 1
    assert result[0].chunk_id == "b"
    assert result[0].score == pytest.approx(0.95)


def test_build_query_filter_with_section() -> None:
    from rag.retrieval.filters import build_query_filter

    query_filter = build_query_filter({"section": "EDUCATION"})
    assert query_filter is not None
    assert query_filter.must is not None


def test_load_chunks_from_json_roundtrip(tmp_path) -> None:
    from rag.ingest.pipeline import load_chunks_from_json

    chunk = _education_chunk()
    path = tmp_path / "cv_test.chunks.json"
    path.write_text(
        '{"doc_id":"cv_test","candidate_name":"Jane Doe","chunks":[{"chunk_id":"cv_test_education_item_0","doc_id":"cv_test","candidate_name":"Jane Doe","section":"EDUCATION","chunk_type":"education_item","chunk_index":0,"text":"UPC","job_title":null,"company":null,"date_range":null,"degree":"BSc","institution":"UPC"}]}',
        encoding="utf-8",
    )
    loaded = load_chunks_from_json(path)
    assert len(loaded) == 1
    assert loaded[0].chunk_id == chunk.chunk_id.replace("_0", "_0")
    assert loaded[0].institution == "UPC"
