from __future__ import annotations

import pytest

from rag.ingest.pipeline import _chunk_to_dict, contextualize_chunks
from rag.ingest.contextualizer import ChunkContextualizer
from rag.retrieval.qdrant_payload import (
    attach_hybrid_vectors,
    build_qdrant_payload,
    build_qdrant_point,
    prepare_point_drafts,
)
from rag.retrieval.vector_store import (
    embedding_text_for_chunk,
    prepare_hybrid_upsert_batch,
    upsert_chunks,
    validate_upsert_batch,
)
from rag.shared.models import CVChunk
from rag.shared.schemas import QdrantPointDraft, SparseVectorData


def _experience_chunk() -> CVChunk:
    return CVChunk(
        chunk_id="cv_test_experience_item_0",
        doc_id="cv_test",
        candidate_name="Jane Doe",
        section="EXPERIENCE",
        chunk_type="experience_item",
        chunk_index=0,
        text=(
            "Senior Software Engineer\n"
            "Acme Corp\n"
            "01/2020 – Present\n"
            "Built Python services."
        ),
        job_title="Senior Software Engineer",
        company="Acme Corp",
        date_range="01/2020 – Present",
    )


def _skills_chunk() -> CVChunk:
    return CVChunk(
        chunk_id="cv_test_skills_list_0",
        doc_id="cv_test",
        candidate_name="Jane Doe",
        section="SKILLS",
        chunk_type="skills_list",
        chunk_index=0,
        text="Python\nGo\nKubernetes",
    )


def test_build_text_to_embed_includes_metadata_and_original_text() -> None:
    chunk = _experience_chunk()
    original_text = chunk.text
    embed = ChunkContextualizer().build_text_to_embed(chunk)

    assert chunk.text == original_text
    assert "Document: CV" not in embed
    assert "Candidate: Jane Doe" in embed
    assert "Section: EXPERIENCE" in embed
    assert "Role: Senior Software Engineer" in embed
    assert "Company: Acme Corp" in embed
    assert "Period: 01/2020 – Present" in embed
    assert embed.endswith(f"Text: {original_text}")
    assert "Context:" in embed


def test_build_text_to_embed_omits_null_optional_fields() -> None:
    chunk = _skills_chunk()
    embed = ChunkContextualizer().build_text_to_embed(chunk)

    assert "Role:" not in embed
    assert "Company:" not in embed
    assert "Degree:" not in embed
    assert "Institution:" not in embed
    assert "Period:" not in embed
    assert "Candidate: Jane Doe" in embed


def test_contextualize_chunk_preserves_text_and_sets_text_to_embed() -> None:
    chunk = _experience_chunk()
    original_text = chunk.text
    augmented = ChunkContextualizer().contextualize_chunk(chunk)

    assert augmented.text == original_text
    assert augmented.text_to_embed is not None
    assert augmented.text_to_embed != original_text
    assert original_text in augmented.text_to_embed


def test_contextualize_chunks_batch() -> None:
    chunks = [_experience_chunk(), _skills_chunk()]
    result = contextualize_chunks(chunks)
    assert len(result) == 2
    assert all(c.text_to_embed for c in result)


def test_build_qdrant_payload_consolidated_no_vector_metadata_key() -> None:
    chunk = _experience_chunk()
    payload = build_qdrant_payload(chunk)

    assert payload.text == chunk.text
    assert payload.doc_id == "cv_test"
    assert payload.company == "Acme Corp"
    assert payload.chunk_id == chunk.chunk_id
    dumped = payload.model_dump()
    assert "vector_metadata" not in dumped


def _sparse_vector() -> SparseVectorData:
    return SparseVectorData(indices=[0, 1], values=[0.5, 0.7])


def test_build_qdrant_point_uses_embed_text_and_payload_text() -> None:
    chunk = _experience_chunk()
    vector = [0.1, 0.2, 0.3]
    point = build_qdrant_point(chunk, vector, sparse_vector=_sparse_vector())

    assert point.chunk_id == chunk.chunk_id
    assert point.dense_vector == vector
    assert point.payload.text == chunk.text
    assert "Document: CV" not in point.embed_text
    assert point.embed_text != chunk.text
    assert point.qdrant_point_id


def test_prepare_hybrid_upsert_batch_validates_vectors() -> None:
    chunks = [_experience_chunk(), _skills_chunk()]
    dense_vectors = [[0.1, 0.2], [0.3, 0.4]]
    sparse_vectors = [
        SparseVectorData(indices=[0], values=[1.0]),
        SparseVectorData(indices=[1], values=[0.5]),
    ]
    batch = prepare_hybrid_upsert_batch(
        chunks,
        dense_vectors,
        sparse_vectors,
        contextualize=False,
    )

    validate_upsert_batch(batch, hybrid=True)
    assert len(batch.points) == 2
    for point, vector in zip(batch.points, dense_vectors, strict=True):
        assert point.dense_vector == vector
        assert point.payload.text in (chunks[0].text, chunks[1].text)


def test_embedding_text_for_chunk_prefers_text_to_embed() -> None:
    chunk = ChunkContextualizer().contextualize_chunk(_experience_chunk())
    assert embedding_text_for_chunk(chunk) == chunk.text_to_embed
    assert embedding_text_for_chunk(_experience_chunk()) == _experience_chunk().text


def test_attach_hybrid_vectors_length_mismatch_raises() -> None:
    drafts = prepare_point_drafts([_experience_chunk()])
    with pytest.raises(ValueError, match="must match"):
        attach_hybrid_vectors(drafts, [], [])


def test_upsert_without_client_raises_not_implemented() -> None:
    batch = prepare_hybrid_upsert_batch(
        [_experience_chunk()],
        [[0.1, 0.2]],
        [_sparse_vector()],
        contextualize=False,
    )
    with pytest.raises(NotImplementedError, match="Qdrant client"):
        upsert_chunks(batch)


def test_chunk_to_dict_exports_text_to_embed_without_vector_metadata() -> None:
    chunk = ChunkContextualizer().contextualize_chunk(_experience_chunk())
    exported = _chunk_to_dict(chunk)

    assert "vector_metadata" not in exported
    assert exported["text"] == chunk.text
    assert exported["text_to_embed"] == chunk.text_to_embed


def test_empty_chunk_text_raises() -> None:
    chunk = _experience_chunk().model_copy(update={"text": "   "})
    with pytest.raises(ValueError, match="empty text"):
        ChunkContextualizer().build_text_to_embed(chunk)


def test_build_qdrant_point_to_upsert_dict() -> None:
    point = build_qdrant_point(_experience_chunk(), [0.5, 0.6], sparse_vector=_sparse_vector())
    upsert = point.to_upsert_dict()
    assert upsert["id"] == point.qdrant_point_id
    assert upsert["vector"]["dense"] == [0.5, 0.6]
    assert upsert["payload"]["text"] == _experience_chunk().text


def test_point_draft_without_vector_cannot_upsert_dict() -> None:
    draft = QdrantPointDraft(
        chunk_id="x",
        embed_text="embed",
        payload=build_qdrant_payload(_experience_chunk()),
    )
    with pytest.raises(ValueError, match="dense_vector is required"):
        draft.to_upsert_dict()
