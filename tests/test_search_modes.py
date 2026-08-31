from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.http import models as qmodels

from rag.retrieval.hybrid_search import search as hybrid_search
from rag.retrieval.lexical_search import search as lexical_search
from rag.retrieval.router import retrieve
from rag.shared.schemas import QdrantChunkPayload, RetrievedChunk, SearchMode
from rag.shared.settings import RagSettings


def _sample_payload(chunk_id: str = "chunk_a") -> QdrantChunkPayload:
    return QdrantChunkPayload(
        chunk_id=chunk_id,
        doc_id="cv_a",
        candidate_name="Alice",
        section="SKILLS",
        chunk_type="skills_list",
        chunk_index=0,
        text="Python expert",
    )


def _sample_hit(chunk_id: str = "chunk_a", score: float = 0.9) -> qmodels.ScoredPoint:
    return qmodels.ScoredPoint(
        id=chunk_id,
        version=1,
        score=score,
        payload=_sample_payload(chunk_id).model_dump(),
    )


def _settings() -> RagSettings:
    return RagSettings(
        qdrant_url="http://127.0.0.1:6333",
        qdrant_collection="test",
        retrieval_top_k=5,
        rerank_top_k=2,
    )


def test_lexical_search_queries_sparse_vector() -> None:
    client = MagicMock()
    client.query_points.return_value = MagicMock(points=[_sample_hit()])

    with patch("rag.retrieval.lexical_search.SparseEmbedder") as embedder_cls:
        embedder_cls.return_value.embed.return_value = MagicMock(
            indices=[1],
            values=[0.8],
        )
        results = lexical_search(
            "Python",
            keyword="Python",
            client=client,
            settings=_settings(),
        )

    assert len(results) == 1
    assert results[0].chunk_id == "chunk_a"
    client.query_points.assert_called_once()
    call_kwargs = client.query_points.call_args.kwargs
    assert call_kwargs["using"] == "sparse"
    assert call_kwargs["limit"] == 5


def test_lexical_search_requires_non_empty_query() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        lexical_search("   ", client=MagicMock(), settings=_settings())


def test_hybrid_search_applies_rrf_and_optional_rerank() -> None:
    client = MagicMock()
    client.query_points.return_value = MagicMock(
        points=[_sample_hit("a", 0.5), _sample_hit("b", 0.4)]
    )
    reranked = [
        RetrievedChunk(chunk_id="b", score=0.99, payload=_sample_payload("b")),
    ]

    with (
        patch("rag.retrieval.hybrid_search.DenseEmbedder") as dense_cls,
        patch("rag.retrieval.hybrid_search.SparseEmbedder") as sparse_cls,
        patch(
            "rag.retrieval.hybrid_search.rerank_candidates",
            return_value=reranked,
        ) as rerank_mock,
    ):
        dense_cls.return_value.embed.return_value = [0.1, 0.2]
        sparse_cls.return_value.embed.return_value = MagicMock(
            indices=[0],
            values=[1.0],
        )
        results = hybrid_search(
            "Python developers",
            client=client,
            settings=_settings(),
        )

    assert results == reranked
    rerank_mock.assert_called_once()
    fusion_query = client.query_points.call_args.kwargs["query"]
    assert isinstance(fusion_query, qmodels.FusionQuery)


def test_hybrid_search_can_skip_rerank() -> None:
    client = MagicMock()
    client.query_points.return_value = MagicMock(points=[_sample_hit()])

    with (
        patch("rag.retrieval.hybrid_search.DenseEmbedder") as dense_cls,
        patch("rag.retrieval.hybrid_search.SparseEmbedder") as sparse_cls,
        patch("rag.retrieval.hybrid_search.rerank_candidates") as rerank_mock,
    ):
        dense_cls.return_value.embed.return_value = [0.1]
        sparse_cls.return_value.embed.return_value = MagicMock(
            indices=[0],
            values=[1.0],
        )
        results = hybrid_search(
            "Python",
            rerank_results=False,
            client=client,
            settings=_settings(),
        )

    rerank_mock.assert_not_called()
    assert len(results) == 1


def test_router_dispatches_by_mode() -> None:
    profile_result = [
        RetrievedChunk(chunk_id="p", score=0.0, payload=_sample_payload("p"))
    ]
    lexical_result = [
        RetrievedChunk(chunk_id="l", score=0.5, payload=_sample_payload("l"))
    ]
    hybrid_result = [
        RetrievedChunk(chunk_id="h", score=0.8, payload=_sample_payload("h"))
    ]
    client = MagicMock()

    with (
        patch(
            "rag.retrieval.router.scroll_by_candidate",
            return_value=profile_result,
        ) as scroll_mock,
        patch(
            "rag.retrieval.router.lexical_search",
            return_value=lexical_result,
        ) as lexical_mock,
        patch(
            "rag.retrieval.router.hybrid_search",
            return_value=hybrid_result,
        ) as hybrid_mock,
    ):
        assert retrieve(
            "ignored",
            mode=SearchMode.PROFILE,
            candidate_name="Alice",
            client=client,
        ) == profile_result
        assert retrieve(
            "Python",
            mode=SearchMode.LEXICAL,
            keyword="Python",
            client=client,
        ) == lexical_result
        assert retrieve(
            "Python experience",
            mode=SearchMode.HYBRID,
            client=client,
        ) == hybrid_result

    scroll_mock.assert_called_once_with(
        client,
        candidate_name="Alice",
        doc_id=None,
        settings=scroll_mock.call_args.kwargs["settings"],
    )
    lexical_mock.assert_called_once()
    hybrid_mock.assert_called_once()


def test_router_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported search mode"):
        retrieve("query", mode="semantic")  # type: ignore[arg-type]
