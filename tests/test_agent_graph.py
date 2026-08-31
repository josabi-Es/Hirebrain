from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("langgraph")

from rag.agent.graph import build_graph, run_agent  # noqa: E402
from rag.shared.schemas import QdrantChunkPayload, RetrievedChunk, SearchMode  # noqa: E402


def _chunk() -> RetrievedChunk:
    payload = QdrantChunkPayload(
        chunk_id="cv_a_skills_0",
        doc_id="cv_a",
        candidate_name="Alice Example",
        section="SKILLS",
        chunk_type="skills_list",
        chunk_index=0,
        text="Python, FastAPI, and data engineering.",
    )
    return RetrievedChunk(chunk_id=payload.chunk_id, score=0.92, payload=payload)


def setup_function() -> None:
    build_graph.cache_clear()


def test_graph_generates_answer_from_retrieved_chunks() -> None:
    model = MagicMock()
    model.invoke.return_value = SimpleNamespace(content="Alice knows Python.")

    with (
        patch(
            "rag.agent.nodes.chat_json",
            return_value={
                "in_domain": True,
                "mode": "hybrid",
                "candidate_name": None,
                "keyword": None,
                "doc_id": None,
                "filters": {},
                "reason": "skill search",
            },
        ),
        patch("rag.agent.nodes.chat", return_value="Python FastAPI"),
        patch("rag.agent.nodes.get_answer_chat_model", return_value=model),
        patch("rag.agent.nodes.retrieve", return_value=[_chunk()]) as retrieve_mock,
    ):
        state = run_agent("Who has Python and FastAPI experience?")

    assert state["answer"] == "Alice knows Python."
    assert state["search_mode"] == SearchMode.HYBRID
    assert state["rewritten_query"] == "Python FastAPI"
    assert state["sources"][0]["doc_id"] == "cv_a"
    retrieve_mock.assert_called_once()


def test_graph_rejects_out_of_domain_query_without_retrieval() -> None:
    with (
        patch(
            "rag.agent.nodes.chat_json",
            return_value={
                "in_domain": False,
                "mode": "hybrid",
                "candidate_name": None,
                "keyword": None,
                "doc_id": None,
                "filters": {},
                "reason": "unrelated topic",
            },
        ),
        patch("rag.agent.nodes.retrieve") as retrieve_mock,
    ):
        state = run_agent("Tell me a joke about cats")

    assert state["in_domain"] is False
    assert "indexed CV corpus" in state["answer"]
    retrieve_mock.assert_not_called()


def test_graph_returns_no_evidence_when_retrieval_is_empty() -> None:
    with (
        patch(
            "rag.agent.nodes.chat_json",
            return_value={
                "in_domain": True,
                "mode": "lexical",
                "candidate_name": None,
                "keyword": "Unknown University",
                "doc_id": None,
                "filters": {"section": "EDUCATION"},
                "reason": "education keyword",
            },
        ),
        patch("rag.agent.nodes.chat", return_value="Unknown University"),
        patch("rag.agent.nodes.retrieve", return_value=[]),
    ):
        state = run_agent("Which candidate graduated from Unknown University?")

    assert state["search_mode"] == SearchMode.LEXICAL
    assert "not have enough evidence" in state["answer"]
    assert state["sources"] == []


def test_graph_persists_messages_by_thread_id() -> None:
    router_decision = {
        "in_domain": True,
        "mode": "hybrid",
        "candidate_name": None,
        "keyword": None,
        "doc_id": None,
        "filters": {},
        "reason": "skill search",
    }
    model = MagicMock()
    model.invoke.side_effect = [
        SimpleNamespace(content="First answer."),
        SimpleNamespace(content="Second answer."),
        SimpleNamespace(content="Other thread answer."),
    ]

    with (
        patch("rag.agent.nodes.chat_json", return_value=router_decision),
        patch(
            "rag.agent.nodes.chat",
            side_effect=[
                "Python",
                "FastAPI",
                "Data",
            ],
        ),
        patch("rag.agent.nodes.get_answer_chat_model", return_value=model),
        patch("rag.agent.nodes.retrieve", return_value=[_chunk()]),
    ):
        first = run_agent("Who knows Python?", thread_id="thread-a")
        second = run_agent("Who knows FastAPI?", thread_id="thread-a")
        other = run_agent("Who knows data engineering?", thread_id="thread-b")

    assert first["thread_id"] == "thread-a"
    assert [message["content"] for message in second["messages"]] == [
        "Who knows Python?",
        "First answer.",
        "Who knows FastAPI?",
        "Second answer.",
    ]
    assert [message["content"] for message in other["messages"]] == [
        "Who knows data engineering?",
        "Other thread answer.",
    ]
