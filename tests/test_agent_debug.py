from __future__ import annotations

from rag.agent.debug import format_agent_debug, search_mode_value
from rag.shared.schemas import SearchMode


def test_search_mode_value_with_enum() -> None:
    assert search_mode_value({"search_mode": SearchMode.PROFILE}) == "profile"


def test_search_mode_value_with_string() -> None:
    assert search_mode_value({"search_mode": "hybrid"}) == "hybrid"


def test_search_mode_value_missing() -> None:
    assert search_mode_value({}) == "unknown"


def test_format_agent_debug_includes_routing_and_sources() -> None:
    state = {
        "thread_id": "thread-a",
        "search_mode": SearchMode.PROFILE,
        "in_domain": True,
        "route_reason": "profile of one candidate",
        "rewritten_query": "summarize his profile",
        "candidate_name": "Austin Gutierrez",
        "doc_id": "cv_austin_gutierrez",
        "messages": [
            {"role": "user", "content": "Which candidate graduated from Daniel University?"},
            {"role": "assistant", "content": "Austin Gutierrez graduated from Daniel University."},
            {"role": "user", "content": "summarize his profile"},
            {"role": "assistant", "content": "Summary here."},
        ],
        "sources": [
            {
                "candidate_name": "Austin Gutierrez",
                "doc_id": "cv_austin_gutierrez",
                "section": "EDUCATION",
                "chunk_id": "cv_austin_gutierrez_education_item_0",
            }
        ],
    }

    text = format_agent_debug(state)

    assert "thread_id=thread-a" in text
    assert "mode=profile" in text
    assert "in_domain=True" in text
    assert "reason=profile of one candidate" in text
    assert "rewritten_query=summarize his profile" in text
    assert "candidate_name=Austin Gutierrez" in text
    assert "doc_id=cv_austin_gutierrez" in text
    assert "history_turns=4" in text
    assert "--- Sources ---" in text
    assert (
        "- Austin Gutierrez | cv_austin_gutierrez | EDUCATION | "
        "cv_austin_gutierrez_education_item_0"
    ) in text


def test_format_agent_debug_omits_sources_when_empty() -> None:
    state = {
        "thread_id": "thread-b",
        "search_mode": SearchMode.HYBRID,
        "in_domain": True,
        "route_reason": "skill search",
        "messages": [],
        "sources": [],
    }

    text = format_agent_debug(state)

    assert "--- Sources ---" not in text
    assert "history_turns=0" in text
