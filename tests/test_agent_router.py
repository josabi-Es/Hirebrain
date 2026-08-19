from __future__ import annotations

from unittest.mock import patch

from rag.agent.nodes import (
    _name_from_history,
    _normalize_decision,
    _recover_profile_target,
    route_after_router,
    router_node,
)
from rag.shared.schemas import RouterDecision, SearchMode


def test_router_node_uses_llm_decision_for_profile_query() -> None:
    with patch(
        "rag.agent.nodes.chat_json",
        return_value={
            "in_domain": True,
            "mode": "profile",
            "candidate_name": "Austin Gutierrez",
            "keyword": None,
            "doc_id": None,
            "filters": {},
            "reason": "candidate profile request",
        },
    ):
        state = router_node({"query": "Summarize Austin Gutierrez profile"})

    assert state["in_domain"] is True
    assert state["search_mode"] == SearchMode.PROFILE
    assert state["candidate_name"] == "Austin Gutierrez"
    assert route_after_router(state) == "rewrite"


def test_router_node_rejects_out_of_domain_query() -> None:
    with patch(
        "rag.agent.nodes.chat_json",
        return_value={
            "in_domain": False,
            "mode": "hybrid",
            "candidate_name": None,
            "keyword": None,
            "doc_id": None,
            "filters": {},
            "reason": "weather is outside CV screening",
        },
    ):
        state = router_node({"query": "What is the weather tomorrow?"})

    assert state["in_domain"] is False
    assert route_after_router(state) == "reject"


def test_router_node_falls_back_to_heuristic_when_llm_fails() -> None:
    with patch("rag.agent.nodes.chat_json", side_effect=RuntimeError("offline")):
        state = router_node(
            {"query": "Which candidate graduated from Martin University?"}
        )

    assert state["in_domain"] is True
    assert state["search_mode"] == SearchMode.LEXICAL
    assert state["keyword"] == "Martin University"
    assert state["filters"] == {
        "section": "EDUCATION",
        "institution": "Martin University",
    }


def test_normalize_decision_recovers_candidate_from_prior_sources() -> None:
    decision = RouterDecision(
        in_domain=True,
        mode=SearchMode.PROFILE,
        candidate_name=None,
        reason="profile of one candidate",
    )
    normalized = _normalize_decision(
        "Summarize his profile",
        decision,
        history=[
            {
                "role": "user",
                "content": "Which candidate graduated from Daniel University?",
            },
            {
                "role": "assistant",
                "content": (
                    "## Education\n\n- **Austin Gutierrez** — graduated from "
                    "Daniel University, Computer Programming program"
                ),
            },
        ],
        sources=[
            {
                "candidate_name": "Austin Gutierrez",
                "doc_id": "cv_austin_gutierrez",
                "section": "EDUCATION",
                "chunk_id": "cv_austin_gutierrez_education_item_0",
            }
        ],
    )

    assert normalized.mode == SearchMode.PROFILE
    assert normalized.candidate_name == "Austin Gutierrez"
    assert normalized.doc_id == "cv_austin_gutierrez"


def test_name_from_history_prefers_assistant_bold_name_over_user_institution() -> None:
    history = [
        {"role": "user", "content": "Which candidate graduated from Daniel University?"},
        {
            "role": "assistant",
            "content": "- **Austin Gutierrez** — graduated from Daniel University",
        },
        {"role": "user", "content": "Summarize his profile"},
    ]

    assert _name_from_history(history) == "Austin Gutierrez"


def test_recover_profile_target_ignores_institution_in_user_only_history() -> None:
    name, doc_id, reason = _recover_profile_target(
        "Summarize his profile",
        history=[
            {"role": "user", "content": "Which candidate graduated from Daniel University?"},
            {"role": "user", "content": "Summarize his profile"},
        ],
    )

    assert name is None
    assert doc_id is None
    assert reason is None


def test_router_node_recovers_pronoun_follow_up_from_prior_sources() -> None:
    with patch(
        "rag.agent.nodes.chat_json",
        return_value={
            "in_domain": True,
            "mode": "profile",
            "candidate_name": None,
            "keyword": None,
            "doc_id": None,
            "filters": {},
            "reason": "profile of one candidate",
        },
    ):
        state = router_node(
            {
                "query": "Summarize his profile",
                "messages": [
                    {
                        "role": "user",
                        "content": "Which candidate graduated from Daniel University?",
                    },
                    {
                        "role": "assistant",
                        "content": (
                            "## Education\n\n- **Austin Gutierrez** — graduated from "
                            "Daniel University"
                        ),
                    },
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
        )

    assert state["search_mode"] == SearchMode.PROFILE
    assert state["candidate_name"] == "Austin Gutierrez"
    assert state["doc_id"] == "cv_austin_gutierrez"


def test_normalize_decision_forces_hybrid_for_aggregate_after_prior_session() -> None:
    decision = RouterDecision(
        in_domain=True,
        mode=SearchMode.PROFILE,
        candidate_name="Austin Gutierrez",
        reason="profile inferred from history",
    )
    normalized = _normalize_decision(
        "Who has Python experience?",
        decision,
        history=[
            {"role": "user", "content": "Summarize his profile"},
            {"role": "assistant", "content": "## Austin Profile\n\nSkills: Python"},
        ],
        sources=[
            {
                "candidate_name": "Austin Gutierrez",
                "doc_id": "cv_austin_gutierrez",
                "section": "SKILLS",
                "chunk_id": "cv_austin_gutierrez_skills_list_0",
            }
        ],
    )

    assert normalized.mode == SearchMode.HYBRID
    assert normalized.candidate_name is None
    assert normalized.doc_id is None


def test_normalize_decision_does_not_pin_prior_candidate_on_new_lexical_topic() -> None:
    decision = RouterDecision(
        in_domain=True,
        mode=SearchMode.PROFILE,
        candidate_name=None,
        reason="profile of one candidate",
    )
    normalized = _normalize_decision(
        "Which candidate graduated from Martin University?",
        decision,
        sources=[
            {
                "candidate_name": "Austin Gutierrez",
                "doc_id": "cv_austin_gutierrez",
                "section": "EDUCATION",
                "chunk_id": "cv_austin_gutierrez_education_item_0",
            }
        ],
    )

    assert normalized.mode == SearchMode.HYBRID
    assert normalized.candidate_name is None


def test_recover_profile_target_uses_history_on_follow_up_without_sources() -> None:
    name, doc_id, reason = _recover_profile_target(
        "Summarize his profile",
        history=[
            {"role": "user", "content": "Which candidate graduated from Daniel University?"},
            {
                "role": "assistant",
                "content": "- **Austin Gutierrez** — graduated from Daniel University",
            },
        ],
    )

    assert name == "Austin Gutierrez"
    assert doc_id is None
    assert reason == "resolved from conversation history"


def test_router_node_does_not_pin_prior_candidate_on_corpus_question() -> None:
    with patch(
        "rag.agent.nodes.chat_json",
        return_value={
            "in_domain": True,
            "mode": "profile",
            "candidate_name": "Austin Gutierrez",
            "keyword": None,
            "doc_id": "cv_austin_gutierrez",
            "filters": {},
            "reason": "inferred from conversation",
        },
    ):
        state = router_node(
            {
                "query": "Who has Python experience?",
                "messages": [
                    {"role": "user", "content": "Summarize his profile"},
                    {"role": "assistant", "content": "## Austin Profile"},
                ],
                "sources": [
                    {
                        "candidate_name": "Austin Gutierrez",
                        "doc_id": "cv_austin_gutierrez",
                        "section": "SKILLS",
                        "chunk_id": "cv_austin_gutierrez_skills_list_0",
                    }
                ],
            }
        )

    assert state["search_mode"] == SearchMode.HYBRID
    assert state.get("candidate_name") is None
    assert state.get("doc_id") is None
