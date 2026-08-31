from __future__ import annotations

from rag.cli.intent import detect_search_plan, extract_candidate_name
from rag.shared.schemas import SearchMode


def test_detect_profile_mode() -> None:
    plan = detect_search_plan("Summarize the profile of Austin Gutierrez")
    assert plan.mode == SearchMode.PROFILE
    assert plan.candidate_name == "Austin Gutierrez"


def test_detect_lexical_mode() -> None:
    plan = detect_search_plan("Which candidate graduated from Martin University?")
    assert plan.mode == SearchMode.LEXICAL
    assert plan.keyword == "Martin University"
    assert plan.filters == {
        "section": "EDUCATION",
        "institution": "Martin University",
    }


def test_detect_hybrid_mode() -> None:
    plan = detect_search_plan("Who has experience with Python?")
    assert plan.mode == SearchMode.HYBRID


def test_flag_only_lexical_query() -> None:
    from argparse import Namespace

    from rag.cli.ask_main import _resolve_plan

    plan = _resolve_plan(
        Namespace(
            question=None,
            mode="lexical",
            candidate=None,
            keyword="Martin University",
            filters=["section=EDUCATION"],
        )
    )
    assert plan.mode.value == "lexical"
    assert plan.keyword == "Martin University"
    assert plan.filters == {"section": "EDUCATION"}


def test_flag_only_profile_query() -> None:
    from argparse import Namespace

    from rag.cli.ask_main import _resolve_plan

    plan = _resolve_plan(
        Namespace(
            question=None,
            mode="profile",
            candidate="Austin Gutierrez",
            keyword=None,
            filters=None,
        )
    )
    assert plan.mode == SearchMode.PROFILE
    assert plan.candidate_name == "Austin Gutierrez"
