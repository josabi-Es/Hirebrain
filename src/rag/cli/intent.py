"""Heuristic intent detection for cv-ask (placeholder until LangGraph detector)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rag.shared.schemas import SearchMode

_PROFILE_HINTS = (
    "summarize the profile",
    "summarize profile",
    "profile of",
    "resume of",
    "tell me about",
    "describe the profile",
    "overview of",
)

_CONTACT_HINTS = (
    "contact info",
    "contact information",
    "contact details",
    "email of",
    "email for",
    "phone of",
    "phone number of",
    "how to contact",
    "how can i contact",
    "how do i contact",
    "reach out to",
)

_COMPANY_HINTS = (
    "work at",
    "works at",
    "worked at",
    "working at",
    "work for",
    "works for",
    "worked for",
    "employed at",
    "employed by",
)

_LEXICAL_HINTS = (
    "graduated from",
    "studied at",
    "studied in",
    "which candidate",
    "who graduated",
    "who studied",
    "university",
    "college",
    "institution",
    "degree from",
)

_FULL_PROFILE_RE = re.compile(
    r"\b(summari[sz]e|summary|profile|overview|resume of|describe|who is|"
    r"tell me (?:more )?about)\b",
    re.IGNORECASE,
)


def is_full_profile_request(query: str) -> bool:
    """True for 'summarize/profile of/tell me about' style full-profile requests."""
    return bool(_FULL_PROFILE_RE.search(query))


@dataclass(frozen=True)
class SearchPlan:
    """Resolved retrieval plan for a user question."""

    mode: SearchMode
    query: str
    keyword: str | None = None
    candidate_name: str | None = None
    doc_id: str | None = None
    filters: dict[str, Any] | None = None


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


_PROFILE_NAME_NOISE = frozenset(
    {
        "his",
        "her",
        "he",
        "she",
        "him",
        "them",
        "their",
        "that",
        "this",
        "the",
        "profile",
        "resume",
        "cv",
        "candidate",
    }
)


def _is_usable_candidate_name(name: str) -> bool:
    tokens = _normalize(name).split()
    if not tokens:
        return False
    if tokens[0] in _PROFILE_NAME_NOISE:
        return False
    return not all(token in _PROFILE_NAME_NOISE for token in tokens)


def extract_candidate_name(query: str) -> str | None:
    """Extract a candidate full name from profile-style questions."""
    patterns = (
        r"profile of\s+(.+?)(?:\?|$)",
        r"summarize(?: the profile of)?\s+(.+?)(?:\?|$)",
        r"overview of\s+(.+?)(?:\?|$)",
        r"resume of\s+(.+?)(?:\?|$)",
        r"tell me about\s+(.+?)(?:\?|$)",
        r"describe(?: the profile of)?\s+(.+?)(?:\?|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            name = match.group(1).strip(" .")
            if name and _is_usable_candidate_name(name):
                return name
    return None


def extract_contact_candidate(query: str) -> str | None:
    """Extract a candidate name from contact-style questions."""
    patterns = (
        r"(?:contact info(?:rmation)?|contact details)\s+(?:of|for)\s+(.+?)(?:\?|$)",
        r"(.+?)'s\s+(?:contact|email|phone)",
        r"(?:email|phone(?:\s+number)?)\s+(?:of|for)\s+(.+?)(?:\?|$)",
        r"(?:how (?:can|do) i contact|how to contact|reach out to|contact)\s+(.+?)(?:\?|$)",
        r"(?:contact info(?:rmation)?|contact details)\s+of\s*(.+?)(?:\?|$)",
        r"contact info\s*of\s*(.+?)(?:\?|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            name = match.group(1).strip(" .'")
            name = re.sub(
                r"\b(contact|info|information|details|the|me|give|please)\b",
                "",
                name,
                flags=re.IGNORECASE,
            ).strip(" .'")
            if name:
                return name
    return None


def extract_company(query: str) -> str | None:
    """Extract an employer name from company-style questions."""
    pattern = (
        r"(?:works?|worked|working|employed)\s+(?:at|for|by)\s+(.+?)"
        r"(?:\s+company\b)?(?:\?|\.|$)"
    )
    match = re.search(pattern, query, flags=re.IGNORECASE)
    if match:
        company = match.group(1).strip(" .'")
        company = re.sub(r"\s+company$", "", company, flags=re.IGNORECASE).strip(" .'")
        if company and company.lower() not in {"the", "a", "an"}:
            return company
    return None


def extract_lexical_keyword(query: str) -> str | None:
    """Extract institution or keyword from education-style questions."""
    patterns = (
        r"graduated from\s+(.+?)(?:\?|$)",
        r"studied at\s+(.+?)(?:\?|$)",
        r"studied in\s+(.+?)(?:\?|$)",
        r"from\s+([A-Z][\w\s.&-]+?)(?:\?|$)",
        r"at\s+([A-Z][\w\s.&-]+?)(?:\?|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            keyword = match.group(1).strip(" .")
            if keyword and keyword.lower() not in {"the", "a", "an"}:
                return keyword
    quoted = re.search(r'"([^"]+)"|\'([^\']+)\'', query)
    if quoted:
        return (quoted.group(1) or quoted.group(2) or "").strip()
    return None


def detect_search_plan(query: str) -> SearchPlan:
    """Infer search mode and parameters from a natural-language question."""
    normalized = _normalize(query)

    if any(hint in normalized for hint in _CONTACT_HINTS):
        return SearchPlan(
            mode=SearchMode.PROFILE,
            query=query,
            candidate_name=extract_contact_candidate(query),
        )

    if any(hint in normalized for hint in _PROFILE_HINTS):
        return SearchPlan(
            mode=SearchMode.PROFILE,
            query=query,
            candidate_name=extract_candidate_name(query),
        )

    if any(hint in normalized for hint in _COMPANY_HINTS):
        company = extract_company(query)
        if company:
            return SearchPlan(
                mode=SearchMode.LEXICAL,
                query=query,
                keyword=company,
                filters={"company": company},
            )

    if any(hint in normalized for hint in _LEXICAL_HINTS):
        keyword = extract_lexical_keyword(query)
        filters: dict[str, Any] = {"section": "EDUCATION"}
        if keyword:
            filters["institution"] = keyword
        return SearchPlan(
            mode=SearchMode.LEXICAL,
            query=query,
            keyword=keyword,
            filters=filters,
        )

    return SearchPlan(mode=SearchMode.HYBRID, query=query)


def example_queries() -> list[tuple[str, SearchPlan]]:
    """Built-in examples aligned with the indexed corpus."""
    return [
        (
            "Summarize the profile of Austin Gutierrez",
            SearchPlan(
                mode=SearchMode.PROFILE,
                query="Summarize the profile of Austin Gutierrez",
                candidate_name="Austin Gutierrez",
            ),
        ),
        (
            "Which candidate graduated from Martin University?",
            SearchPlan(
                mode=SearchMode.LEXICAL,
                query="Which candidate graduated from Martin University?",
                keyword="Martin University",
                filters={"section": "EDUCATION", "institution": "Martin University"},
            ),
        ),
        (
            "Who has experience with Python?",
            SearchPlan(mode=SearchMode.HYBRID, query="Who has experience with Python?"),
        ),
    ]
