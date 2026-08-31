"""LangGraph node functions for CV screening."""

from __future__ import annotations

import re
from typing import Any

from rag.agent.llm import chat, chat_json, get_answer_chat_model
from rag.agent.prompts import (
    ANSWER_SYSTEM_PROMPT,
    REWRITE_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    build_answer_prompt,
    build_rewrite_prompt,
    build_router_prompt,
    no_evidence_answer,
    out_of_domain_answer,
)
from rag.agent.state import AgentState
from rag.cli.intent import (
    detect_search_plan,
    extract_candidate_name,
    is_full_profile_request,
)
from rag.retrieval.router import retrieve
from rag.shared.schemas import RetrievedChunk, RouterDecision, SearchMode
from rag.shared.settings import get_settings

_VALID_MODES = {mode.value for mode in SearchMode}
# Capitalized tokens, allowing internal caps like "McCoy" or "O'Brien".
_NAME_RE = re.compile(r"\b([A-Z][\w']*(?:\s+[A-Z][\w']*)*)\b")


_DOMAIN_HINTS = (
    "candidate",
    "cv",
    "resume",
    "profile",
    "skill",
    "skills",
    "experience",
    "education",
    "degree",
    "university",
    "college",
    "employer",
    "company",
    "worked",
    "works",
    "graduated",
    "screen",
    "contact",
    "email",
    "phone",
)

_GREETINGS = frozenset(
    {
        "hi",
        "hii",
        "hey",
        "hello",
        "yo",
        "hola",
        "thanks",
        "thank you",
        "thx",
        "bye",
        "goodbye",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "whats up",
        "what's up",
        "sup",
        "ok",
        "okay",
    }
)

_HISTORY_TURNS = 6


def _looks_domain_related(query: str) -> bool:
    normalized = query.lower()
    return any(hint in normalized for hint in _DOMAIN_HINTS)


def _is_greeting(query: str) -> bool:
    normalized = query.lower().strip(" .!?,")
    if normalized in _GREETINGS:
        return True
    # Compound greetings like "hi, good morning" or "hello! how are you".
    segments = [
        segment.strip(" .!?,")
        for segment in re.split(r"[,.!?;]+", normalized)
        if segment.strip(" .!?,")
    ]
    return len(segments) > 1 and all(segment in _GREETINGS for segment in segments)


def _recent_history(state: AgentState) -> list[dict[str, Any]]:
    return list(state.get("messages", []))[-_HISTORY_TURNS:]


def _fallback_decision(query: str) -> RouterDecision:
    plan = detect_search_plan(query)
    return RouterDecision(
        in_domain=_looks_domain_related(query),
        mode=plan.mode,
        candidate_name=plan.candidate_name,
        keyword=plan.keyword,
        doc_id=plan.doc_id,
        filters=plan.filters or {},
        reason="LLM router failed; used heuristic fallback.",
    )


def _coerce_router_payload(payload: Any) -> dict[str, Any]:
    """Sanitize a raw LLM router dict so an odd field never crashes validation.

    Unknown keys are dropped by the schema (``extra="ignore"``); here we additionally
    coerce an invalid/missing ``mode`` to "hybrid" so the router degrades gracefully
    instead of raising and falling back to the heuristic path.
    """
    if not isinstance(payload, dict):
        raise ValueError("Router payload must be a JSON object")
    data = dict(payload)
    mode = data.get("mode")
    if isinstance(mode, str):
        mode = mode.strip().lower()
    if mode not in _VALID_MODES:
        data["mode"] = SearchMode.HYBRID.value
    else:
        data["mode"] = mode
    if not isinstance(data.get("filters"), dict):
        data["filters"] = {}
    return data


_NAME_STOPWORDS = frozenset(
    {
        "i",
        "the",
        "who",
        "what",
        "which",
        "give",
        "me",
        "tell",
        "show",
        "summarize",
        "profile",
        "resume",
        "contact",
        "info",
        "information",
        "skills",
        "experience",
        "education",
        "company",
        "candidate",
        "candidates",
    }
)

_INSTITUTION_MARKERS = frozenset(
    {"university", "college", "institute", "school", "academy"}
)

_BOLD_NAME_RE = re.compile(r"\*\*([^*]+?)\*\*")


def _looks_like_institution(name: str) -> bool:
    tokens = name.lower().split()
    return any(token in _INSTITUTION_MARKERS for token in tokens)


def _pick_person_name(names: list[str]) -> str | None:
    filtered = [
        name.strip()
        for name in names
        if name.strip()
        and name.lower() not in _NAME_STOPWORDS
        and not _looks_like_institution(name)
    ]
    if not filtered:
        return None
    multi = [name for name in filtered if " " in name]
    if multi:
        return multi[-1]
    return filtered[-1]


_AGGREGATE_HINTS = (
    "who ",
    "which candidate",
    "which candidates",
    "list ",
    "how many",
    "all candidates",
    "everyone",
    "anybody",
    "anyone",
)

_FOLLOW_UP_PRONOUN_RE = re.compile(
    r"\b(his|her|he|she|him|them|their|that candidate|this candidate)\b",
    re.IGNORECASE,
)

_IMPLICIT_PROFILE_RE = re.compile(
    r"\b(?:"
    r"summarize(?: the)? profile|"
    r"contact info(?:rmation)?|"
    r"contact details|"
    r"what are (?:his|her|their) skills|"
    r"(?:give|show)(?: me)? (?:his|her|their) (?:skills|experience|contact)"
    r")\b",
    re.IGNORECASE,
)


def _is_aggregate_question(query: str) -> bool:
    normalized = query.lower().strip()
    return normalized.startswith("who") or any(
        hint in normalized for hint in _AGGREGATE_HINTS
    )


def _is_follow_up_candidate_query(query: str) -> bool:
    """True when the query refers to a prior candidate via pronoun or implicit profile."""
    if _is_aggregate_question(query):
        return False
    if _FOLLOW_UP_PRONOUN_RE.search(query):
        return True
    if extract_candidate_name(query):
        return False
    return bool(_IMPLICIT_PROFILE_RE.search(query))


def _candidate_from_prior_sources(
    sources: list[dict[str, Any]] | None,
) -> tuple[str | None, str | None]:
    """Recover a single candidate from the previous turn's retrieval sources."""
    names = {
        str(source.get("candidate_name", "")).strip()
        for source in sources or []
        if source.get("candidate_name")
    }
    doc_ids = {
        str(source.get("doc_id", "")).strip()
        for source in sources or []
        if source.get("doc_id")
    }
    if len(names) == 1:
        return names.pop(), doc_ids.pop() if len(doc_ids) == 1 else None
    return None, None


def _name_from_history(history: list[dict[str, Any]] | None) -> str | None:
    """Best-effort: recover the most recently mentioned candidate name.

    Assistant turns are scanned first for Markdown-bold names (e.g. **Austin Gutierrez**).
    User turns are scanned next, skipping institution names such as "Daniel University".
    """
    for turn in reversed(history or []):
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            bold_names = _BOLD_NAME_RE.findall(content)
            picked = _pick_person_name(bold_names)
            if picked:
                return picked
            capitalized = _NAME_RE.findall(content)
            picked = _pick_person_name([name for name in capitalized if " " in name])
            if picked:
                return picked
            continue
        if role == "user":
            picked = _pick_person_name(_NAME_RE.findall(content))
            if picked:
                return picked
    return None


def _recover_profile_target(
    query: str,
    *,
    history: list[dict[str, Any]] | None = None,
    sources: list[dict[str, Any]] | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Return (candidate_name, doc_id, recovery_reason) for an unresolved profile route."""
    extracted = extract_candidate_name(query)
    if not _is_follow_up_candidate_query(query):
        if extracted:
            return extracted, None, "extracted from query"
        return None, None, None

    prior_name, prior_doc_id = _candidate_from_prior_sources(sources)
    if prior_name or prior_doc_id:
        return prior_name, prior_doc_id, "resolved from prior turn sources"

    from_history = _name_from_history(history)
    if from_history:
        return from_history, None, "resolved from conversation history"

    if extracted:
        return extracted, None, "extracted from query"

    return None, None, None


def _normalize_decision(
    query: str,
    decision: RouterDecision,
    *,
    history: list[dict[str, Any]] | None = None,
    sources: list[dict[str, Any]] | None = None,
) -> RouterDecision:
    if not decision.in_domain:
        return decision
    if decision.mode == SearchMode.PROFILE and _is_aggregate_question(query):
        return decision.model_copy(
            update={
                "mode": SearchMode.HYBRID,
                "candidate_name": None,
                "doc_id": None,
                "reason": (
                    f"{decision.reason} Aggregate corpus query; forced hybrid retrieval."
                ).strip(),
            }
        )
    if decision.mode == SearchMode.PROFILE and not (
        decision.candidate_name or decision.doc_id
    ):
        recovered_name, recovered_doc_id, recovery_reason = _recover_profile_target(
            query,
            history=history,
            sources=sources,
        )
        if recovered_name or recovered_doc_id:
            label = recovered_name or recovered_doc_id
            return decision.model_copy(
                update={
                    "candidate_name": recovered_name,
                    "doc_id": recovered_doc_id,
                    "reason": (
                        f"{decision.reason} Recovered candidate '{label}' "
                        f"({recovery_reason})."
                    ).strip(),
                }
            )
        return decision.model_copy(
            update={
                "mode": SearchMode.HYBRID,
                "reason": (
                    f"{decision.reason} Profile route lacked candidate/doc_id; "
                    "falling back to hybrid retrieval."
                ).strip(),
            }
        )
    if not decision.reason:
        return decision.model_copy(update={"reason": f"Routed query: {query}"})
    return decision


def _chunks_from_state(state: AgentState) -> list[RetrievedChunk]:
    return [
        RetrievedChunk.model_validate(chunk)
        for chunk in state.get("retrieved_chunks", [])
    ]


def _sources(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    sources: list[dict[str, Any]] = []
    for chunk in chunks:
        payload = chunk.payload
        key = (payload.doc_id, payload.section, payload.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "doc_id": payload.doc_id,
                "candidate_name": payload.candidate_name,
                "section": payload.section,
                "chunk_id": payload.chunk_id,
                "score": chunk.score,
            }
        )
    return sources


def router_node(state: AgentState) -> AgentState:
    """Classify the query and reject clearly out-of-domain requests."""
    query = state.get("query", "").strip()
    if not query:
        return {
            **state,
            "in_domain": False,
            "route_reason": "Empty query.",
            "answer": out_of_domain_answer(),
        }

    if _is_greeting(query):
        return {
            **state,
            "in_domain": False,
            "route_reason": "Greeting or small talk; out of CV screening domain.",
        }

    history = _recent_history(state)
    try:
        decision = RouterDecision.model_validate(
            _coerce_router_payload(
                chat_json(
                    build_router_prompt(query, history=history),
                    system=ROUTER_SYSTEM_PROMPT,
                    temperature=0.0,
                    history=history,
                )
            )
        )
    except Exception:
        decision = _fallback_decision(query)

    decision = _normalize_decision(
        query,
        decision,
        history=history,
        sources=state.get("sources"),
    )
    return {
        **state,
        "in_domain": decision.in_domain,
        "route_reason": decision.reason,
        "search_mode": decision.mode,
        "candidate_name": decision.candidate_name,
        "keyword": decision.keyword,
        "doc_id": decision.doc_id,
        "filters": decision.filters or None,
    }


_EXACT_FILTER_FIELDS = ("company", "institution", "degree")


def _has_exact_metadata_filter(filters: dict[str, Any] | None) -> bool:
    return bool(filters) and any(filters.get(field) for field in _EXACT_FILTER_FIELDS)


def rewrite_node(state: AgentState) -> AgentState:
    """Rewrite in-domain queries into retrieval-friendly text."""
    query = state.get("query", "").strip()
    mode = state.get("search_mode", SearchMode.HYBRID)
    if mode == SearchMode.PROFILE:
        return {**state, "rewritten_query": query}

    # Exact-metadata lexical search scrolls by filter and ignores the query text, so
    # rewriting it is pointless (and risks LLM placeholder/template noise).
    if mode == SearchMode.LEXICAL and _has_exact_metadata_filter(state.get("filters")):
        return {**state, "rewritten_query": state.get("keyword") or query}

    history = _recent_history(state)
    try:
        rewritten = chat(
            build_rewrite_prompt(
                query,
                mode=mode,
                keyword=state.get("keyword"),
                candidate_name=state.get("candidate_name"),
                history=history,
            ),
            system=REWRITE_SYSTEM_PROMPT,
            temperature=0.0,
            history=history,
        ).strip(" \n\t\"'")
    except Exception:
        rewritten = query

    return {**state, "rewritten_query": rewritten or query}


def retrieve_node(state: AgentState) -> AgentState:
    """Retrieve relevant chunks using the unified retrieval router."""
    query = (state.get("rewritten_query") or state.get("query") or "").strip()
    mode = state.get("search_mode", SearchMode.HYBRID)
    original = (state.get("query") or "").strip()
    rerank_top_k = None
    if mode == SearchMode.HYBRID and _is_aggregate_question(original):
        rerank_top_k = get_settings().retrieval_top_k

    chunks = retrieve(
        query,
        mode=mode,
        filters=state.get("filters"),
        keyword=state.get("keyword"),
        candidate_name=state.get("candidate_name"),
        doc_id=state.get("doc_id"),
        rerank_top_k=rerank_top_k,
    )
    return {
        **state,
        "retrieved_chunks": [chunk.model_dump() for chunk in chunks],
        "sources": _sources(chunks),
    }


def _append_turn(state: AgentState, answer: str) -> list[dict[str, Any]]:
    history = list(state.get("messages", []))
    query = (state.get("query") or "").strip()
    return history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ]


_META_PREFIX_RE = re.compile(
    r"^\s*(based on|according to|from)\b[^\n.:,]*"
    r"(provided|retrieved|given|context|chunk|cv|resume)[^\n.:,]*[.:,]\s*",
    flags=re.IGNORECASE,
)
_SOURCES_TAIL_RE = re.compile(
    r"\n+\s*(sources?|relevant candidates?)\s*:.*$",
    flags=re.IGNORECASE | re.DOTALL,
)


def _clean_answer(answer: str) -> str:
    """Strip meta boilerplate and model-generated source/candidate listings."""
    cleaned = _SOURCES_TAIL_RE.sub("", answer)
    cleaned = _META_PREFIX_RE.sub("", cleaned)
    return cleaned.strip()


def generate_node(state: AgentState) -> AgentState:
    """Generate an evidence-grounded answer from retrieved chunks."""
    chunks = _chunks_from_state(state)
    question = (state.get("query") or "").strip()
    mode = state.get("search_mode", SearchMode.HYBRID)
    full_profile = mode == SearchMode.PROFILE and is_full_profile_request(question)
    response = get_answer_chat_model().invoke(
        [
            ("system", ANSWER_SYSTEM_PROMPT),
            ("human", build_answer_prompt(question, chunks, mode=mode, full_profile=full_profile)),
        ]
    )
    answer = _clean_answer(str(response.content or "").strip())
    return {
        **state,
        "answer": answer,
        "sources": _sources(chunks),
        "messages": _append_turn(state, answer),
    }


def reject_node(state: AgentState) -> AgentState:
    """Return a controlled response for off-domain or unsupported queries."""
    if not state.get("in_domain", False):
        answer = out_of_domain_answer()
    else:
        answer = no_evidence_answer()
    return {
        **state,
        "answer": answer,
        "sources": [],
        "messages": _append_turn(state, answer),
    }


def route_after_router(state: AgentState) -> str:
    """Route to rewrite for in-domain queries, otherwise reject."""
    return "rewrite" if state.get("in_domain", False) else "reject"


def route_after_retrieve(state: AgentState) -> str:
    """Route to generation only when retrieval produced evidence."""
    return "generate" if state.get("retrieved_chunks") else "reject"
