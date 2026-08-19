"""Shared debug formatting for agent routing metadata."""

from __future__ import annotations

import json
from typing import Any

from rag.agent.state import AgentState


def search_mode_value(state: dict[str, Any]) -> str:
    """Return the string value of ``search_mode`` from an agent state dict."""
    mode = state.get("search_mode")
    return getattr(mode, "value", str(mode or "unknown"))


def _format_optional(key: str, value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, dict) and not value:
        return None
    if isinstance(value, dict):
        return f"{key}={json.dumps(value, sort_keys=True)}"
    return f"{key}={value}"


def format_agent_debug(state: AgentState) -> str:
    """Format routing metadata and sources for CLI verbose and Chainlit debug."""
    lines = [
        "--- Routing ---",
        f"thread_id={state.get('thread_id', '')}",
        f"mode={search_mode_value(state)}",
        f"in_domain={state.get('in_domain')}",
        f"reason={state.get('route_reason', '')}",
    ]

    if state.get("rewritten_query"):
        lines.append(f"rewritten_query={state['rewritten_query']}")

    for key in ("candidate_name", "doc_id", "keyword"):
        formatted = _format_optional(key, state.get(key))
        if formatted:
            lines.append(formatted)

    filters = _format_optional("filters", state.get("filters"))
    if filters:
        lines.append(filters)

    messages = state.get("messages") or []
    lines.append(f"history_turns={len(messages)}")

    sources = state.get("sources") or []
    if sources:
        lines.append("")
        lines.append("--- Sources ---")
        for source in sources:
            lines.append(
                f"- {source.get('candidate_name')} | {source.get('doc_id')} | "
                f"{source.get('section')} | {source.get('chunk_id')}"
            )

    return "\n".join(lines)
