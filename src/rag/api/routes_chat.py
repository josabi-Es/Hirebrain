"""Chat API routes for the LangGraph CV screening agent."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from rag.agent.graph import run_agent

router = APIRouter()


class ChatRequest(BaseModel):
    """Request body for a chat turn."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1)
    thread_id: str | None = None


class ChatResponse(BaseModel):
    """Response body returned by the CV screening agent."""

    answer: str
    thread_id: str
    mode: str | None = None
    in_domain: bool
    route_reason: str = ""
    rewritten_query: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    thread_id = request.thread_id or str(uuid.uuid4())
    state = run_agent(request.query, thread_id=thread_id)
    mode = state.get("search_mode")
    return ChatResponse(
        answer=state.get("answer", ""),
        thread_id=thread_id,
        mode=getattr(mode, "value", str(mode)) if mode is not None else None,
        in_domain=bool(state.get("in_domain", False)),
        route_reason=state.get("route_reason", ""),
        rewritten_query=state.get("rewritten_query"),
        sources=state.get("sources", []),
    )


def register_routes(_app: Any) -> None:
    """Register chat endpoints on the FastAPI app."""
    _app.include_router(router)
