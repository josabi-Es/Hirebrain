"""LangGraph graph definition."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from rag.agent.nodes import (
    generate_node,
    reject_node,
    retrieve_node,
    rewrite_node,
    route_after_retrieve,
    route_after_router,
    router_node,
)
from rag.agent.state import AgentState


@lru_cache(maxsize=1)
def build_graph() -> Any:
    """Build and compile the CV screening agent graph."""
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("reject", reject_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"rewrite": "rewrite", "reject": "reject"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"generate": "generate", "reject": "reject"},
    )
    graph.add_edge("generate", END)
    graph.add_edge("reject", END)
    return graph.compile(checkpointer=MemorySaver())


def run_agent(query: str, *, thread_id: str | None = None) -> AgentState:
    """Run the compiled CV screening graph for a single user query."""
    active_thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": active_thread_id}}
    state = build_graph().invoke({"query": query}, config=config)
    return {**state, "thread_id": active_thread_id}


async def astream_agent(
    query: str,
    *,
    thread_id: str | None = None,
) -> AsyncIterator[tuple[str, str | AgentState]]:
    """Stream final-answer tokens and yield the final agent state."""
    active_thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": active_thread_id}}
    final_state: AgentState | None = None

    async for mode, chunk in build_graph().astream(
        {"query": query},
        config=config,
        stream_mode=["messages", "values"],
    ):
        if mode == "messages":
            token, metadata = chunk
            content = getattr(token, "content", "")
            if content and metadata.get("langgraph_node") == "generate":
                yield "token", str(content)
        elif mode == "values":
            final_state = chunk

    yield "final", {**(final_state or {}), "thread_id": active_thread_id}
