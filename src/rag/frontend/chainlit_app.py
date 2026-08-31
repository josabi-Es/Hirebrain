"""Chainlit frontend for the LangGraph CV screening agent."""

from __future__ import annotations

import os
from typing import cast

import chainlit as cl

from rag.agent.debug import format_agent_debug
from rag.agent.graph import astream_agent
from rag.agent.state import AgentState


def _chainlit_verbose_enabled() -> bool:
    return os.getenv("RAG_CHAINLIT_VERBOSE", "").lower() in {"1", "true", "yes"}


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Stream the agent's final answer."""
    answer = cl.Message(content="")
    await answer.send()

    final_state: AgentState = {}

    async for kind, data in astream_agent(
        message.content,
        thread_id=cl.context.session.id,
    ):
        if kind == "token":
            await answer.stream_token(cast(str, data))
            continue
        final_state = cast(AgentState, data)

    final_answer = final_state.get("answer", "")
    if final_answer:
        answer.content = final_answer

    await answer.update()

    if _chainlit_verbose_enabled():
        debug_text = format_agent_debug(final_state)
        await cl.Message(content=debug_text, author="debug").send()
