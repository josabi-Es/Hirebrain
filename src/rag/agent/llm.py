"""Small Ollama helpers for the LangGraph agent."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _temperature() -> float:
    raw = os.getenv("OLLAMA_TEMPERATURE", "0.2").strip()
    return float(raw or "0.2")


def answer_temperature() -> float:
    """Temperature for evidence-grounded RAG answers (generate node, ask CLI)."""
    raw = os.getenv("OLLAMA_ANSWER_TEMPERATURE", "0.2").strip()
    return float(raw or "0.2")


def _client() -> Any:
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError("ollama package is required for the RAG agent") from exc

    host = os.getenv("OLLAMA_HOST", "").strip()
    return ollama.Client(host=host) if host else ollama


@lru_cache(maxsize=4)
def get_chat_model(temperature: float | None = None) -> Any:
    """Return a LangChain Ollama chat model instrumented for LangGraph streaming."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise RuntimeError("langchain-ollama is required for streaming generation") from exc

    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3:8b"),
        base_url=os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434",
        temperature=_temperature() if temperature is None else temperature,
    )


def get_answer_chat_model() -> Any:
    """Chat model for evidence-grounded RAG answers (``OLLAMA_ANSWER_TEMPERATURE``)."""
    return get_chat_model(temperature=answer_temperature())


def _response_content(response: Any) -> str:
    if isinstance(response, dict):
        return str(response.get("message", {}).get("content", ""))
    message = getattr(response, "message", None)
    if isinstance(message, dict):
        return str(message.get("content", ""))
    content = getattr(message, "content", "") if message is not None else ""
    return str(content)


def _build_messages(
    prompt: str,
    *,
    system: str,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})
    return messages


def chat(
    prompt: str,
    *,
    system: str,
    temperature: float | None = None,
    history: list[dict[str, str]] | None = None,
    json_mode: bool = False,
) -> str:
    """Send a chat request to Ollama and return text content.

    When ``json_mode`` is True, Ollama is asked to emit valid JSON, which makes
    structured router output far more reliable for small models. ``history`` lets
    the caller pass prior conversation turns for coreference resolution.
    """
    options: dict[str, Any] = {
        "temperature": _temperature() if temperature is None else temperature
    }
    extra: dict[str, Any] = {"format": "json"} if json_mode else {}
    response = _client().chat(
        model=os.getenv("OLLAMA_MODEL", "llama3:8b"),
        messages=_build_messages(prompt, system=system, history=history),
        options=options,
        **extra,
    )
    content = _response_content(response).strip()
    if not content:
        raise RuntimeError("Ollama returned an empty response")
    return content


def chat_json(
    prompt: str,
    *,
    system: str,
    temperature: float = 0.0,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Ask Ollama for JSON and parse the first JSON object in the response."""
    content = chat(
        prompt,
        system=system,
        temperature=temperature,
        history=history,
        json_mode=True,
    )
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match is None:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object from Ollama")
    return parsed
