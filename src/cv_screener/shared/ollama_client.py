"""Single point of construction for every Ollama client used in this project.

cv_creator (LLM enrichment) and rag.agent (router/rewrite/generate) both import
this instead of building their own `ollama.Client`/`ChatOllama` instance.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "llama3:8b"
DEFAULT_HOST = "http://127.0.0.1:11434"


def model_name() -> str:
    return os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)


def host() -> str:
    return os.getenv("OLLAMA_HOST", DEFAULT_HOST)


def temperature() -> float:
    return float(os.getenv("OLLAMA_TEMPERATURE", "0.2") or "0.2")


@lru_cache(maxsize=1)
def raw_client() -> Any:
    """Low-level `ollama` client, used for one-shot chat() calls (enrichment, router)."""
    import ollama

    return ollama.Client(host=host())


@lru_cache(maxsize=4)
def chat_model(model_temperature: float | None = None) -> Any:
    """LangChain ChatOllama, used by the LangGraph agent for streaming generation."""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model_name(),
        base_url=host(),
        temperature=temperature() if model_temperature is None else model_temperature,
    )


def _response_content(response: Any) -> str:
    if isinstance(response, dict):
        return str(response.get("message", {}).get("content", ""))
    message = getattr(response, "message", None)
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
    return str(content or "")


def chat(
    prompt: str,
    *,
    system: str,
    model_temperature: float | None = None,
    history: list[dict[str, str]] | None = None,
    json_mode: bool = False,
) -> str:
    """Send a one-shot chat request to Ollama and return the text content."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for turn in history or []:
        if turn.get("role") in {"user", "assistant"} and turn.get("content", "").strip():
            messages.append({"role": turn["role"], "content": turn["content"].strip()})
    messages.append({"role": "user", "content": prompt})

    response = raw_client().chat(
        model=model_name(),
        messages=messages,
        options={"temperature": temperature() if model_temperature is None else model_temperature},
        **({"format": "json"} if json_mode else {}),
    )
    content = _response_content(response).strip()
    if not content:
        raise RuntimeError("Ollama returned an empty response")
    return content


def chat_json(
    prompt: str,
    *,
    system: str,
    model_temperature: float = 0.0,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Ask Ollama for JSON and parse the first JSON object in the response."""
    content = chat(
        prompt,
        system=system,
        model_temperature=model_temperature,
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
