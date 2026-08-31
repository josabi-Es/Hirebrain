"""Tests for Ollama temperature helpers."""

from __future__ import annotations

from rag.agent.llm import answer_temperature


def test_answer_temperature_default(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_ANSWER_TEMPERATURE", raising=False)
    assert answer_temperature() == 0.2


def test_answer_temperature_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_ANSWER_TEMPERATURE", "0.15")
    assert answer_temperature() == 0.15
