"""LLM enrichment for CV summary and experience descriptions."""

from __future__ import annotations

import json
import re
from typing import Any

from cv_creator.core.exceptions import LLMInvalidSchemaError, LLMTimeoutError
from cv_creator.core.models import CVData, LLMConfig

SYSTEM_HINT = "All CV content must be in English (en-US). Return only valid JSON."

WRITER_PROMPT = """You are an expert Executive Resume Writer.

Given the candidate skeleton JSON below, enrich this CV in English to make it highly competitive for Recruiters and an AI HR RAG system.
Return a single JSON object with this exact structure:
{{
  "summary": "A cohesive 3-sentence professional summary highlighting the candidate's core competencies, total experience, and primary value proposition.",
  "experience": [
    {{
      "job_title": "...",
      "company": "...",
      "description": "3 strong bullet-point style sentences starting with action verbs, focusing on qualitative milestones and business value."
    }}
  ]
}}

Rules for Content Generation:
- Tone: Professional, objective, and active. STRICTLY DO NOT use first-person pronouns (I, me, my, we).
- Semantic Alignment (CRITICAL): The descriptions MUST logically bridge the `job_title` and the exact `skills` listed. Explicitly describe HOW those specific tools, methodologies, or skills were utilized to solve problems or deliver projects.
- Milestones over Metrics: DO NOT force artificial percentages (%), financial numbers, or fake KPIs. Focus instead on qualitative milestones: what was built, organized, managed, or improved. Describe the deliverables and the business value of the work.
- Scale & Context: Imply the scale and complexity of the work using natural language appropriate for the domain (e.g., 'cross-functional teams', 'enterprise-level systems', 'high-volume customer base', 'multidisciplinary projects') rather than precise numbers.
- Seniority Awareness (CRITICAL): Match narrative depth to each `job_title` exactly as given. Juniors execute and support; mid-level roles own delivery; Seniors design, lead, and strategize; managers and directors focus on teams, strategy, and stakeholder alignment. Never describe a Senior/Director role with junior-level scope, and never describe a Junior/entry role as leading organization-wide strategy.
- Career Coherence: Treat the experience list as a single ascending career arc (oldest entry → most recent). Earlier roles should sound less senior than later ones; the summary must reflect the seniority of the most recent `job_title`, not exaggerate or understate it.

Rules for Schema & Stability (CRITICAL):
- You MUST keep `job_title` and `company` EXACTLY as provided in the skeleton. Do not change capitalization or fix typos.
- Improve ONLY the `summary` and `description` fields.
- Do not add or remove experience entries.
- Do not invent new hard skills, degrees, or tools not already present or heavily implied by the skeleton's 'skills' array.

Candidate skeleton:
{skeleton_json}
"""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise LLMInvalidSchemaError("Could not parse JSON from LLM response") from None


def _match_descriptions(
    base_cv: CVData,
    generated_experience: list[dict[str, Any]],
) -> list[str]:
    descriptions: list[str] = []
    by_key = {
        (str(item.get("job_title", "")).strip(), str(item.get("company", "")).strip()): str(
            item.get("description", "")
        ).strip()
        for item in generated_experience
    }
    for exp in base_cv.experience:
        key = (exp.job_title, exp.company)
        description = by_key.get(key, "").strip()
        if not description:
            raise LLMInvalidSchemaError(
                f"Missing description for experience item: {exp.job_title} at {exp.company}"
            )
        descriptions.append(description)
    return descriptions


def _call_ollama(prompt: str, cfg: LLMConfig) -> str:
    try:
        import ollama
    except ImportError as e:
        raise LLMTimeoutError("ollama package not installed") from e

    client_kwargs: dict[str, Any] = {}
    if cfg.host:
        client_kwargs["host"] = cfg.host
    client = ollama.Client(**client_kwargs) if client_kwargs else ollama

    response = client.chat(
        model=cfg.model,
        messages=[
            {"role": "system", "content": SYSTEM_HINT},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": cfg.temperature},
    )
    return response["message"]["content"]


def enrich_cv_with_llm(cv_data: CVData, llm_cfg: LLMConfig) -> CVData:
    """Enrich CV summary and experience descriptions using Ollama."""
    skeleton_json = cv_data.model_dump_json(indent=2)
    last_error: Exception | None = None

    for _ in range(llm_cfg.max_retries):
        prompt = WRITER_PROMPT.format(skeleton_json=skeleton_json)

        try:
            raw = _call_ollama(prompt, llm_cfg)
            data = _extract_json(raw)
            summary = str(data.get("summary", "")).strip()
            if not summary:
                raise LLMInvalidSchemaError("Missing summary in LLM response")
            descriptions = _match_descriptions(
                cv_data,
                list(data.get("experience", [])),
            )

            updated_experience = []
            for index, exp in enumerate(cv_data.experience):
                updated_experience.append(
                    exp.model_copy(update={"description": descriptions[index]})
                )
            return cv_data.model_copy(
                update={
                    "summary": summary,
                    "experience": updated_experience,
                }
            )
        except (LLMInvalidSchemaError, LLMTimeoutError, KeyError, TypeError) as e:
            last_error = e
        except Exception as e:
            last_error = e

    if last_error:
        raise LLMTimeoutError(f"LLM enrichment failed: {last_error}") from last_error
    raise LLMTimeoutError("LLM enrichment failed without explicit error")
