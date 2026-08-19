from __future__ import annotations

import re
from dataclasses import dataclass

from rag.shared.models import CVChunk, CVSection

DATE_LINE_PATTERN = re.compile(
    r"^\s*\d{2}/\d{4}\s*[–\-]\s*(?:Present|\d{2}/\d{4})\s*$",
    re.IGNORECASE,
)

SECTION_TO_CHUNK_TYPE: dict[str, str] = {
    "SUMMARY": "summary_chunk",
    "EXPERIENCE": "experience_item",
    "EDUCATION": "education_item",
    "SKILLS": "skills_list",
    "CONTACT_INFO": "contact_info",
}

DEFAULT_MAX_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 120


@dataclass(frozen=True)
class ParsedEntry:
    """Structured block parsed from EXPERIENCE or EDUCATION text."""

    line1: str | None
    line2: str | None
    date_range: str | None
    description: str
    full_text: str


class ExperienceParser:
    """Splits EXPERIENCE section text into one block per job (date-line anchor)."""

    def parse(self, text: str) -> list[ParsedEntry]:
        return _parse_dated_entries(text)


class EducationParser:
    """Splits EDUCATION section text into one block per degree (date-line anchor)."""

    def parse(self, text: str) -> list[ParsedEntry]:
        return _parse_dated_entries(text)


def _parse_dated_entries(text: str) -> list[ParsedEntry]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    date_indices = [
        index for index, line in enumerate(lines) if DATE_LINE_PATTERN.match(line)
    ]
    if not date_indices:
        return []

    entries: list[ParsedEntry] = []
    for pos, date_idx in enumerate(date_indices):
        next_date_idx = (
            date_indices[pos + 1] if pos + 1 < len(date_indices) else len(lines)
        )
        line1 = lines[date_idx - 2] if date_idx >= 2 else None
        line2 = lines[date_idx - 1] if date_idx >= 1 else None
        date_range = lines[date_idx]
        if pos + 1 < len(date_indices):
            desc_end = max(date_idx + 1, next_date_idx - 2)
        else:
            desc_end = len(lines)
        description_lines = lines[date_idx + 1 : desc_end]
        description = "\n".join(description_lines).strip()

        block_lines = [line for line in (line1, line2, date_range) if line]
        if description:
            block_lines.append(description)
        full_text = "\n".join(block_lines)

        entries.append(
            ParsedEntry(
                line1=line1,
                line2=line2,
                date_range=date_range,
                description=description,
                full_text=full_text,
            )
        )
    return entries


def split_by_size(
    text: str,
    max_size: int = DEFAULT_MAX_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split long text into overlapping chunks by character window."""
    normalized = text.strip()
    if not normalized:
        return []
    if len(normalized) <= max_size:
        return [normalized]

    pieces: list[str] = []
    start = 0
    while start < len(normalized):
        end = start + max_size
        piece = normalized[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return pieces


class CVChunker:
    """Builds retrieval chunks with metadata from extracted CV sections."""

    def __init__(
        self,
        *,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        experience_parser: ExperienceParser | None = None,
        education_parser: EducationParser | None = None,
    ) -> None:
        self._max_chunk_size = max_chunk_size
        self._chunk_overlap = chunk_overlap
        self._experience_parser = experience_parser or ExperienceParser()
        self._education_parser = education_parser or EducationParser()

    def chunk_sections(
        self,
        doc_id: str,
        candidate_name: str | None,
        sections: list[CVSection],
    ) -> list[CVChunk]:
        chunks: list[CVChunk] = []
        type_counters: dict[str, int] = {}

        for section in sections:
            body = section.text.strip()
            if not body:
                continue

            section_name = section.section_name
            if section_name == "EXPERIENCE":
                chunks.extend(
                    self._chunk_experience(doc_id, candidate_name, body, type_counters)
                )
            elif section_name == "EDUCATION":
                chunks.extend(
                    self._chunk_education(doc_id, candidate_name, body, type_counters)
                )
            else:
                chunk_type = SECTION_TO_CHUNK_TYPE.get(section_name, "other")
                chunks.extend(
                    self._chunk_plain_section(
                        doc_id=doc_id,
                        candidate_name=candidate_name,
                        section=section_name,
                        chunk_type=chunk_type,
                        text=body,
                        type_counters=type_counters,
                    )
                )
        return chunks

    def _chunk_experience(
        self,
        doc_id: str,
        candidate_name: str | None,
        text: str,
        type_counters: dict[str, int],
    ) -> list[CVChunk]:
        entries = self._experience_parser.parse(text)
        if not entries:
            return self._chunk_plain_section(
                doc_id=doc_id,
                candidate_name=candidate_name,
                section="EXPERIENCE",
                chunk_type="other",
                text=text,
                type_counters=type_counters,
            )

        chunks: list[CVChunk] = []
        for entry in entries:
            chunks.extend(
                self._chunks_from_entry_text(
                    doc_id=doc_id,
                    candidate_name=candidate_name,
                    section="EXPERIENCE",
                    chunk_type="experience_item",
                    text=entry.full_text,
                    type_counters=type_counters,
                    job_title=entry.line1,
                    company=entry.line2,
                    date_range=entry.date_range,
                )
            )
        return chunks

    def _chunk_education(
        self,
        doc_id: str,
        candidate_name: str | None,
        text: str,
        type_counters: dict[str, int],
    ) -> list[CVChunk]:
        entries = self._education_parser.parse(text)
        if not entries:
            return self._chunk_plain_section(
                doc_id=doc_id,
                candidate_name=candidate_name,
                section="EDUCATION",
                chunk_type="education_item",
                text=text,
                type_counters=type_counters,
            )

        chunks: list[CVChunk] = []
        for entry in entries:
            chunks.extend(
                self._chunks_from_entry_text(
                    doc_id=doc_id,
                    candidate_name=candidate_name,
                    section="EDUCATION",
                    chunk_type="education_item",
                    text=entry.full_text,
                    type_counters=type_counters,
                    degree=entry.line1,
                    institution=entry.line2,
                    date_range=entry.date_range,
                )
            )
        return chunks

    def _chunk_plain_section(
        self,
        doc_id: str,
        candidate_name: str | None,
        section: str,
        chunk_type: str,
        text: str,
        type_counters: dict[str, int],
    ) -> list[CVChunk]:
        return self._chunks_from_entry_text(
            doc_id=doc_id,
            candidate_name=candidate_name,
            section=section,
            chunk_type=chunk_type,
            text=text,
            type_counters=type_counters,
        )

    def _chunks_from_entry_text(
        self,
        doc_id: str,
        candidate_name: str | None,
        section: str,
        chunk_type: str,
        text: str,
        type_counters: dict[str, int],
        *,
        job_title: str | None = None,
        company: str | None = None,
        date_range: str | None = None,
        degree: str | None = None,
        institution: str | None = None,
    ) -> list[CVChunk]:
        pieces = split_by_size(
            text,
            max_size=self._max_chunk_size,
            overlap=self._chunk_overlap,
        )
        chunks: list[CVChunk] = []
        for piece in pieces:
            chunk_index = type_counters.get(chunk_type, 0)
            type_counters[chunk_type] = chunk_index + 1
            chunk_id = f"{doc_id}_{chunk_type}_{chunk_index}"
            chunks.append(
                CVChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    candidate_name=candidate_name,
                    section=section,
                    chunk_type=chunk_type,
                    chunk_index=chunk_index,
                    text=piece,
                    job_title=job_title,
                    company=company,
                    date_range=date_range,
                    degree=degree,
                    institution=institution,
                )
            )
        return chunks
