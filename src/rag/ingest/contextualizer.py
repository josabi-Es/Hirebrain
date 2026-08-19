"""Chunk augmentation for contextual retrieval (text_to_embed)."""

from __future__ import annotations

from rag.shared.models import CVChunk

_SECTION_CONTEXT: dict[str, str] = {
    "EXPERIENCE": "The following text details the candidate's professional experience.",
    "EDUCATION": "The following text details the candidate's education.",
    "SKILLS": "The following text lists the candidate's skills.",
    "SUMMARY": "The following text is the candidate's professional summary.",
    "CONTACT_INFO": "The following text contains the candidate's contact information.",
}

_CHUNK_TYPE_CONTEXT: dict[str, str] = {
    "experience_item": "The following text details one work experience entry.",
    "education_item": "The following text details one education entry.",
    "skills_list": "The following text lists the candidate's skills.",
    "summary_chunk": "The following text is the candidate's professional summary.",
    "contact_info": "The following text contains the candidate's contact information.",
}


def _has_value(value: str | None) -> bool:
    return value is not None and value.strip() != ""


class ChunkContextualizer:
    """Builds augmented embedding text from chunk metadata without mutating chunk.text."""

    def contextualize_chunks(self, chunks: list[CVChunk]) -> list[CVChunk]:
        """Return chunks with text_to_embed populated; original text unchanged."""
        return [self.contextualize_chunk(chunk) for chunk in chunks]

    def contextualize_chunk(self, chunk: CVChunk) -> CVChunk:
        """Return a copy of chunk with text_to_embed set."""
        if not isinstance(chunk, CVChunk):
            raise TypeError(f"Expected CVChunk, got {type(chunk).__name__}")
        if not chunk.text.strip():
            raise ValueError(f"Chunk {chunk.chunk_id!r} has empty text")
        embed_text = self.build_text_to_embed(chunk)
        return chunk.model_copy(update={"text_to_embed": embed_text})

    def build_text_to_embed(self, chunk: CVChunk) -> str:
        """
        Build a readable prefix from metadata plus the original chunk body.

        Null or blank optional fields are omitted from the prefix.
        """
        if not isinstance(chunk, CVChunk):
            raise TypeError(f"Expected CVChunk, got {type(chunk).__name__}")
        if not chunk.text.strip():
            raise ValueError(f"Chunk {chunk.chunk_id!r} has empty text")

        prefix_parts: list[str] = []
        if _has_value(chunk.candidate_name):
            prefix_parts.append(f"Candidate: {chunk.candidate_name}")
        prefix_parts.append(f"Doc: {chunk.doc_id}")
        prefix_parts.append(f"Section: {chunk.section}")
        prefix_parts.append(f"Type: {chunk.chunk_type}")

        optional_labels: tuple[tuple[str, str | None], ...] = (
            ("Role", chunk.job_title),
            ("Company", chunk.company),
            ("Degree", chunk.degree),
            ("Institution", chunk.institution),
            ("Period", chunk.date_range),
        )
        for label, value in optional_labels:
            if _has_value(value):
                prefix_parts.append(f"{label}: {value}")

        context = self._resolve_context(chunk)
        prefix = ". ".join(prefix_parts) + f". Context: {context}"
        return f"{prefix}\n\nText: {chunk.text}"

    @staticmethod
    def _resolve_context(chunk: CVChunk) -> str:
        if chunk.chunk_type in _CHUNK_TYPE_CONTEXT:
            return _CHUNK_TYPE_CONTEXT[chunk.chunk_type]
        if chunk.section in _SECTION_CONTEXT:
            return _SECTION_CONTEXT[chunk.section]
        return "The following text contains relevant candidate information."
