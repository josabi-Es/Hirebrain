from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class CVDocument(BaseModel):
    """Represents extracted and normalized text for a CV document."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    doc_id: str
    candidate_name: str | None
    raw_text: str
    clean_text: str


class CVSection(BaseModel):
    """Represents one semantic section extracted from a CV."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    doc_id: str
    section_name: str
    text: str


class CVChunk(BaseModel):
    """Represents one retrieval-ready chunk with metadata for vector indexing."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    chunk_id: str
    doc_id: str
    candidate_name: str | None
    section: str
    chunk_type: str
    chunk_index: int
    text: str
    job_title: str | None = None
    company: str | None = None
    date_range: str | None = None
    degree: str | None = None
    institution: str | None = None
    text_to_embed: str | None = None

    def embedding_text(self) -> str:
        """Text sent to the embedding model (contextualized when available)."""
        if self.text_to_embed and self.text_to_embed.strip():
            return self.text_to_embed
        return self.text

    def to_vector_metadata(self) -> dict[str, str | int]:
        """Metadata serializable for vector stores (Chroma, Qdrant, etc.)."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "candidate_name": self.candidate_name or "",
            "section": self.section,
            "chunk_type": self.chunk_type,
            "chunk_index": self.chunk_index,
            "job_title": self.job_title or "",
            "company": self.company or "",
            "date_range": self.date_range or "",
            "degree": self.degree or "",
            "institution": self.institution or "",
        }
