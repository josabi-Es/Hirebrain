from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from cv_creator.core.paths import DEFAULT_RAG_CHUNKS_DIR, DEFAULT_RAG_CLEAN_TEXT_DIR
from rag.ingest.chunker import CVChunker, ExperienceParser
from rag.ingest.extractor import CVExtractor
from rag.ingest.paths import resolve_pdf_reference
from rag.ingest.sectioner import CVSectioner
from rag.ingest.contextualizer import ChunkContextualizer
from rag.shared.models import CVChunk, CVDocument, CVSection


class ExtractionResult(BaseModel):
    """Result of extracting one CV PDF."""

    model_config = ConfigDict(extra="forbid")

    document: CVDocument
    sections: list[CVSection]
    chunks: list[CVChunk]
    source_path: Path


def resolve_pdf_paths(
    positional: list[str],
    input_dir: Path | None,
    base_dir: Path | None = None,
) -> list[Path]:
    """Collect unique PDF paths from positional args and optional input directory."""
    base = (base_dir or Path.cwd()).resolve()
    resolved: list[Path] = []
    seen: set[Path] = set()

    for reference in positional:
        path = resolve_pdf_reference(reference, base_dir=base)
        key = path.resolve()
        if key not in seen:
            seen.add(key)
            resolved.append(key)

    if input_dir is not None:
        directory = input_dir.resolve()
        if not directory.is_dir():
            raise NotADirectoryError(f"No es un directorio: {directory}")
        for pdf_path in sorted(directory.glob("*.pdf")):
            key = pdf_path.resolve()
            if key not in seen:
                seen.add(key)
                resolved.append(key)

    return resolved


def contextualize_chunks(
    chunks: list[CVChunk],
    contextualizer: ChunkContextualizer | None = None,
) -> list[CVChunk]:
    """Augment chunks with text_to_embed; original text is preserved."""
    if not chunks:
        return []
    active = contextualizer or ChunkContextualizer()
    return active.contextualize_chunks(chunks)


def run_extraction(
    pdf_path: Path,
    chunker: CVChunker | None = None,
    *,
    contextualize: bool = True,
    contextualizer: ChunkContextualizer | None = None,
) -> ExtractionResult:
    """Extract, segment, and chunk a single CV PDF."""
    document, sections = CVExtractor().extract(pdf_path.resolve())
    active_chunker = chunker or CVChunker()
    chunks = active_chunker.chunk_sections(
        doc_id=document.doc_id,
        candidate_name=document.candidate_name,
        sections=sections,
    )
    if contextualize:
        chunks = contextualize_chunks(chunks, contextualizer=contextualizer)
    return ExtractionResult(
        document=document,
        sections=sections,
        chunks=chunks,
        source_path=pdf_path.resolve(),
    )


def run_batch(paths: list[Path]) -> list[ExtractionResult]:
    """Extract and segment multiple CV PDFs."""
    return [run_extraction(path) for path in paths]


DEFAULT_CLEAN_TEXT_DIR = DEFAULT_RAG_CLEAN_TEXT_DIR


def order_sections_for_export(sections: list[CVSection]) -> list[CVSection]:
    """Order sections for export: CONTACT_INFO first, then canonical section order."""
    order_index = {
        name: index for index, name in enumerate(CVSectioner.CANONICAL_SECTION_ORDER)
    }

    contact_sections = [s for s in sections if s.section_name == "CONTACT_INFO"]
    other_sections = [s for s in sections if s.section_name != "CONTACT_INFO"]
    other_sections.sort(
        key=lambda section: (
            order_index.get(section.section_name, len(CVSectioner.CANONICAL_SECTION_ORDER)),
            section.section_name,
        )
    )
    return contact_sections + other_sections


def _format_experience_section_for_export(text: str) -> str:
    """Render EXPERIENCE body as numbered sub-blocks (one per job)."""
    entries = ExperienceParser().parse(text)
    if not entries:
        return text
    sub_blocks = [
        f"--- {index} ---\n{entry.full_text}" for index, entry in enumerate(entries, start=1)
    ]
    return "\n\n".join(sub_blocks)


def format_sections_export(sections: list[CVSection]) -> str:
    """Build debug export text from structured CV sections."""
    ordered = order_sections_for_export(sections)
    blocks: list[str] = []

    for section in ordered:
        body = section.text.strip()
        if not body:
            continue
        if section.section_name == "EXPERIENCE":
            body = _format_experience_section_for_export(body)
        blocks.append(f"=== {section.section_name} ===\n{body}")

    return "\n\n".join(blocks)


def export_clean_text_files(
    results: list[ExtractionResult],
    output_dir: Path | None = None,
) -> list[Path]:
    """Write structured section exports to .txt files under output_dir."""
    target_dir = (output_dir or DEFAULT_CLEAN_TEXT_DIR).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for result in results:
        file_path = target_dir / f"{result.document.doc_id}.txt"
        file_path.write_text(
            format_sections_export(result.sections),
            encoding="utf-8",
        )
        written.append(file_path)
    return written


def format_console(result: ExtractionResult) -> str:
    """Format one extraction result for human-readable console output."""
    lines: list[str] = [
        f"\n{'=' * 72}",
        f"Archivo: {result.source_path}",
        f"Doc ID: {result.document.doc_id}",
        f"Candidate Name: {result.document.candidate_name}",
        f"Secciones exportables: {len(result.sections)}",
        "",
        "=== CONTACT INFO ===",
    ]

    contact = next(
        (section for section in result.sections if section.section_name == "CONTACT_INFO"),
        None,
    )
    if contact is None:
        lines.append("No contact info detectada.")
    else:
        lines.append(contact.text)

    lines.append("")
    lines.append("=== SECTIONS ===")
    for section in result.sections:
        preview = section.text[:140].replace("\n", " ")
        lines.append(f"- {section.section_name}: {preview}")

    lines.append("")
    lines.append(f"=== CHUNKS ({len(result.chunks)}) ===")
    for chunk in result.chunks:
        label = chunk.chunk_type
        if chunk.company:
            label = f"{label} @ {chunk.company}"
        preview = chunk.text[:100].replace("\n", " ")
        lines.append(f"- {chunk.chunk_id} [{label}]: {preview}")

    return "\n".join(lines)


def format_json(results: list[ExtractionResult]) -> str:
    """Serialize batch extraction results to JSON."""
    payload = [
        {
            "source_path": str(result.source_path),
            "doc_id": result.document.doc_id,
            "candidate_name": result.document.candidate_name,
            "sections": [
                {
                    "section_name": section.section_name,
                    "text": section.text,
                }
                for section in result.sections
            ],
            "chunks": [_chunk_to_dict(chunk) for chunk in result.chunks],
        }
        for result in results
    ]
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _chunk_from_dict(data: dict[str, str | int | None]) -> CVChunk:
    """Deserialize chunk from JSON export."""
    return CVChunk.model_validate(data)


def load_chunks_from_json(path: Path) -> list[CVChunk]:
    """Load CVChunk list from a single *.chunks.json artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_chunks = payload.get("chunks", [])
    if not isinstance(raw_chunks, list):
        raise ValueError(f"Invalid chunks file (missing chunks list): {path}")
    return [_chunk_from_dict(item) for item in raw_chunks]


def load_chunks_from_directory(directory: Path) -> list[CVChunk]:
    """Load and merge chunks from all *.chunks.json files in a directory."""
    resolved = directory.resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"Not a directory: {resolved}")

    chunks: list[CVChunk] = []
    for json_path in sorted(resolved.glob("*.chunks.json")):
        chunks.extend(load_chunks_from_json(json_path))
    return chunks


def _chunk_to_dict(chunk: CVChunk) -> dict[str, str | int | None]:
    """Serialize chunk for JSON export (no duplicate vector_metadata nesting)."""
    data: dict[str, str | int | None] = {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "candidate_name": chunk.candidate_name,
        "section": chunk.section,
        "chunk_type": chunk.chunk_type,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "job_title": chunk.job_title,
        "company": chunk.company,
        "date_range": chunk.date_range,
        "degree": chunk.degree,
        "institution": chunk.institution,
    }
    if chunk.text_to_embed:
        data["text_to_embed"] = chunk.text_to_embed
    return data


def format_chunks_json(results: list[ExtractionResult]) -> str:
    """Serialize only chunks from batch extraction results."""
    payload = [
        {
            "source_path": str(result.source_path),
            "doc_id": result.document.doc_id,
            "candidate_name": result.document.candidate_name,
            "chunks": [_chunk_to_dict(chunk) for chunk in result.chunks],
        }
        for result in results
    ]
    return json.dumps(payload, indent=2, ensure_ascii=False)


DEFAULT_CHUNKS_DIR = DEFAULT_RAG_CHUNKS_DIR


def format_chunks_export(chunks: list[CVChunk]) -> str:
    """Build human-readable debug export from CV chunks."""
    blocks: list[str] = []
    for chunk in chunks:
        header_parts = [chunk.chunk_id, chunk.chunk_type, chunk.section]
        if chunk.company:
            header_parts.append(chunk.company)
        elif chunk.institution:
            header_parts.append(chunk.institution)
        header = " | ".join(header_parts)
        blocks.append(f"=== {header} ===\n{chunk.text.strip()}")
    return "\n\n".join(blocks)


def export_chunk_files(
    results: list[ExtractionResult],
    output_dir: Path | None = None,
) -> list[Path]:
    """Write per-CV chunk JSON files under output_dir."""
    target_dir = (output_dir or DEFAULT_CHUNKS_DIR).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for result in results:
        file_path = target_dir / f"{result.document.doc_id}.chunks.json"
        payload = {
            "doc_id": result.document.doc_id,
            "candidate_name": result.document.candidate_name,
            "source_path": str(result.source_path),
            "chunks": [_chunk_to_dict(chunk) for chunk in result.chunks],
        }
        file_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written.append(file_path)
    return written
