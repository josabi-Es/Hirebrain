"""Project-wide default paths (overridable via environment variables)."""

from __future__ import annotations

import os
from pathlib import Path

# Corpus: synthetic PDF CVs (cv-creator output, cv-extract input)
DEFAULT_CVS_DIR = Path(os.getenv("CV_OUTPUT_DIR", "data/cvs"))

# RAG pipeline derivatives (safe to delete and regenerate)
DEFAULT_RAG_CLEAN_TEXT_DIR = Path(
    os.getenv("RAG_CLEAN_TEXT_DIR", "artifacts/rag/clean_text")
)
DEFAULT_RAG_CHUNKS_DIR = Path(os.getenv("RAG_CHUNKS_DIR", "artifacts/rag/chunks"))

# Hugging Face portrait cache
DEFAULT_PHOTO_CACHE_DIR = Path(os.getenv("HF_PHOTO_CACHE_DIR", ".cache/photos"))

# Pre-refactor layout; used only as fallback when resolving PDF filenames
LEGACY_CVS_DIR = Path("output")


def resolve_under_base(base_dir: Path, configured: Path) -> Path:
    """Resolve a path that may be relative to the project base directory."""
    if configured.is_absolute():
        return configured.resolve()
    return (base_dir / configured).resolve()


def cvs_dir(base_dir: Path | None = None) -> Path:
    """Directory where generated CV PDFs are stored."""
    base = (base_dir or Path.cwd()).resolve()
    return resolve_under_base(base, DEFAULT_CVS_DIR)


def pdf_search_dirs(base_dir: Path) -> tuple[Path, ...]:
    """Directories searched when resolving a bare PDF filename."""
    base = base_dir.resolve()
    candidates = [
        base,
        resolve_under_base(base, DEFAULT_CVS_DIR),
        resolve_under_base(base, LEGACY_CVS_DIR),
    ]
    seen: set[Path] = set()
    ordered: list[Path] = []
    for directory in candidates:
        if directory not in seen:
            seen.add(directory)
            ordered.append(directory)
    return tuple(ordered)
