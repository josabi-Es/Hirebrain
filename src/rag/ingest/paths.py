from __future__ import annotations

from pathlib import Path

from cv_creator.core.paths import cvs_dir, pdf_search_dirs


def resolve_pdf_path(base_dir: Path, filename: str) -> Path | None:
    """Resolve a PDF filename under the configured search directories."""
    for directory in pdf_search_dirs(base_dir):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def resolve_pdf_reference(reference: str, base_dir: Path | None = None) -> Path:
    """
    Resolve a user-provided PDF reference.

    - Absolute or relative existing paths are used as-is.
    - Bare filenames are searched under the project base, data/cvs/, and legacy output/.
    """
    base = (base_dir or Path.cwd()).resolve()
    candidate = Path(reference)

    if candidate.is_file():
        return candidate.resolve()

    if candidate.parent == Path("."):
        found = resolve_pdf_path(base, candidate.name)
        if found is not None:
            return found.resolve()

    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    searched = ", ".join(str(directory) for directory in pdf_search_dirs(base))
    raise FileNotFoundError(f"PDF no encontrado: {reference} (buscado en {searched})")


def default_input_dir(base_dir: Path | None = None) -> Path:
    """Default directory for batch PDF extraction (--input-dir)."""
    return cvs_dir(base_dir)
