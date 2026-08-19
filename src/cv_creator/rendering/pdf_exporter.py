"""HTML to PDF export via WeasyPrint."""

from __future__ import annotations

from pathlib import Path

from cv_creator.core.exceptions import PdfRenderError


def export_pdf(html: str, output_path: Path) -> Path:
    """Convert HTML string to a PDF file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from weasyprint import HTML
    except ImportError as e:
        raise PdfRenderError(
            "WeasyPrint is not installed. Run: uv sync"
        ) from e
    except OSError as e:
        raise PdfRenderError(
            f"WeasyPrint system libraries unavailable: {e}. "
            "See https://doc.courtbouillon.org/weasyprint/stable/first_steps.html"
        ) from e

    try:
        HTML(string=html, base_url=str(output_path.parent)).write_pdf(str(output_path))
    except Exception as e:
        raise PdfRenderError(f"PDF generation failed: {e}") from e

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise PdfRenderError("PDF file is empty or was not created")

    return output_path
