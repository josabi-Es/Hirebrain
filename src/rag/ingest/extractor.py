from __future__ import annotations

from pathlib import Path

import pymupdf as fitz

from rag.ingest.normalizer import normalize_text
from rag.ingest.sectioner import CVSectioner
from rag.shared.models import CVDocument, CVSection


class CVExtractor:
    """Extracts text from PDF files and delegates sectioning to CVSectioner."""

    def __init__(self, sectioner: CVSectioner | None = None) -> None:
        self._sectioner = sectioner or CVSectioner()

    def extract(self, pdf_path: Path) -> tuple[CVDocument, list[CVSection]]:
        """Extract and segment a CV PDF into document and sections."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

        doc_id = pdf_path.stem
        pages_text = self._extract_pages_text(pdf_path)
        raw_text = "\n\n".join(text for text in pages_text if text.strip())
        clean_text = normalize_text(raw_text)

        sectioning = self._sectioner.sectionize(clean_text, doc_id)

        document = CVDocument(
            doc_id=doc_id,
            candidate_name=sectioning.candidate_name,
            raw_text=raw_text,
            clean_text=clean_text,
        )
        return document, sectioning.sections

    def _extract_pages_text(self, pdf_path: Path) -> list[str]:
        pages_text: list[str] = []
        with fitz.open(str(pdf_path)) as pymupdf_doc:
            for pymupdf_page in pymupdf_doc:
                pages_text.append(self._extract_page_with_pymupdf(pymupdf_page))
        return pages_text

    @staticmethod
    def _extract_page_with_pymupdf(page: fitz.Page) -> str:
        text = page.get_text("text")
        return text or ""
