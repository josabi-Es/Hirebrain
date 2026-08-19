from __future__ import annotations

from rag.ingest.pipeline import format_sections_export
from rag.shared.models import CVSection

from test_chunking import KENNETH_EXPERIENCE


def test_format_sections_export_splits_experience_subblocks() -> None:
    sections = [
        CVSection(doc_id="cv_test", section_name="SUMMARY", text="Short summary."),
        CVSection(doc_id="cv_test", section_name="EXPERIENCE", text=KENNETH_EXPERIENCE),
    ]
    exported = format_sections_export(sections)

    assert "=== EXPERIENCE ===" in exported
    assert "--- 1 ---" in exported
    assert "--- 2 ---" in exported
    assert "--- 3 ---" in exported
    assert "UrbanGrid Mobility" in exported
    assert "BluePeak Consulting" in exported
    assert "Sterling Partners" in exported

    experience_block = exported.split("=== EXPERIENCE ===", maxsplit=1)[1].strip()
    first_job, remainder = experience_block.split("--- 2 ---", maxsplit=1)
    assert "BluePeak Consulting" not in first_job
    second_job, _ = remainder.split("--- 3 ---", maxsplit=1)
    assert "Sterling Partners" not in second_job
