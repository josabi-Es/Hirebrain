from __future__ import annotations

from rag.ingest.chunker import CVChunker, ExperienceParser, split_by_size
from rag.shared.models import CVSection

KENNETH_EXPERIENCE = """Junior Financial Analyst
UrbanGrid Mobility
08/2025 – Present
Utilized SQL and Power BI to analyze complex financial data sets.
Junior Financial Analyst
BluePeak Consulting
05/2023 – 02/2025
Applied Excel and ERP Systems expertise to analyze financial performance.
Junior Financial Analyst
Sterling Partners
02/2021 – 04/2023
Utilized Power BI and SQL to analyze large datasets."""

BRIAN_PARK_EXPERIENCE_TAIL = """Key Account Manager
BluePeak Consulting
12/2024 – Present
Utilized CRM expertise to optimize sales pipelines and drive revenue growth. Developed and
executed customized solution plans to meet client needs, resulting in significant upsells and
cross-sells. Fostered strong relationships with key stakeholders, ensuring seamless project
delivery and exceeding client expectations.
Senior Account Executive
CloudNine Systems
03/2021 – 06/2024
Applied Contract Management skills to ensure successful deal closures and manage complex
sales processes."""


def test_experience_parser_splits_three_jobs() -> None:
    entries = ExperienceParser().parse(KENNETH_EXPERIENCE)
    assert len(entries) == 3
    assert entries[0].line1 == "Junior Financial Analyst"
    assert entries[0].line2 == "UrbanGrid Mobility"
    assert entries[0].date_range == "08/2025 – Present"
    assert "SQL and Power BI" in entries[0].full_text
    assert entries[0].description == (
        "Utilized SQL and Power BI to analyze complex financial data sets."
    )
    assert "BluePeak Consulting" not in entries[0].description
    assert entries[1].description == (
        "Applied Excel and ERP Systems expertise to analyze financial performance."
    )
    assert "Sterling Partners" not in entries[1].description
    assert "BluePeak Consulting" not in entries[0].full_text


def test_chunker_produces_experience_items_and_metadata() -> None:
    sections = [
        CVSection(doc_id="cv_test", section_name="SUMMARY", text="Short summary."),
        CVSection(doc_id="cv_test", section_name="EXPERIENCE", text=KENNETH_EXPERIENCE),
        CVSection(doc_id="cv_test", section_name="SKILLS", text="SQL\nPower BI"),
    ]
    chunks = CVChunker().chunk_sections(
        doc_id="cv_test",
        candidate_name="Test User",
        sections=sections,
    )

    experience_chunks = [c for c in chunks if c.chunk_type == "experience_item"]
    assert len(experience_chunks) == 3
    assert experience_chunks[0].job_title == "Junior Financial Analyst"
    assert experience_chunks[0].company == "UrbanGrid Mobility"
    assert experience_chunks[0].section == "EXPERIENCE"
    assert "BluePeak Consulting" not in experience_chunks[0].text
    assert "Sterling Partners" not in experience_chunks[1].text

    metadata = experience_chunks[0].to_vector_metadata()
    assert metadata["doc_id"] == "cv_test"
    assert metadata["chunk_type"] == "experience_item"
    assert metadata["company"] == "UrbanGrid Mobility"

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_experience_parser_no_bleed_between_jobs() -> None:
    entries = ExperienceParser().parse(BRIAN_PARK_EXPERIENCE_TAIL)
    assert len(entries) == 2
    assert "CloudNine Systems" not in entries[0].description
    assert "Senior Account Executive" not in entries[0].description
    assert entries[1].line2 == "CloudNine Systems"


def test_experience_fallback_when_no_dates() -> None:
    sections = [
        CVSection(
            doc_id="cv_test",
            section_name="EXPERIENCE",
            text="Unstructured experience block without dates.",
        ),
    ]
    chunks = CVChunker().chunk_sections(
        doc_id="cv_test",
        candidate_name=None,
        sections=sections,
    )
    assert len(chunks) == 1
    assert chunks[0].section == "EXPERIENCE"
    assert chunks[0].chunk_type == "other"


def test_split_by_size_overlap() -> None:
    text = "a" * 1000
    pieces = split_by_size(text, max_size=400, overlap=50)
    assert len(pieces) >= 2
    assert all(len(piece) <= 400 for piece in pieces)
