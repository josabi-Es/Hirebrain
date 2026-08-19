"""Unit tests for simplified Pydantic models."""

from cv_creator.core.models import CVData, EducationEntry, ExperienceEntry, GenerationConfig


def test_cv_data_shape():
    cv = CVData(
        full_name="Alex Morgan",
        email="alex@example.com",
        gender="male",
        phone="+1 555-0100",
        location="Austin, TX",
        summary="Analyst with strong SQL skills.",
        education=[
            EducationEntry(
                degree="B.S. in Statistics",
                institution="State University",
                start_year=2016,
                start_month=9,
                year_end=2020,
                end_month=6,
            )
        ],
        experience=[
            ExperienceEntry(
                job_title="Data Analyst",
                company="DataFlow Inc",
                start_year=2021,
                start_month=1,
                end_year=None,
                end_month=None,
                description="Built reporting dashboards used by finance teams.",
            )
        ],
        skills=["SQL", "Python"],
    )
    assert cv.education[0].degree == "B.S. in Statistics"
    assert cv.experience[0].job_title == "Data Analyst"
    assert cv.skills == ["SQL", "Python"]


def test_generation_config_defaults():
    cfg = GenerationConfig()
    assert cfg.locale == "en_US"
