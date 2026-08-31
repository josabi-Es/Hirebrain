"""Regression tests for single LOCATION extraction in CONTACT_INFO."""

from __future__ import annotations

import re

import pytest

from rag.ingest.sectioner import CVSectioner


def _contact_text(clean_text: str) -> str | None:
    result = CVSectioner().sectionize(clean_text, doc_id="cv_test")
    contact = next(
        (section for section in result.sections if section.section_name == "CONTACT_INFO"),
        None,
    )
    return contact.text if contact else None


def _candidate_name(clean_text: str) -> str | None:
    return CVSectioner().sectionize(clean_text, doc_id="cv_test").candidate_name


def _location_lines(contact_text: str | None) -> list[str]:
    if not contact_text:
        return []
    return [line for line in contact_text.splitlines() if line.startswith("LOCATION:")]


KIMBERLY_HEADER = """\
Kimberly Anderson
Solutions Architect
kimberly.anderson@hotmail.com
+1-548-536-2217
South Jay, Barbados
Microservices, Kubernetes
SUMMARY
Seasoned Solutions Architect with a strong track record.
"""

HEATHER_HEADER = """\
Heather Jordan
Full Stack Developer
heather.jordan@yahoo.com
001-254-386-6906
West Sherrifort, Saint Kitts
Django, Git
mid-level, and senior roles.
Developed scalable microservices-based applications using Django and PostgreSQL,
SUMMARY
Highly motivated and detail-oriented developer.
"""

BENJAMIN_HEADER = """\
Benjamin Hamilton
Senior Software Engineer
benjamin.hamilton@gmail.com
444.881.7278x215
Fowlertown, Marshall Islands
Microservices, Docker
Python, React
SUMMARY
Highly experienced software engineer with 8+ years of expertise.
"""

JILL_HEADER = """\
Jill Carter
Machine Learning Engineer
jill.carter@yahoo.com
716.511.3414x9032
Caroltown, Antigua
Statistics, Data Visualization
Pandas, Apache
Spark, NumPy, and Scikit-learn.
SUMMARY
Results-driven Machine Learning Engineer.
"""

ANDREW_LAWRENCE_HEADER = """\
Andrew Lawrence
Software Engineer
andrew.lawrence@example.com
555.123.4567
Rowlandshire, Pitcairn Islands
Go, System Design
SUMMARY
Experienced engineer.
"""

BRANDON_HEADER = """\
Brandon
Sampson
Lake Jonathanbury,
Malaysia
brandon.sampson@gmail.com
001-741-824-1578x6686
SKILLS
REST APIs
SUMMARY
Highly skilled Site Reliability Engineer.
"""


@pytest.mark.parametrize(
    ("clean_text", "expected_location"),
    [
        (KIMBERLY_HEADER, "South Jay, Barbados"),
        (HEATHER_HEADER, "West Sherrifort, Saint Kitts"),
        (BENJAMIN_HEADER, "Fowlertown, Marshall Islands"),
        (JILL_HEADER, "Caroltown, Antigua"),
        (ANDREW_LAWRENCE_HEADER, "Rowlandshire, Pitcairn Islands"),
    ],
)
def test_single_valid_location_in_contact(clean_text: str, expected_location: str) -> None:
    contact = _contact_text(clean_text)
    assert contact is not None
    locations = _location_lines(contact)
    assert len(locations) == 1
    assert locations[0] == f"LOCATION: {expected_location}"


@pytest.mark.parametrize(
    "clean_text",
    [KIMBERLY_HEADER, HEATHER_HEADER, BENJAMIN_HEADER, JILL_HEADER, ANDREW_LAWRENCE_HEADER],
)
def test_contact_info_excludes_skill_and_narrative_false_positives(clean_text: str) -> None:
    contact = _contact_text(clean_text)
    assert contact is not None
    forbidden_fragments = [
        "Microservices",
        "Kubernetes",
        "Django, Git",
        "mid-level",
        "Developed scalable",
        "Data Visualization",
        "Pandas, Apache",
        "Go, System Design",
        "Python, React",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in contact


def test_no_location_when_header_has_only_skills() -> None:
    clean_text = """\
Jane Doe
Software Engineer
jane@example.com
555-0100
Python, React
SUMMARY
Great engineer.
"""
    contact = _contact_text(clean_text)
    assert contact is not None
    assert _location_lines(contact) == []


def test_location_not_inferred_from_summary_body() -> None:
    clean_text = """\
Jane Doe
jane@example.com
SUMMARY
Based in Austin, Texas with 5 years of experience using Python, Django.
"""
    contact = _contact_text(clean_text)
    assert contact is not None
    assert _location_lines(contact) == []


def test_contact_info_includes_email_and_phone() -> None:
    contact = _contact_text(KIMBERLY_HEADER)
    assert contact is not None
    assert "EMAIL: kimberly.anderson@hotmail.com" in contact
    assert re.search(r"PHONE: \+1-548-536-2217", contact)


def test_split_location_city_and_country_on_separate_lines() -> None:
    contact = _contact_text(BRANDON_HEADER)
    assert contact is not None
    locations = _location_lines(contact)
    assert len(locations) == 1
    assert locations[0] == "LOCATION: Lake Jonathanbury, Malaysia"
    assert "Malaysia" not in contact.split("LOCATION:", 1)[0]


def test_candidate_name_detected_when_name_is_split_across_lines() -> None:
    assert _candidate_name(BRANDON_HEADER) == "Brandon Sampson"


def test_candidate_name_skips_role_title_when_name_appears_later() -> None:
    clean_text = """\
Senior Software Engineer
John Doe
john.doe@example.com
555-0100
Austin, Texas
SUMMARY
Great engineer.
"""
    assert _candidate_name(clean_text) == "John Doe"
