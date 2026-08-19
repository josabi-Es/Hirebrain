"""Regression tests for experience seniority vs total career years."""

from __future__ import annotations

from collections import Counter

from cv_creator.core.models import ExperienceEntry, GenerationConfig
from cv_creator.generators.pools import DOMAIN_POOLS, role_level
from cv_creator.generators.seniority import (
    ENTRY_ROLE_MAX_LEVEL,
    career_years,
    is_junior_title,
    min_level_for_career_years,
    tenure_years_at_role_start,
)
from cv_creator.generators.skeleton import build_candidate_skeleton

DATA_SCIENCE_TITLES = set(DOMAIN_POOLS["Data Science"]["job_titles"])


def _has_junior_title(experience: list[ExperienceEntry]) -> bool:
    return any(is_junior_title(exp.job_title) for exp in experience)


def _levels_monotonic_increasing(experience: list[ExperienceEntry]) -> bool:
    """Oldest → newest: seniority should not decrease."""
    levels = [role_level(exp.job_title) for exp in reversed(experience)]
    for idx in range(1, len(levels)):
        if levels[idx] < levels[idx - 1]:
            return False
    return True


def _is_data_science_cv(experience: list[ExperienceEntry]) -> bool:
    return any(exp.job_title in DATA_SCIENCE_TITLES for exp in experience)


def test_min_level_thresholds():
    assert min_level_for_career_years(4.9) == 0
    assert min_level_for_career_years(5.0) == 1
    assert min_level_for_career_years(7.9) == 1
    assert min_level_for_career_years(8.0) == 2
    assert min_level_for_career_years(12.0) == 2


def test_tenure_at_first_role_is_zero():
    cv = build_candidate_skeleton(GenerationConfig(seed=42))
    assert tenure_years_at_role_start(cv.experience, len(cv.experience) - 1) == 0.0


def test_role_level_overrides():
    assert role_level("Junior Software Engineer") == 0
    assert role_level("Software Engineer") == 1
    assert role_level("Senior Software Engineer") == 2
    assert role_level("Solutions Architect") == 3


def test_first_role_never_above_entry_cap():
    for seed in range(500):
        cv = build_candidate_skeleton(GenerationConfig(seed=seed))
        first = cv.experience[-1]
        level = role_level(first.job_title)
        assert level <= ENTRY_ROLE_MAX_LEVEL, (
            f"seed={seed} first={first.job_title!r} level={level}"
        )


def test_first_role_never_manager_or_lead():
    for seed in range(500):
        cv = build_candidate_skeleton(GenerationConfig(seed=seed))
        assert role_level(cv.experience[-1].job_title) < 3, (
            f"seed={seed} first={cv.experience[-1].job_title!r}"
        )


def test_no_duplicate_titles_in_same_cv():
    for seed in range(300):
        cv = build_candidate_skeleton(GenerationConfig(seed=seed))
        titles = [exp.job_title for exp in cv.experience]
        assert len(titles) == len(set(titles)), f"seed={seed} titles={titles}"


def test_no_junior_roles_after_five_years_of_tenure():
    """Junior is allowed on the first job; not on roles that start 5+ years into the career."""
    violations: list[tuple[int, str, str, float]] = []
    for seed in range(500):
        cv = build_candidate_skeleton(GenerationConfig(seed=seed))
        for idx, exp in enumerate(cv.experience):
            tenure = tenure_years_at_role_start(cv.experience, idx)
            if tenure >= 5.0 and is_junior_title(exp.job_title):
                violations.append((seed, exp.job_title, cv.experience[0].job_title, tenure))
    assert not violations, f"Junior role after 5y tenure: {violations[:5]}"


def test_recent_role_senior_plus_when_career_at_least_twelve_years():
    for seed in range(500):
        cv = build_candidate_skeleton(GenerationConfig(seed=seed))
        years = career_years(cv.experience)
        if years >= 12.0:
            assert role_level(cv.experience[0].job_title) >= 2, (
                f"seed={seed} recent={cv.experience[0].job_title!r} years={years:.1f}"
            )


def test_career_progression_non_decreasing():
    for seed in range(200):
        cv = build_candidate_skeleton(GenerationConfig(seed=seed))
        assert _levels_monotonic_increasing(cv.experience), (
            f"seed={seed} titles={[e.job_title for e in cv.experience]}"
        )


def test_short_career_may_include_junior():
    found_junior = False
    for seed in range(300):
        cv = build_candidate_skeleton(GenerationConfig(seed=seed))
        years = career_years(cv.experience)
        if years < 5.0 and _has_junior_title(cv.experience):
            found_junior = True
            break
    assert found_junior, "Expected at least one early-career CV with a Junior title"


def test_data_science_analytics_lead_not_dominant():
    """Recent-role Analytics Lead should not exceed ~40% of long Data Science careers."""
    analytics_lead_count = 0
    data_science_long = 0
    for seed in range(500):
        cv = build_candidate_skeleton(GenerationConfig(seed=seed))
        years = career_years(cv.experience)
        if years < 8.0 or not _is_data_science_cv(cv.experience):
            continue
        data_science_long += 1
        if cv.experience[0].job_title == "Analytics Lead":
            analytics_lead_count += 1

    if data_science_long == 0:
        return

    ratio = analytics_lead_count / data_science_long
    assert ratio <= 0.40, (
        f"Analytics Lead in {analytics_lead_count}/{data_science_long} "
        f"({ratio:.0%}) recent Data Science roles"
    )


def test_data_science_recent_title_diversity():
    """Most common recent title should not appear in >50% of long Data Science CVs."""
    recent_titles: list[str] = []
    for seed in range(500):
        cv = build_candidate_skeleton(GenerationConfig(seed=seed))
        years = career_years(cv.experience)
        if years < 8.0 or not _is_data_science_cv(cv.experience):
            continue
        recent_titles.append(cv.experience[0].job_title)

    if not recent_titles:
        return

    _title, top_count = Counter(recent_titles).most_common(1)[0]
    assert top_count / len(recent_titles) <= 0.50, (
        f"Top recent title {_title!r} appears in {top_count}/{len(recent_titles)} CVs"
    )
