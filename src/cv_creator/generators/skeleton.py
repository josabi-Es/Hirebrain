"""Phase A: deterministic candidate skeleton from minimal pools."""

from __future__ import annotations

import random
from datetime import date
from typing import Sequence

from faker import Faker

from cv_creator.core.models import CVData, EducationEntry, ExperienceEntry, GenerationConfig
from cv_creator.generators.pools import COMPANIES, DEGREES, DOMAIN_POOLS, DOMAINS, role_level
from cv_creator.generators.seniority import (
    ENTRY_ROLE_MAX_LEVEL,
    career_years,
    max_level_for_most_recent,
    min_level_for_most_recent,
    min_level_for_role_at_tenure,
    tenure_years_at_role_start,
)

CURRENT_YEAR = date.today().year
MIN_YEAR = 1995


def _rng_for_config(config: GenerationConfig) -> random.Random:
    return random.Random(config.seed)


def _to_month_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def _from_month_index(index: int) -> tuple[int, int]:
    return index // 12, (index % 12) + 1


def _pick_title(
    rng: random.Random,
    domain_titles: Sequence[str],
    *,
    min_level: int = 0,
    max_level: int = 4,
    exclude_titles: set[str] | None = None,
) -> str:
    excluded = exclude_titles or set()

    def _filter(allow_repeat: bool) -> list[str]:
        return [
            title
            for title in domain_titles
            if min_level <= role_level(title) <= max_level
            and (allow_repeat or title not in excluded)
        ]

    candidates = _filter(allow_repeat=False)
    if not candidates:
        candidates = _filter(allow_repeat=True)
    if not candidates:
        candidates = [
            title for title in domain_titles if role_level(title) >= min_level
        ]
    if not candidates:
        candidates = list(domain_titles)
    return rng.choice(candidates)


def _pick_company(
    rng: random.Random,
    *,
    seen_companies: set[str],
    consecutive_company: str | None = None,
) -> tuple[str, bool]:
    """
    Pick company following business rules.
    Returns (company, is_internal_move).
    """
    if consecutive_company and rng.random() < 0.2:
        return consecutive_company, True

    available = [company for company in COMPANIES if company not in seen_companies]
    if not available:
        available = COMPANIES
    return rng.choice(available), False


def _assign_experience_titles(
    rng: random.Random,
    experience: list[ExperienceEntry],
    domain_titles: Sequence[str],
) -> list[ExperienceEntry]:
    """
    Assign job titles after the timeline exists.

    experience[0] is most recent; experience[-1] is oldest.
    Entry roles are capped at junior/mid; recent roles honor total career years.
    """
    if not experience:
        return experience

    total_years = career_years(experience, current_year=CURRENT_YEAR)
    min_recent = min_level_for_most_recent(total_years)

    assigned_levels: list[int] = []
    used_titles: set[str] = set()
    updated: list[ExperienceEntry] = []

    # Oldest → newest: monotonic progression, tenure-aware mins, +1 level steps.
    for index in range(len(experience) - 1, -1, -1):
        exp = experience[index]
        is_oldest = index == len(experience) - 1
        is_most_recent = index == 0
        tenure = tenure_years_at_role_start(experience, index)

        if is_oldest:
            min_level = min_level_for_role_at_tenure(tenure)
            max_level = ENTRY_ROLE_MAX_LEVEL
        else:
            older_level = assigned_levels[0]
            tenure_min = min_level_for_role_at_tenure(tenure)
            min_level = max(older_level, tenure_min)
            max_level = older_level + 1

        if is_most_recent:
            prior_level = assigned_levels[0] if assigned_levels else ENTRY_ROLE_MAX_LEVEL
            min_level = max(min_level, min_recent, prior_level)
            max_level = max_level_for_most_recent(total_years, prior_level=prior_level)
            if max_level < min_level:
                max_level = min_level

        title = _pick_title(
            rng,
            domain_titles,
            min_level=min_level,
            max_level=max_level,
            exclude_titles=used_titles,
        )
        level = role_level(title)
        used_titles.add(title)
        assigned_levels.insert(0, level)
        updated.insert(
            0,
            exp.model_copy(update={"job_title": title}),
        )

    return updated


def build_candidate_skeleton(config: GenerationConfig) -> CVData:
    """Build structural CV data with strict backward chronology."""
    rng = _rng_for_config(config)
    if config.seed is not None:
        Faker.seed(config.seed)
    fake = Faker(config.locale)
    if config.seed is not None:
        fake.seed_instance(config.seed)

    gender = rng.choice(["male", "female"])
    first = fake.first_name_male() if gender == "male" else fake.first_name_female()
    last = fake.last_name()
    full_name = f"{first} {last}"
    local = f"{first.lower()}.{last.lower()}".replace("'", "")
    domain = fake.free_email_domain() or "example.com"
    email = f"{local}@{domain}"
    phone = fake.phone_number()
    location = f"{fake.city()}, {fake.country()}"

    selected_domain = rng.choice(DOMAINS)
    domain_catalog = DOMAIN_POOLS[selected_domain]
    domain_titles = domain_catalog["job_titles"]
    domain_skills = domain_catalog["skills"]

    num_exp = rng.randint(2, 4)
    most_recent_start_year = CURRENT_YEAR - rng.randint(0, 3)
    most_recent_start_month = rng.randint(1, 12)
    experience: list[ExperienceEntry] = []
    seen_companies: set[str] = set()

    for index in range(num_exp):
        if index == 0:
            company, _ = _pick_company(rng, seen_companies=seen_companies)
            start_year = most_recent_start_year
            start_month = most_recent_start_month
            end_year = None
            end_month = None
        else:
            previous_role = experience[index - 1]
            previous_start_index = _to_month_index(
                previous_role.start_year,
                previous_role.start_month,
            )

            company, is_internal_move = _pick_company(
                rng,
                seen_companies=seen_companies,
                consecutive_company=previous_role.company,
            )

            gap_months = 0 if is_internal_move else rng.randint(0, 6)
            end_month_index = previous_start_index - (gap_months + 1)
            duration_months = rng.randint(12, 48)
            start_month_index = end_month_index - (duration_months - 1)

            min_index = _to_month_index(MIN_YEAR, 1)
            if start_month_index < min_index:
                start_month_index = min_index
            if end_month_index < start_month_index:
                end_month_index = start_month_index

            start_year, start_month = _from_month_index(start_month_index)
            end_year, end_month = _from_month_index(end_month_index)

        seen_companies.add(company)
        experience.append(
            ExperienceEntry(
                job_title="",
                company=company,
                start_year=start_year,
                start_month=start_month,
                end_year=end_year,
                end_month=end_month,
                description="",
            )
        )

    experience = _assign_experience_titles(rng, experience, domain_titles)

    oldest_exp = experience[-1]
    oldest_exp_start_index = _to_month_index(oldest_exp.start_year, oldest_exp.start_month)
    job_search_gap_months = rng.randint(1, 3)
    education_end_index = oldest_exp_start_index - job_search_gap_months
    education_duration_months = rng.randint(36, 60)
    education_start_index = education_end_index - (education_duration_months - 1)
    education_start_index = max(education_start_index, _to_month_index(MIN_YEAR - 5, 1))
    education_start_year, education_start_month = _from_month_index(education_start_index)
    education_end_year, education_end_month = _from_month_index(education_end_index)

    education = [
        EducationEntry(
            degree=rng.choice(DEGREES),
            institution=f"{fake.last_name()} University",
            start_year=education_start_year,
            start_month=education_start_month,
            year_end=education_end_year,
            end_month=education_end_month,
        )
    ]

    summary = "Professional profile generated by LLM."

    return CVData(
        full_name=full_name,
        email=email,
        gender=gender,
        phone=phone,
        location=location,
        summary=summary,
        education=education,
        experience=experience,
        skills=rng.sample(domain_skills, k=min(7, len(domain_skills))),
    )
