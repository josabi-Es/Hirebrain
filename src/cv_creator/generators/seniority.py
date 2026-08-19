"""Career seniority rules aligned with total professional experience."""

from __future__ import annotations

from datetime import date

from cv_creator.core.models import ExperienceEntry
from cv_creator.generators.pools import role_level

CURRENT_MONTH = date.today().month


def career_span_months(
    experience: list[ExperienceEntry],
    *,
    current_year: int | None = None,
    current_month: int = CURRENT_MONTH,
) -> int:
    """
    Months from the oldest role start through the end of the most recent role
    (or today when the current role is ongoing).
    """
    if not experience:
        return 0

    year = current_year if current_year is not None else date.today().year
    oldest = experience[-1]
    newest = experience[0]
    start_index = oldest.start_year * 12 + (oldest.start_month - 1)

    if newest.end_year is None or newest.end_month is None:
        end_index = year * 12 + (current_month - 1)
    else:
        end_index = newest.end_year * 12 + (newest.end_month - 1)

    return max(0, end_index - start_index + 1)


def career_years(experience: list[ExperienceEntry], **kwargs: object) -> float:
    """Total career span in fractional years."""
    return career_span_months(experience, **kwargs) / 12.0


def min_level_for_career_years(years: float) -> int:
    """
    Minimum seniority level for any role given total years worked.

    Policy (strict):
    - >= 12 years: Senior+ (level 2+)
    - >= 8 years: no base/mid-only titles (level 2+)
    - >= 5 years: no Junior/entry (level 1+)
    """
    if years >= 12:
        return 2
    if years >= 8:
        return 2
    if years >= 5:
        return 1
    return 0


def min_level_for_most_recent(years: float) -> int:
    """Minimum level for the current/most recent role."""
    return min_level_for_career_years(years)


def min_level_for_role_at_tenure(tenure_years: float) -> int:
    """
    Minimum seniority for a role given years already worked before that role starts.

    Uses the same thresholds as total-career policy, evaluated at hire time.
    """
    return min_level_for_career_years(tenure_years)


def tenure_months_at_role_start(
    experience: list[ExperienceEntry],
    role_index: int,
) -> int:
    """
    Months from the oldest role start until this role's start.

    experience[0] is most recent; role_index follows that ordering.
    """
    if not experience or role_index < 0 or role_index >= len(experience):
        return 0

    oldest = experience[-1]
    role = experience[role_index]
    oldest_start = oldest.start_year * 12 + (oldest.start_month - 1)
    role_start = role.start_year * 12 + (role.start_month - 1)
    return max(0, role_start - oldest_start)


def tenure_years_at_role_start(
    experience: list[ExperienceEntry],
    role_index: int,
) -> float:
    """Years already worked before this role begins."""
    return tenure_months_at_role_start(experience, role_index) / 12.0


# First job out of college: junior or mid only.
ENTRY_ROLE_MAX_LEVEL = 1


def max_level_for_most_recent(total_years: float, *, prior_level: int) -> int:
    """Upper bound for the current role based on career length and prior step."""
    if total_years >= 15:
        ceiling = 4
    elif total_years >= 10:
        ceiling = 3
    else:
        ceiling = 3
    return min(ceiling, prior_level + 1)


def is_junior_title(job_title: str) -> bool:
    return role_level(job_title) == 0
