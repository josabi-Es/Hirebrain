"""Phase D: Jinja2 HTML rendering."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from cv_creator.core.exceptions import TemplateRenderError
from cv_creator.core.models import CVData, CvTemplate, EducationEntry, ExperienceEntry

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
AVAILABLE_TEMPLATES: tuple[CvTemplate, ...] = ("classic", "modern", "minimal")


def _format_month_year(year: int, month: int) -> str:
    """Format as MM/YYYY for CV date ranges."""
    return f"{month:02d}/{year}"


def _format_experience_range(job: ExperienceEntry) -> str:
    """Experience line: MM/YYYY – MM/YYYY or MM/YYYY – Present."""
    start = _format_month_year(job.start_year, job.start_month)
    if job.end_year is None or job.end_month is None:
        return f"{start} – Present"
    end = _format_month_year(job.end_year, job.end_month)
    return f"{start} – {end}"


def _format_education_range(edu: EducationEntry) -> str:
    """Education line: MM/YYYY – MM/YYYY."""
    start = _format_month_year(edu.start_year, edu.start_month)
    end = _format_month_year(edu.year_end, edu.end_month)
    return f"{start} – {end}"


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    env.filters["format_experience_range"] = _format_experience_range
    env.filters["format_education_range"] = _format_education_range
    return env


def _load_theme_css(template: CvTemplate) -> str:
    """Load printable CSS for the selected template theme."""
    theme_path = _TEMPLATES_DIR / "themes" / f"{template}.css"
    if not theme_path.exists():
        raise TemplateRenderError(f"Theme CSS not found for template: {template}")
    return theme_path.read_text(encoding="utf-8")


def _photo_to_data_uri(photo_path: str | None) -> str | None:
    """Embed image as data URI to keep WeasyPrint path resolution simple."""
    if not photo_path:
        return None
    path = Path(photo_path)
    if not path.exists() or not path.is_file():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def render_html(cv_data: CVData, template: CvTemplate = "classic") -> str:
    """Render CV to HTML using the selected template."""
    if template not in AVAILABLE_TEMPLATES:
        raise TemplateRenderError(
            f"Unknown template '{template}'. Available: {', '.join(AVAILABLE_TEMPLATES)}"
        )

    env = _env()
    template_file = f"{template}.html"
    try:
        tpl = env.get_template(template_file)
    except Exception as e:
        raise TemplateRenderError(f"Failed to load {template} template: {e}") from e

    try:
        html = tpl.render(cv=cv_data, photo_src=_photo_to_data_uri(cv_data.photo_path))
        return html.replace("/*__THEME_CSS__*/", _load_theme_css(template))
    except Exception as e:
        raise TemplateRenderError(f"Failed to render {template} template: {e}") from e
