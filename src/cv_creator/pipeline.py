"""End-to-end cv-creator pipeline orchestration."""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

from cv_creator.core.exceptions import PhotoGenerationError
from cv_creator.core.models import PipelineConfig, PipelineResult
from cv_creator.generators.skeleton import build_candidate_skeleton
from cv_creator.integration.llm_service import enrich_cv_with_llm
from cv_creator.integration.photo_service import generar_foto_perfil
from cv_creator.core.models import CvTemplate
from cv_creator.rendering.pdf_exporter import export_pdf
from cv_creator.rendering.renderer import AVAILABLE_TEMPLATES, render_html

logger = logging.getLogger(__name__)


def _slugify(name: str) -> str:
    s = re.sub(r"[^\w\s-]", "", name.lower())
    return re.sub(r"[-\s]+", "_", s).strip("_") or "candidate"


def _resolve_template(config: PipelineConfig, batch_index: int = 0) -> CvTemplate:
    """Pick template: explicit config, seed-based rotation, or batch index."""
    if config.render.template:
        return config.render.template
    seed = config.generation.seed
    if seed is not None:
        return AVAILABLE_TEMPLATES[seed % len(AVAILABLE_TEMPLATES)]
    return AVAILABLE_TEMPLATES[batch_index % len(AVAILABLE_TEMPLATES)]


def _estimate_age_from_cv(cv_data: object) -> int:
    """
    Estimate age for synthetic portrait prompt.
    Uses education start year when available, otherwise a sensible default.
    """
    try:
        education = cv_data.education  # type: ignore[attr-defined]
    except Exception:
        return 30
    if not education:
        return 30
    # Typical graduation path starts around 18 years old.
    first_edu_start = min(item.start_year for item in education)
    current_year = date.today().year
    return max(22, min(65, current_year - first_edu_start + 18))


def run_pipeline(config: PipelineConfig, *, batch_index: int = 0) -> PipelineResult:
    """Execute minimal pipeline and return PDF artifact."""
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    skeleton_cv = build_candidate_skeleton(config.generation)
    cv_data = skeleton_cv
    if config.photo.enabled:
        slug = _slugify(cv_data.full_name)
        cache_dir = Path(config.photo.cache_dir)
        photo_path = cache_dir / f"{slug}.png"
        try:
            if not photo_path.exists():
                generated = generar_foto_perfil(
                    cv_data.gender,
                    _estimate_age_from_cv(cv_data),
                    str(photo_path),
                )
                photo_path = Path(generated)
            cv_data = cv_data.model_copy(update={"photo_path": str(photo_path.resolve())})
        except PhotoGenerationError as exc:
            if config.photo.fail_on_error:
                raise
            logger.warning("Photo generation skipped due to error: %s", exc)

    cv_data = enrich_cv_with_llm(cv_data, config.llm)
    template = _resolve_template(config, batch_index)
    html = render_html(cv_data, template=template)
    slug = _slugify(cv_data.full_name)
    pdf_path = export_pdf(html, out_dir / f"cv_{slug}.pdf")

    return PipelineResult(
        cv_data=cv_data,
        pdf_path=str(pdf_path),
    )


def generate_cvs(
    count: int = 1,
    *,
    config: PipelineConfig | None = None,
    base_seed: int | None = None,
) -> list[PipelineResult]:
    """Generate multiple CVs with incremental seeds."""
    cfg = config or PipelineConfig()
    results: list[PipelineResult] = []
    for i in range(count):
        run_cfg = cfg.model_copy(deep=True)
        if base_seed is not None:
            run_cfg.generation.seed = base_seed + i
        results.append(run_pipeline(run_cfg, batch_index=i))
    return results
