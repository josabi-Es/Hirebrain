"""End-to-end pipeline tests with mocked LLM responses."""

from pathlib import Path
import json

import pytest
from PIL import Image

from cv_creator.core.exceptions import LLMTimeoutError
from cv_creator.core.models import GenerationConfig, LLMConfig, PhotoConfig, PipelineConfig
from cv_creator.generators.skeleton import build_candidate_skeleton
from cv_creator.pipeline import run_pipeline
from cv_creator.rendering.renderer import render_html


@pytest.fixture
def base_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        llm=LLMConfig(model="llama3:8b"),
        photo=PhotoConfig(enabled=False),
        output_dir=str(tmp_path / "out"),
    )


def test_build_skeleton_minimal(base_config: PipelineConfig):
    cfg = base_config.generation.model_copy(update={"seed": 12345})
    cv = build_candidate_skeleton(cfg)
    assert cv.full_name
    assert cv.experience
    assert cv.education
    assert cv.skills
    assert all(1 <= exp.start_month <= 12 for exp in cv.experience)
    assert all(exp.end_month is None or 1 <= exp.end_month <= 12 for exp in cv.experience)

    # Experience must be strictly backward chronological.
    for idx in range(1, len(cv.experience)):
        newer = cv.experience[idx - 1]
        older = cv.experience[idx]
        newer_start = newer.start_year * 12 + newer.start_month
        older_end = older.end_year * 12 + older.end_month if older.end_year and older.end_month else 0
        assert older_end < newer_start

    edu = cv.education[0]
    assert 1 <= edu.start_month <= 12
    assert 1 <= edu.end_month <= 12
    oldest_exp = cv.experience[-1]
    oldest_exp_start = oldest_exp.start_year * 12 + oldest_exp.start_month
    edu_end = edu.year_end * 12 + edu.end_month
    assert edu_end < oldest_exp_start


def test_render_html_shows_month_year_dates():
    cv = build_candidate_skeleton(GenerationConfig(seed=42))
    html = render_html(cv)
    exp = cv.experience[0]
    edu = cv.education[0]

    assert f"{exp.start_month:02d}/{exp.start_year}" in html
    if exp.end_year is not None and exp.end_month is not None:
        assert f"{exp.end_month:02d}/{exp.end_year}" in html
    else:
        assert "Present" in html
    assert f"{edu.start_month:02d}/{edu.start_year}" in html
    assert f"{edu.end_month:02d}/{edu.year_end}" in html


def test_render_html_supports_all_templates():
    cv = build_candidate_skeleton(GenerationConfig(seed=42))
    for template in ("classic", "modern", "minimal"):
        html = render_html(cv, template=template)
        assert cv.full_name in html
        assert f"{cv.experience[0].start_month:02d}/{cv.experience[0].start_year}" in html


def test_render_html_includes_photo_when_present(tmp_path: Path):
    cv = build_candidate_skeleton(GenerationConfig(seed=7))
    photo_path = tmp_path / "photo.png"
    Image.new("RGB", (64, 64), color=(80, 120, 170)).save(photo_path)
    cv = cv.model_copy(update={"photo_path": str(photo_path)})

    html = render_html(cv)
    assert "data:image/png;base64," in html


def test_pipeline_e2e_with_mocked_llm(base_config: PipelineConfig, monkeypatch: pytest.MonkeyPatch):
    base_config.generation.seed = 999
    base_config.photo.enabled = True

    def fake_call(prompt: str, cfg: LLMConfig) -> str:  # noqa: ARG001
        marker = "Candidate skeleton:"
        skeleton_json = prompt.split(marker, maxsplit=1)[1].strip()
        skeleton = json.loads(skeleton_json)
        experience = []
        for item in skeleton["experience"]:
            experience.append(
                {
                    "job_title": item["job_title"],
                    "company": item["company"],
                    "description": f"Delivered measurable impact as {item['job_title']} at {item['company']}.",
                }
            )
        return json.dumps(
            {
                "summary": "Software professional focused on delivery, collaboration, and measurable impact.",
                "experience": experience,
            }
        )

    def fake_export_pdf(html: str, output_path: Path) -> Path:
        output_path.write_bytes(b"%PDF-1.4\n% mock cv pdf\n")
        return output_path

    def fake_generate_photo(gender: str, age: int, output_file: str) -> str:
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color=(120, 100, 160)).save(path)
        return str(path)

    monkeypatch.setattr("cv_creator.integration.llm_service._call_ollama", fake_call)
    monkeypatch.setattr("cv_creator.pipeline.generar_foto_perfil", fake_generate_photo)
    monkeypatch.setattr("cv_creator.pipeline.export_pdf", fake_export_pdf)
    result = run_pipeline(base_config)
    assert result.pdf_path
    assert Path(result.pdf_path).exists()
    assert Path(result.pdf_path).read_bytes().startswith(b"%PDF")
    assert result.cv_data.summary.startswith("Software professional")
    assert result.cv_data.photo_path is not None
    assert Path(result.cv_data.photo_path).exists()


def test_pipeline_fails_when_llm_missing_required_fields(
    base_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
):
    base_config.generation.seed = 123

    def broken_call(prompt: str, cfg: LLMConfig) -> str:  # noqa: ARG001
        return '{"summary": "Only summary", "experience": []}'

    monkeypatch.setattr("cv_creator.integration.llm_service._call_ollama", broken_call)
    with pytest.raises(LLMTimeoutError):
        run_pipeline(base_config)
