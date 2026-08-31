"""Pydantic domain models for the simplified cv-creator pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from cv_creator.core.paths import DEFAULT_CVS_DIR, DEFAULT_PHOTO_CACHE_DIR

CvTemplate = Literal["classic", "modern", "minimal"]


class GenerationConfig(BaseModel):
    seed: int | None = None
    locale: str = "en_US"


class LLMConfig(BaseModel):
    model: str = "llama3:8b"
    host: str | None = None
    max_retries: int = 3
    temperature: float = 0.7


class PhotoConfig(BaseModel):
    enabled: bool = True
    fail_on_error: bool = False
    cache_dir: str = str(DEFAULT_PHOTO_CACHE_DIR)


class RenderConfig(BaseModel):
    template: CvTemplate | None = None


class PipelineConfig(BaseModel):
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    photo: PhotoConfig = Field(default_factory=PhotoConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    output_dir: str = str(DEFAULT_CVS_DIR)


class EducationEntry(BaseModel):
    degree: str
    institution: str
    start_year: int
    start_month: int = Field(ge=1, le=12)
    year_end: int
    end_month: int = Field(ge=1, le=12)


class ExperienceEntry(BaseModel):
    job_title: str
    company: str
    start_year: int
    start_month: int = Field(ge=1, le=12)
    end_year: int | None = None
    end_month: int | None = Field(default=None, ge=1, le=12)
    description: str = ""


class CVData(BaseModel):
    full_name: str
    email: str
    gender: Literal["male", "female"]
    phone: str
    location: str
    summary: str
    education: list[EducationEntry] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    photo_path: str | None = None


class PipelineResult(BaseModel):
    cv_data: CVData
    pdf_path: str
