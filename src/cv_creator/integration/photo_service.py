"""Profile photo generation using Hugging Face Inference API."""

from __future__ import annotations

import io
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image

from cv_creator.core.exceptions import PhotoGenerationError


def _normalize_gender(genero: str) -> str:
    value = genero.strip().lower()
    if value in {"male", "m", "man", "hombre"}:
        return "male"
    if value in {"female", "f", "woman", "mujer"}:
        return "female"
    raise PhotoGenerationError(f"Unsupported gender value: {genero}")


def _resolve_age(edad: int) -> int:
    if edad < 18:
        return 18
    if edad > 75:
        return 75
    return edad


def _build_prompt(gender: str, age: int) -> str:
    return (
        f"Photorealistic corporate headshot of a {age} year old {gender}, "
        "wearing professional business attire, neutral light grey background, "
        "studio lighting, sharp focus, 8k, professional linkedin profile picture style."
    )


def _save_image(image_obj: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_obj.save(output_path)


def generar_foto_perfil(genero: str, edad: int, nombre_archivo: str) -> str:
    """
    Generate a corporate profile photo using black-forest-labs/FLUX.1-schnell.

    Args:
        genero: Candidate gender ('male' or 'female')
        edad: Candidate age in years
        nombre_archivo: Target image path for the generated photo

    Returns:
        Absolute path to the generated image file.
    """
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise PhotoGenerationError("HF_TOKEN is not configured in environment variables")

    gender = _normalize_gender(genero)
    age = _resolve_age(int(edad))
    output_path = Path(nombre_archivo).expanduser().resolve()
    prompt = _build_prompt(gender, age)
    model_id = os.getenv("HF_PHOTO_MODEL_ID", "black-forest-labs/FLUX.1-schnell")
    timeout_seconds = float(os.getenv("HF_PHOTO_TIMEOUT_SECONDS", "60"))

    try:
        client = InferenceClient(token=hf_token, timeout=timeout_seconds)
        result = client.text_to_image(
            prompt=prompt,
            model=model_id,
        )
        if isinstance(result, Image.Image):
            _save_image(result, output_path)
        elif isinstance(result, bytes):
            image = Image.open(io.BytesIO(result))
            _save_image(image, output_path)
        else:
            raise PhotoGenerationError("Unexpected response type from Hugging Face API")
        return str(output_path)
    except PhotoGenerationError:
        raise
    except Exception as exc:
        raise PhotoGenerationError(f"Failed to generate profile photo: {exc}") from exc
