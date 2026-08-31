"""Unit tests for Hugging Face profile photo service."""

from pathlib import Path

import pytest
from PIL import Image

from cv_creator.core.exceptions import PhotoGenerationError
from cv_creator.integration.photo_service import generar_foto_perfil


def test_generar_foto_perfil_with_mocked_hf_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    class FakeClient:
        def __init__(self, token: str, timeout: float):  # noqa: ARG002
            self.token = token

        def text_to_image(self, prompt: str, model: str):  # noqa: ARG002
            return Image.new("RGB", (32, 32), color=(10, 90, 140))

    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr("cv_creator.integration.photo_service.InferenceClient", FakeClient)

    out = tmp_path / "photo.png"
    result_path = generar_foto_perfil("female", 29, str(out))
    assert Path(result_path).exists()
    assert Path(result_path).suffix.lower() == ".png"


def test_generar_foto_perfil_requires_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr("cv_creator.integration.photo_service.load_dotenv", lambda: None)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(PhotoGenerationError):
        generar_foto_perfil("male", 31, str(tmp_path / "photo.png"))

