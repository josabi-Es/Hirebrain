from __future__ import annotations

from pathlib import Path

import pytest

from cv_creator.core.paths import cvs_dir, pdf_search_dirs
from rag.ingest.paths import resolve_pdf_path, resolve_pdf_reference


def test_pdf_search_dirs_includes_cvs_and_legacy(tmp_path: Path) -> None:
    dirs = pdf_search_dirs(tmp_path)
    assert tmp_path.resolve() in dirs
    assert (tmp_path / "data" / "cvs").resolve() in dirs
    assert (tmp_path / "output").resolve() in dirs


def test_resolve_pdf_path_finds_under_data_cvs(tmp_path: Path) -> None:
    pdf = tmp_path / "data" / "cvs" / "cv_test.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF")

    found = resolve_pdf_path(tmp_path, "cv_test.pdf")
    assert found == pdf


def test_resolve_pdf_path_legacy_output(tmp_path: Path) -> None:
    pdf = tmp_path / "output" / "cv_legacy.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF")

    found = resolve_pdf_path(tmp_path, "cv_legacy.pdf")
    assert found == pdf


def test_resolve_pdf_reference_bare_filename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = cvs_dir(tmp_path) / "cv_named.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF")
    monkeypatch.chdir(tmp_path)

    resolved = resolve_pdf_reference("cv_named.pdf")
    assert resolved == pdf.resolve()


def test_resolve_pdf_reference_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="no_encontrado.pdf"):
        resolve_pdf_reference("no_encontrado.pdf")
