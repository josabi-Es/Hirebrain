from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cv_creator.core.paths import DEFAULT_CVS_DIR, DEFAULT_RAG_CHUNKS_DIR, DEFAULT_RAG_CLEAN_TEXT_DIR
from rag.ingest.pipeline import (
    export_chunk_files,
    export_clean_text_files,
    format_chunks_json,
    format_console,
    format_json,
    resolve_pdf_paths,
    run_batch,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extrae y segmenta CVs en PDF para el pipeline RAG.",
    )
    parser.add_argument(
        "pdfs",
        nargs="*",
        metavar="PDF",
        help=(
            "Ruta(s) a PDF o nombre de archivo "
            f"(busca en ./, {DEFAULT_CVS_DIR}/ y output/ legacy)."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=f"Procesa todos los *.pdf del directorio (p. ej. {DEFAULT_CVS_DIR}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Salida estructurada en JSON.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Directorio base para resolver nombres de archivo (default: cwd).",
    )
    parser.add_argument(
        "--clean-text-dir",
        type=Path,
        default=DEFAULT_RAG_CLEAN_TEXT_DIR,
        help=f"Carpeta donde guardar clean_text por CV (default: {DEFAULT_RAG_CLEAN_TEXT_DIR}).",
    )
    parser.add_argument(
        "--no-export-clean-text",
        action="store_true",
        help="No escribir archivos .txt con el clean_text normalizado.",
    )
    parser.add_argument(
        "--chunks-json",
        action="store_true",
        help="Salida JSON solo con chunks y metadata (sin secciones completas).",
    )
    parser.add_argument(
        "--export-chunks-dir",
        nargs="?",
        const=DEFAULT_RAG_CHUNKS_DIR,
        default=None,
        type=Path,
        metavar="DIR",
        help=(
            "Escribe un .chunks.json por CV con chunks y metadata "
            f"(default al usar flag sin DIR: {DEFAULT_RAG_CHUNKS_DIR})."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for cv-extract."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.pdfs and args.input_dir is None:
        parser.error("Indica al menos un PDF o usa --input-dir.")

    try:
        pdf_paths = resolve_pdf_paths(
            positional=args.pdfs,
            input_dir=args.input_dir,
            base_dir=args.base_dir,
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not pdf_paths:
        print("Error: no se encontraron archivos PDF para procesar.", file=sys.stderr)
        return 1

    try:
        results = run_batch(pdf_paths)
    except ModuleNotFoundError as exc:
        if "pymupdf" in str(exc).lower() or "fitz" in str(exc).lower():
            print(
                "Error: falta PyMuPDF. Instala con: uv sync --extra rag",
                file=sys.stderr,
            )
            return 1
        raise
    except Exception as exc:
        print(f"Error durante la extracción: {exc}", file=sys.stderr)
        return 1

    written_paths: list[Path] = []
    if not args.no_export_clean_text:
        written_paths = export_clean_text_files(results, output_dir=args.clean_text_dir)

    chunk_written_paths: list[Path] = []
    if args.export_chunks_dir is not None:
        chunk_written_paths = export_chunk_files(
            results,
            output_dir=args.export_chunks_dir,
        )

    if args.chunks_json:
        print(format_chunks_json(results))
    elif args.json:
        print(format_json(results))
    else:
        for result in results:
            print(format_console(result))
        if written_paths:
            print(f"\n{'=' * 72}")
            print(f"Clean text exportado en: {args.clean_text_dir.resolve()}")
            for path in written_paths:
                print(f"  - {path.name}")
        if chunk_written_paths:
            print(f"\nChunks exportados en: {chunk_written_paths[0].parent}")
            for path in chunk_written_paths:
                print(f"  - {path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
