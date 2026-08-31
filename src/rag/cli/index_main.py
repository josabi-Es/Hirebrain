from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cv_creator.core.paths import DEFAULT_RAG_CHUNKS_DIR
from rag.retrieval.indexer import index_chunks_from_directory, index_chunks_from_json
from rag.shared.settings import get_settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index CV chunk JSON artifacts into Qdrant (hybrid dense + sparse).",
    )
    parser.add_argument(
        "json_files",
        nargs="*",
        metavar="JSON",
        help="Path(s) to *.chunks.json files.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=f"Index all *.chunks.json under this directory (default: {DEFAULT_RAG_CHUNKS_DIR}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    settings = get_settings()
    total = 0

    if args.json_files:
        for reference in args.json_files:
            path = Path(reference).resolve()
            if not path.is_file():
                print(f"Error: archivo no encontrado: {path}", file=sys.stderr)
                return 1
            count = index_chunks_from_json(path)
            total += count
            print(f"Indexados {count} chunks desde {path.name}")
    else:
        input_dir = (args.input_dir or DEFAULT_RAG_CHUNKS_DIR).resolve()
        if not input_dir.is_dir():
            print(f"Error: directorio no encontrado: {input_dir}", file=sys.stderr)
            return 1
        total = index_chunks_from_directory(input_dir)
        print(f"Indexados {total} chunks desde {input_dir}")

    print(
        f"Listo: {total} puntos en colección {settings.qdrant_collection!r} "
        f"({settings.qdrant_url})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
