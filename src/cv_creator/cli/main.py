"""CLI entry point for cv-creator."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

from cv_creator.core.paths import DEFAULT_CVS_DIR, DEFAULT_PHOTO_CACHE_DIR
from cv_creator.core.models import GenerationConfig, LLMConfig, PhotoConfig, PipelineConfig, RenderConfig
from cv_creator.pipeline import generate_cvs
from cv_creator.rendering.renderer import AVAILABLE_TEMPLATES

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate synthetic CVs with LLM-enriched summary and experience.",
    )
    parser.add_argument("-n", "--count", type=int, default=1, help="Number of CVs to generate")
    parser.add_argument(
        "-o",
        "--output",
        default=os.getenv("CV_OUTPUT_DIR", str(DEFAULT_CVS_DIR)),
        help="Directory for generated CV PDFs (default: data/cvs)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", "llama3:8b"),
        help="Ollama model name",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("OLLAMA_HOST"),
        help="Optional Ollama host URL",
    )
    parser.add_argument(
        "--template",
        choices=AVAILABLE_TEMPLATES,
        default=os.getenv("CV_TEMPLATE") or None,
        help=(
            "CV layout template. If omitted, rotates automatically across "
            f"{', '.join(AVAILABLE_TEMPLATES)} per generated CV."
        ),
    )

    args = parser.parse_args(argv)

    config = PipelineConfig(
        generation=GenerationConfig(
            seed=args.seed,
        ),
        llm=LLMConfig(model=args.model, host=args.host),
        photo=PhotoConfig(
            enabled=os.getenv("HF_PHOTO_ENABLED", "true").lower() == "true",
            fail_on_error=os.getenv("HF_PHOTO_FAIL_ON_ERROR", "false").lower() == "true",
            cache_dir=os.getenv("HF_PHOTO_CACHE_DIR", str(DEFAULT_PHOTO_CACHE_DIR)),
        ),
        render=RenderConfig(template=args.template if args.template else None),
        output_dir=args.output,
    )

    results = generate_cvs(args.count, config=config, base_seed=args.seed)

    for r in results:
        print(f"Generated: {r.cv_data.full_name}")
        print(f"  PDF: {r.pdf_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
