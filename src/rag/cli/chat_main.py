"""CLI entry point for the LangGraph CV screening agent."""

from __future__ import annotations

import argparse
import sys

from rag.agent.debug import format_agent_debug
from rag.agent.graph import run_agent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask the LangGraph CV screening agent.",
    )
    parser.add_argument(
        "question",
        nargs="+",
        help="Natural-language question about the indexed CV corpus.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print routing metadata and sources.",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Conversation thread id to continue an in-memory LangGraph session.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    query = " ".join(args.question).strip()

    try:
        state = run_agent(query, thread_id=args.thread_id)
    except Exception as exc:
        print(f"Agent error: {exc}", file=sys.stderr)
        return 1

    print(state.get("answer", ""))
    if args.verbose:
        if not state.get("thread_id") and args.thread_id:
            state = {**state, "thread_id": args.thread_id}
        print(f"\n{format_agent_debug(state)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
