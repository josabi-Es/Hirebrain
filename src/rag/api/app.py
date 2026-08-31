"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from rag.api.routes_chat import register_routes as register_chat_routes


def create_app() -> FastAPI:
    """Create FastAPI app with search and chat routes."""
    app = FastAPI(title="RAG-powered CV Screener")
    register_chat_routes(app)
    return app


app = create_app()
