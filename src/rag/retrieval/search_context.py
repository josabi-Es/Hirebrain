"""Shared setup for Qdrant search operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from rag.retrieval.filters import build_query_filter
from rag.shared.settings import RagSettings, get_settings


@dataclass(frozen=True)
class SearchContext:
    """Resolved client, settings, limit, and filter for a retrieval query."""

    settings: RagSettings
    client: QdrantClient
    limit: int
    query_filter: qmodels.Filter | None


def build_search_context(
    *,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    client: QdrantClient | None = None,
    settings: RagSettings | None = None,
) -> SearchContext:
    """Resolve settings, Qdrant client, top-k limit, and payload filter."""
    active = settings or get_settings()
    return SearchContext(
        settings=active,
        client=client or QdrantClient(url=active.qdrant_url),
        limit=top_k or active.retrieval_top_k,
        query_filter=build_query_filter(filters),
    )
