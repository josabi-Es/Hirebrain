"""RAG runtime settings (vector store, embeddings, retrieval)."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

DEFAULT_PROFILE_SECTIONS = ("SUMMARY", "SKILLS", "EXPERIENCE", "EDUCATION")


class RagSettings(BaseModel):
    """Configuration for Qdrant retrieval and embedding models."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    qdrant_url: str = Field(default="http://127.0.0.1:6333")
    qdrant_collection: str = Field(default="cv_chunks")
    dense_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    sparse_model: str = Field(default="Qdrant/bm25")
    reranker_model: str = Field(default="jinaai/jina-reranker-v2-base-multilingual")
    retrieval_top_k: int = Field(default=12, ge=1)
    rerank_top_k: int = Field(default=8, ge=1)
    profile_scroll_limit: int = Field(default=200, ge=1)
    dense_vector_name: str = Field(default=DENSE_VECTOR_NAME)
    sparse_vector_name: str = Field(default=SPARSE_VECTOR_NAME)
    dense_vector_size: int = Field(default=384, ge=1)

    @classmethod
    def from_env(cls) -> RagSettings:
        return cls(
            qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "cv_chunks"),
            dense_model=os.getenv(
                "RAG_DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            sparse_model=os.getenv("RAG_SPARSE_MODEL", "Qdrant/bm25"),
            reranker_model=os.getenv(
                "RAG_RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual"
            ),
            retrieval_top_k=int(os.getenv("RAG_RETRIEVAL_TOP_K", "12")),
            rerank_top_k=int(os.getenv("RAG_RERANK_TOP_K", "8")),
            profile_scroll_limit=int(os.getenv("RAG_PROFILE_SCROLL_LIMIT", "200")),
        )


@lru_cache(maxsize=1)
def get_settings() -> RagSettings:
    """Return cached RAG settings loaded from environment."""
    return RagSettings.from_env()
