"""Local embedding models via fastembed (dense + sparse BM25)."""

from __future__ import annotations

from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding

from rag.shared.schemas import SparseVectorData
from rag.shared.settings import RagSettings, get_settings


@lru_cache(maxsize=4)
def _get_dense_model(model_name: str) -> TextEmbedding:
    return TextEmbedding(model_name=model_name)


@lru_cache(maxsize=4)
def _get_sparse_model(model_name: str) -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=model_name)


class DenseEmbedder:
    """Dense sentence embeddings for semantic retrieval."""

    def __init__(self, settings: RagSettings | None = None) -> None:
        active = settings or get_settings()
        self._model_name = active.dense_model
        self._model = _get_dense_model(self._model_name)

    def embed(self, text: str) -> list[float]:
        results = list(self._model.embed([text]))
        if not results:
            raise ValueError("Dense embedder returned no vectors")
        return results[0].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vector.tolist() for vector in self._model.embed(texts)]


class SparseEmbedder:
    """Sparse BM25-style embeddings for lexical retrieval."""

    def __init__(self, settings: RagSettings | None = None) -> None:
        active = settings or get_settings()
        self._model_name = active.sparse_model
        self._model = _get_sparse_model(self._model_name)

    def embed(self, text: str) -> SparseVectorData:
        results = list(self._model.embed([text]))
        if not results:
            raise ValueError("Sparse embedder returned no vectors")
        sparse = results[0]
        return SparseVectorData(
            indices=sparse.indices.tolist(),
            values=sparse.values.tolist(),
        )

    def embed_batch(self, texts: list[str]) -> list[SparseVectorData]:
        if not texts:
            return []
        return [
            SparseVectorData(
                indices=sparse.indices.tolist(),
                values=sparse.values.tolist(),
            )
            for sparse in self._model.embed(texts)
        ]


def sparse_to_qdrant(sparse: SparseVectorData) -> dict[str, list[int] | list[float]]:
    """Convert SparseVectorData to Qdrant client sparse vector kwargs."""
    return {"indices": sparse.indices, "values": sparse.values}
