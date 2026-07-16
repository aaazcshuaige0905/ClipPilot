from __future__ import annotations

from pathlib import Path
from typing import Any

from clippilot.rag.schemas import KnowledgeChunk, RetrievalQuery


class ChromaStoreError(RuntimeError):
    """Raise when Chroma operations fail or the dependency is unavailable."""


class ChromaVectorStore:
    """Wrap Chroma collection access behind a small project-specific interface."""

    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._available = False
        try:
            import chromadb  # type: ignore
        except ImportError:
            return

        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._available = True

    @property
    def available(self) -> bool:
        """Return whether the Chroma dependency is available."""

        return self._available and self._collection is not None

    def upsert_chunks(self, chunks: list[KnowledgeChunk], embeddings: list[list[float]]) -> None:
        """Insert or update chunk vectors in the Chroma collection."""

        if not self.available:
            raise ChromaStoreError("Chroma is not available. Install `chromadb` to enable dense retrieval.")
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[chunk.flattened_metadata() for chunk in chunks],
        )

    def query(self, query: RetrievalQuery, embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        """Query the Chroma collection and normalize the returned rows."""

        if not self.available or not embedding:
            return []
        where = self._where_filter(query.metadata_filters())
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        rows: list[dict[str, Any]] = []
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            similarity = max(0.0, 1.0 - float(distance or 0.0))
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "text": document,
                    "metadata": metadata or {},
                    "score": round(similarity, 6),
                }
            )
        return rows

    @staticmethod
    def _where_filter(filters: dict[str, str]) -> dict[str, Any] | None:
        """Convert equality filters into a Chroma where clause."""

        active_filters = {key: value for key, value in filters.items() if value}
        if not active_filters:
            return None
        if len(active_filters) == 1:
            key, value = next(iter(active_filters.items()))
            return {key: value}
        return {"$and": [{key: value} for key, value in active_filters.items()]}
