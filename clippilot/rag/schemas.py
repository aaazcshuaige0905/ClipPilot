from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp string."""

    return datetime.now(timezone.utc).isoformat()


class KnowledgeChunk(BaseModel):
    """Represent one strategy-focused chunk stored in the RAG corpus."""

    chunk_id: str
    text: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    source_file: str = Field(..., min_length=1)
    knowledge_type: str = Field(..., min_length=1)
    strategy_type: str = Field(..., min_length=1)
    platform: str | None = None
    language: str | None = None
    style: str | None = None
    duration_band: str | None = None
    stage: str = "planning"
    priority: str = "should"
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def flattened_metadata(self) -> dict[str, Any]:
        """Return a flat metadata dictionary suitable for Chroma filters."""

        payload: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "title": self.title,
            "source_file": self.source_file,
            "knowledge_type": self.knowledge_type,
            "strategy_type": self.strategy_type,
            "platform": self.platform,
            "language": self.language,
            "style": self.style,
            "duration_band": self.duration_band,
            "stage": self.stage,
            "priority": self.priority,
            "tags": ",".join(self.tags),
        }
        for key, value in self.metadata.items():
            if key not in payload and value is not None:
                payload[key] = value
        return {key: value for key, value in payload.items() if value not in (None, "", [])}


class RetrievalQuery(BaseModel):
    """Represent the planner-oriented retrieval query for one task."""

    text: str = Field(..., min_length=1)
    platform: str | None = None
    language: str | None = None
    style: str | None = None
    duration_band: str | None = None
    stage: str = "planning"
    top_k_dense: int = Field(default=8, ge=1)
    top_k_bm25: int = Field(default=8, ge=1)
    top_k_final: int = Field(default=3, ge=1)
    tags: list[str] = Field(default_factory=list)

    def metadata_filters(self) -> dict[str, str]:
        """Return compact equality filters for stores that support metadata filtering."""

        filters: dict[str, str] = {"stage": self.stage}
        if self.platform:
            filters["platform"] = self.platform
        if self.language:
            filters["language"] = self.language
        if self.style:
            filters["style"] = self.style
        if self.duration_band:
            filters["duration_band"] = self.duration_band
        return filters


class RetrievedChunk(BaseModel):
    """Represent one retrieved chunk together with retrieval diagnostics."""

    chunk_id: str
    text: str
    title: str
    source_file: str
    knowledge_type: str
    strategy_type: str
    platform: str | None = None
    language: str | None = None
    style: str | None = None
    duration_band: str | None = None
    stage: str = "planning"
    priority: str = "should"
    tags: list[str] = Field(default_factory=list)
    dense_score: float = 0.0
    bm25_score: float = 0.0
    metadata_score: float = 0.0
    final_score: float = 0.0
    retrieval_sources: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalTrace(BaseModel):
    """Represent debugging details for one retrieval pass."""

    dense_candidate_count: int = 0
    bm25_candidate_count: int = 0
    merged_candidate_count: int = 0
    filters: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class RetrievedContext(BaseModel):
    """Represent the final RAG context injected into downstream agents."""

    query: RetrievalQuery
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    retrieval_mode: str = "hybrid_dense_bm25_metadata"
    trace: RetrievalTrace = Field(default_factory=RetrievalTrace)
    generated_at: str = Field(default_factory=utc_timestamp)


class IndexBuildReport(BaseModel):
    """Represent one index build result for the strategy knowledge base."""

    collection_name: str
    chunk_count: int = 0
    dense_index_built: bool = False
    bm25_index_built: bool = False
    manifest_path: str | None = None
    warnings: list[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=utc_timestamp)
