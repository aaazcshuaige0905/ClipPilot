from __future__ import annotations

from clippilot.rag.schemas import RetrievedChunk


def apply_weighted_fusion(
    chunks: list[RetrievedChunk],
    dense_weight: float,
    bm25_weight: float,
    metadata_weight: float,
) -> list[RetrievedChunk]:
    """Combine dense, lexical, and metadata scores into one final ranking score."""

    if not chunks:
        return []

    max_dense = max((chunk.dense_score for chunk in chunks), default=0.0) or 1.0
    max_bm25 = max((chunk.bm25_score for chunk in chunks), default=0.0) or 1.0
    max_metadata = max((chunk.metadata_score for chunk in chunks), default=0.0) or 1.0

    for chunk in chunks:
        dense_component = chunk.dense_score / max_dense if chunk.dense_score > 0 else 0.0
        bm25_component = chunk.bm25_score / max_bm25 if chunk.bm25_score > 0 else 0.0
        metadata_component = chunk.metadata_score / max_metadata if chunk.metadata_score > 0 else 0.0
        chunk.final_score = round(
            (dense_weight * dense_component)
            + (bm25_weight * bm25_component)
            + (metadata_weight * metadata_component),
            6,
        )
    return sorted(chunks, key=lambda item: item.final_score, reverse=True)
