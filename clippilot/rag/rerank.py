from __future__ import annotations

from clippilot.rag.schemas import RetrievedChunk, RetrievalQuery


def score_metadata_fit(chunk: RetrievedChunk, query: RetrievalQuery) -> float:
    """Compute a small business-aware metadata bonus for one retrieved chunk."""

    score = 0.0
    if chunk.stage == query.stage:
        score += 1.0
    if query.platform and chunk.platform == query.platform:
        score += 1.2
    if query.language and chunk.language == query.language:
        score += 0.8
    if query.style and chunk.style == query.style:
        score += 0.8
    if query.duration_band and chunk.duration_band == query.duration_band:
        score += 0.7
    if chunk.priority == "must":
        score += 0.8
    elif chunk.priority == "should":
        score += 0.4
    if query.tags:
        overlap = len(set(chunk.tags).intersection(set(query.tags)))
        score += min(0.8, overlap * 0.2)
    return round(score, 6)


def apply_metadata_rerank(chunks: list[RetrievedChunk], query: RetrievalQuery) -> list[RetrievedChunk]:
    """Populate metadata scores and re-sort the chunks for downstream fusion."""

    for chunk in chunks:
        chunk.metadata_score = score_metadata_fit(chunk, query)
    return sorted(chunks, key=lambda item: (item.metadata_score, item.dense_score, item.bm25_score), reverse=True)
