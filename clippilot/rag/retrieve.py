from __future__ import annotations

from pathlib import Path

from clippilot.rag.fusion import apply_weighted_fusion
from clippilot.rag.loaders import load_knowledge_chunks
from clippilot.rag.providers import BM25Store, ChromaVectorStore, EmbeddingProviderError, QwenCloudEmbeddingProvider
from clippilot.rag.rerank import apply_metadata_rerank
from clippilot.rag.schemas import RetrievedChunk, RetrievedContext, RetrievalQuery, RetrievalTrace
from clippilot.storage.path_manager import AppSettings, load_settings


def retrieve_context(query: RetrievalQuery, settings: AppSettings | None = None) -> RetrievedContext:
    """Retrieve hybrid strategy context for planner-style downstream agents."""

    active_settings = settings or load_settings()
    trace = RetrievalTrace(filters=query.metadata_filters())
    chunks = _load_chunks_for_retrieval(active_settings)
    bm25_store = BM25Store(chunks)

    dense_rows: list[dict] = []
    chroma_store = ChromaVectorStore(active_settings.rag_chroma_dir, active_settings.rag_collection_name)
    if chroma_store.available and active_settings.qwen_api_key:
        try:
            embedding_provider = QwenCloudEmbeddingProvider(active_settings)
            query_embedding = embedding_provider.embed_query(query.text)
            dense_rows = chroma_store.query(query=query, embedding=query_embedding, top_k=query.top_k_dense)
        except EmbeddingProviderError as exc:
            trace.warnings.append(str(exc))
    else:
        if not active_settings.qwen_api_key:
            trace.warnings.append("Dense retrieval skipped because no Qwen embedding API key is configured.")
        if not chroma_store.available:
            trace.warnings.append("Dense retrieval skipped because `chromadb` is not installed.")

    bm25_rows = bm25_store.search(query=query, top_k=query.top_k_bm25)
    trace.dense_candidate_count = len(dense_rows)
    trace.bm25_candidate_count = len(bm25_rows)

    merged = _merge_results(dense_rows=dense_rows, bm25_rows=bm25_rows)
    reranked = apply_metadata_rerank(merged, query)
    fused = apply_weighted_fusion(
        reranked,
        dense_weight=active_settings.rag_dense_weight,
        bm25_weight=active_settings.rag_bm25_weight,
        metadata_weight=active_settings.rag_metadata_weight,
    )
    final_chunks = fused[: query.top_k_final]
    trace.merged_candidate_count = len(merged)
    return RetrievedContext(query=query, chunks=final_chunks, trace=trace)


def retrieve_knowledge(query: str, top_k: int = 3) -> list[str]:
    """Preserve the old compatibility helper by returning plain chunk texts."""

    settings = load_settings()
    retrieval_query = RetrievalQuery(
        text=query,
        top_k_dense=settings.rag_top_k_dense,
        top_k_bm25=settings.rag_top_k_bm25,
        top_k_final=top_k,
    )
    context = retrieve_context(retrieval_query, settings=settings)
    return [chunk.text for chunk in context.chunks]


def _load_chunks_for_retrieval(settings: AppSettings):
    """Load the persisted BM25 corpus when available, otherwise rebuild from markdown."""

    corpus_path = settings.rag_bm25_dir / "strategy_corpus.json"
    if corpus_path.exists():
        return BM25Store.load(corpus_path).chunks
    return load_knowledge_chunks(settings)


def _merge_results(
    dense_rows: list[dict],
    bm25_rows: list[tuple],
) -> list[RetrievedChunk]:
    """Merge dense and sparse retrieval outputs on chunk id."""

    merged: dict[str, RetrievedChunk] = {}

    for row in dense_rows:
        metadata = row.get("metadata", {})
        chunk_id = str(row.get("chunk_id"))
        merged[chunk_id] = RetrievedChunk(
            chunk_id=chunk_id,
            text=row.get("text", ""),
            title=str(metadata.get("title", chunk_id)),
            source_file=str(metadata.get("source_file", "")),
            knowledge_type=str(metadata.get("knowledge_type", "strategy_rule")),
            strategy_type=str(metadata.get("strategy_type", "strategy_rule")),
            platform=_optional_str(metadata.get("platform")),
            language=_optional_str(metadata.get("language")),
            style=_optional_str(metadata.get("style")),
            duration_band=_optional_str(metadata.get("duration_band")),
            stage=str(metadata.get("stage", "planning")),
            priority=str(metadata.get("priority", "should")),
            tags=_tags_from_metadata(metadata),
            dense_score=float(row.get("score", 0.0)),
            retrieval_sources=["dense"],
            metadata=dict(metadata),
        )

    for chunk, score in bm25_rows:
        existing = merged.get(chunk.chunk_id)
        if existing is None:
            merged[chunk.chunk_id] = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                title=chunk.title,
                source_file=chunk.source_file,
                knowledge_type=chunk.knowledge_type,
                strategy_type=chunk.strategy_type,
                platform=chunk.platform,
                language=chunk.language,
                style=chunk.style,
                duration_band=chunk.duration_band,
                stage=chunk.stage,
                priority=chunk.priority,
                tags=list(chunk.tags),
                bm25_score=float(score),
                retrieval_sources=["bm25"],
                metadata=chunk.flattened_metadata(),
            )
            continue

        existing.bm25_score = float(score)
        if "bm25" not in existing.retrieval_sources:
            existing.retrieval_sources.append("bm25")
    return list(merged.values())


def _optional_str(value: object) -> str | None:
    """Normalize metadata scalar values to optional strings."""

    if value in (None, ""):
        return None
    return str(value)


def _tags_from_metadata(metadata: dict) -> list[str]:
    """Extract stored tags from flattened metadata."""

    tags = metadata.get("tags")
    if not tags:
        return []
    if isinstance(tags, str):
        return [part for part in tags.split(",") if part]
    if isinstance(tags, list):
        return [str(item) for item in tags]
    return []
