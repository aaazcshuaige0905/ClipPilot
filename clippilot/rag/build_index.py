from __future__ import annotations

from pathlib import Path

from clippilot.rag.loaders import load_knowledge_chunks
from clippilot.rag.providers import BM25Store, ChromaVectorStore, EmbeddingProviderError, QwenCloudEmbeddingProvider
from clippilot.rag.schemas import IndexBuildReport
from clippilot.storage.json_io import write_json_file
from clippilot.storage.path_manager import AppSettings, load_settings


def build_local_index(settings: AppSettings | None = None) -> str:
    """Build the current strategy corpus into BM25 and, when configured, Chroma."""

    active_settings = settings or load_settings()
    chunks = load_knowledge_chunks(active_settings)
    report = IndexBuildReport(
        collection_name=active_settings.rag_collection_name,
        chunk_count=len(chunks),
    )

    if not chunks:
        report.warnings.append("No markdown knowledge files were found for RAG indexing.")
        manifest_path = active_settings.rag_manifest_dir / "index_manifest.json"
        report.manifest_path = str(write_json_file(manifest_path, report).relative_to(active_settings.project_root)).replace("\\", "/")
        return report.manifest_path

    bm25_store = BM25Store(chunks)
    bm25_store.save(active_settings.rag_bm25_dir / "strategy_corpus.json")
    report.bm25_index_built = True

    chroma_store = ChromaVectorStore(active_settings.rag_chroma_dir, active_settings.rag_collection_name)
    if chroma_store.available and active_settings.qwen_api_key:
        try:
            embedding_provider = QwenCloudEmbeddingProvider(active_settings)
            embeddings = embedding_provider.embed_documents([chunk.text for chunk in chunks])
            chroma_store.upsert_chunks(chunks, embeddings)
            report.dense_index_built = True
        except EmbeddingProviderError as exc:
            report.warnings.append(str(exc))
    else:
        if not active_settings.qwen_api_key:
            report.warnings.append("Dense index was skipped because no Qwen embedding API key is configured.")
        if not chroma_store.available:
            report.warnings.append("Dense index was skipped because `chromadb` is not installed.")

    manifest_path = active_settings.rag_manifest_dir / "index_manifest.json"
    report.manifest_path = str(write_json_file(manifest_path, report).relative_to(active_settings.project_root)).replace("\\", "/")
    return report.manifest_path
