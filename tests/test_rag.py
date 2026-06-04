from pathlib import Path

from clippilot.rag.chunking import chunk_markdown_document
from clippilot.rag.query_builder import duration_band_for
from clippilot.rag.retrieve import retrieve_context
from clippilot.rag.schemas import RetrievalQuery
from clippilot.storage.path_manager import AppSettings


def _workspace_root(test_name: str) -> Path:
    """Return a writable workspace-local directory for filesystem tests."""

    root = Path("tests_runtime") / test_name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build_test_settings(root: Path) -> AppSettings:
    """Create isolated settings for RAG tests."""

    return AppSettings(
        project_root=root,
        config_path=root / "config.yaml",
        app_name="clip-pilot",
        app_version="0.1.0",
        app_description="test",
        tasks_root_dir=root / "outputs" / "tasks",
        raw_videos_root=root / "data" / "raw_videos",
        allowed_extensions={".mp4", ".mov", ".mkv"},
        min_video_duration_seconds=180,
        max_video_duration_seconds=600,
        asr_provider="mock",
        whisper_model="base",
        highlight_min_candidate_duration=8.0,
        highlight_max_candidate_duration=20.0,
        rag_enabled=True,
        rag_knowledge_dir=root / "clippilot" / "rag" / "knowledge",
        rag_data_dir=root / "data" / "rag",
        rag_chroma_dir=root / "data" / "rag" / "chroma",
        rag_bm25_dir=root / "data" / "rag" / "bm25",
        rag_manifest_dir=root / "data" / "rag" / "manifests",
        rag_collection_name="clip_pilot_strategy",
        rag_top_k_dense=8,
        rag_top_k_bm25=8,
        rag_top_k_final=3,
        rag_dense_weight=0.55,
        rag_bm25_weight=0.30,
        rag_metadata_weight=0.15,
        qwen_embedding_model="text-embedding-v4",
        qwen_embedding_dimensions=1024,
        qwen_embedding_base_url="https://dashscope.aliyuncs.com",
        qwen_embedding_api_path="/api/v1/services/embeddings/text-embedding/text-embedding",
        qwen_api_key="",
    )


def test_chunk_markdown_document_preserves_strategy_metadata() -> None:
    """Ensure markdown chunking produces strategy-oriented metadata-rich chunks."""

    source = Path("platform_rules.md")
    chunks = chunk_markdown_document(
        source,
        "# Douyin\n\n## Hook\n\n- 抖音前3秒必须先给结论，再给过程。\n\n## Subtitle\n\n- 中文字幕尽量控制在每行30字以内。\n",
    )

    assert len(chunks) >= 2
    assert any(chunk.strategy_type == "hook_rule" for chunk in chunks)
    assert any(chunk.platform == "douyin" for chunk in chunks)


def test_retrieve_context_uses_bm25_and_metadata_filters_without_dense_deps() -> None:
    """Ensure retrieval still works from local knowledge files when dense deps are absent."""

    settings = _build_test_settings(_workspace_root("rag_retrieve"))
    settings.rag_knowledge_dir.mkdir(parents=True, exist_ok=True)
    (settings.rag_knowledge_dir / "platform_rules.md").write_text(
        "# Douyin Rules\n\n## Hook\n\n- 抖音前3秒必须给结果预告，优先使用高冲击开头。\n",
        encoding="utf-8",
    )
    (settings.rag_knowledge_dir / "editing_templates.md").write_text(
        "# Pacing\n\n## Fast Pace\n\n- 强节奏剪辑优先保留3到12秒的高密度信息片段。\n",
        encoding="utf-8",
    )

    query = RetrievalQuery(
        text="platform douyin | duration 30 seconds | style powerful | language zh | hook",
        platform="douyin",
        language="zh",
        style="powerful",
        duration_band=duration_band_for(30),
        stage="planning",
        top_k_final=3,
    )

    context = retrieve_context(query=query, settings=settings)

    assert context.chunks
    assert context.chunks[0].platform in {"douyin", None}
    assert context.trace.bm25_candidate_count >= 1
