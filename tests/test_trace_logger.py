from pathlib import Path

from clippilot.core.task_context import TaskContext
from clippilot.harness.trace_logger import log_stage_transition
from clippilot.schemas.user_request import UserRequest
from clippilot.storage.path_manager import AppSettings, build_task_paths


def _workspace_root(test_name: str) -> Path:
    """Return a writable workspace-local directory for filesystem tests."""

    root = Path("tests_runtime") / test_name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build_test_settings(root: Path) -> AppSettings:
    """Create isolated settings for trace logger tests."""

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


def test_log_stage_transition_adds_trace_entries() -> None:
    """Ensure stage transitions are recorded as structured trace entries."""

    settings = _build_test_settings(_workspace_root("trace_logger"))
    context = TaskContext(
        task_id="task123",
        request=UserRequest(
            target_platform="bilibili",
            target_duration=30,
            edit_style="powerful",
            language="zh",
            need_burn_subtitle=True,
        ),
        paths=build_task_paths(settings, "task123", "demo.mp4"),
    )

    log_stage_transition(context, "uploaded", "Uploaded source video saved.")
    assert context.stage == "uploaded"
    assert len(context.traces) == 1
    assert context.traces[0].stage == "uploaded"
