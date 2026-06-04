from pathlib import Path

from clippilot.storage.path_manager import AppSettings
from clippilot.schemas.task_result import TaskResult
from clippilot.schemas.user_request import UserRequest
from clippilot.schemas.video_info import VideoInfo
from clippilot.storage.task_storage import TaskStorage


def _workspace_root(test_name: str) -> Path:
    """Return a writable workspace-local directory for filesystem tests."""

    root = Path("tests_runtime") / test_name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build_test_settings(root: Path) -> AppSettings:
    """Create isolated settings for task list tests."""

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


def test_list_tasks_returns_saved_tasks_in_summary_form() -> None:
    """Ensure task listing returns lightweight summaries for saved task results."""

    settings = _build_test_settings(_workspace_root("task_list"))
    storage = TaskStorage(settings)

    for task_id in ("task_a", "task_b"):
        task_paths = storage.create_task_paths(task_id, "demo.mp4")
        task_result = TaskResult(
            task_id=task_id,
            status="completed",
            stage="completed",
            upload_file_path=f"outputs/tasks/{task_id}/input/source.mp4",
            user_params=UserRequest(
                target_platform="bilibili",
                target_duration=30,
                edit_style="powerful",
                language="zh",
                need_burn_subtitle=True,
            ),
            metadata=VideoInfo(
                duration_seconds=300.0,
                width=1920,
                height=1080,
                fps=30.0,
                has_audio=True,
                file_size_mb=128.0,
                source_path=str(task_paths.source_video_path),
            ),
            transcript_path=f"outputs/tasks/{task_id}/transcript/transcript.json",
            candidates_path=f"outputs/tasks/{task_id}/highlights/candidates.json",
            editing_plan_path=f"outputs/tasks/{task_id}/plan/editing_plan.json",
        )
        storage.save_task_result(task_result, task_paths)

    result = storage.list_tasks(limit=10)
    task_ids = {task.task_id for task in result.tasks}
    assert {"task_a", "task_b"}.issubset(task_ids)
