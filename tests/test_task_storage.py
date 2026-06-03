from pathlib import Path

from clippilot.core.task_context import TaskContext
from clippilot.schemas.user_request import UserRequest
from clippilot.storage.path_manager import AppSettings, build_task_paths
from clippilot.storage.task_storage import TaskStorage


def _workspace_root(test_name: str) -> Path:
    """Return a writable workspace-local directory for filesystem tests."""

    root = Path("tests_runtime") / test_name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build_test_settings(root: Path) -> AppSettings:
    """Create isolated settings for task storage tests."""

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
    )


def test_task_storage_saves_artifact_manifest() -> None:
    """Ensure the new task manifest is persisted inside the task directory."""

    settings = _build_test_settings(_workspace_root("task_storage_manifest"))
    storage = TaskStorage(settings)
    task_paths = storage.create_task_paths("task123", "demo.mp4")
    context = TaskContext(
        task_id="task123",
        request=UserRequest(
            target_platform="bilibili",
            target_duration=30,
            edit_style="powerful",
            language="zh",
            need_burn_subtitle=True,
        ),
        paths=task_paths,
    )
    context.mark_stage("completed")
    context.add_artifact("transcript", "outputs/tasks/task123/transcript/transcript.json")

    manifest_path = storage.save_artifact_manifest(context)
    assert manifest_path.exists()
    assert manifest_path == task_paths.artifact_manifest_path


def test_build_task_paths_includes_trace_and_manifest_files() -> None:
    """Ensure task path generation includes second-wave trace and manifest locations."""

    settings = _build_test_settings(_workspace_root("task_storage_paths"))
    task_paths = build_task_paths(settings, "task999", "source.mp4")
    assert task_paths.clips_dir.name == "clips"
    assert task_paths.final_dir.name == "final"
    assert task_paths.artifact_manifest_path.name == "artifact_manifest.json"
    assert task_paths.trace_log_path.name == "workflow_trace.jsonl"
    assert task_paths.execution_report_path.name == "execution_report.json"
    assert task_paths.review_report_path.name == "review_report.json"
    assert task_paths.final_video_path.name == "final_video.mp4"
    assert task_paths.subtitle_path.name == "subtitles.srt"
