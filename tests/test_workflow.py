from pathlib import Path
import sys
import types

_fake_moviepy = types.ModuleType("moviepy")


class _FakeVideoFileClip:
    """Provide a lightweight moviepy stand-in for workflow unit tests."""


_fake_moviepy.VideoFileClip = _FakeVideoFileClip
sys.modules.setdefault("moviepy", _fake_moviepy)
sys.modules.setdefault("moviepy.video", types.ModuleType("moviepy.video"))
sys.modules.setdefault("moviepy.video.io", types.ModuleType("moviepy.video.io"))
_fake_moviepy_video_file_clip = types.ModuleType("moviepy.video.io.VideoFileClip")
_fake_moviepy_video_file_clip.VideoFileClip = _FakeVideoFileClip
sys.modules.setdefault("moviepy.video.io.VideoFileClip", _fake_moviepy_video_file_clip)

from clippilot.core.task_context import TaskContext
from clippilot.core.workflow import (
    _build_task_result,
    _process_execution_report,
    _process_review_report,
    _process_video_understanding,
)
from clippilot.schemas.editing_plan import EditingClip, EditingPlan
from clippilot.schemas.execution_report import ExecutionReport
from clippilot.schemas.project_state import FineGrainedUnit, ProjectPaths, ProjectState
from clippilot.schemas.review_report import ReviewCheck, ReviewReport
from clippilot.schemas.user_request import UserRequest
from clippilot.schemas.video_info import VideoInfo
from clippilot.schemas.transcript import TranscriptResult, TranscriptSegment
from clippilot.storage.path_manager import AppSettings, load_settings
from clippilot.storage.task_storage import TaskStorage


def _workspace_root(test_name: str) -> Path:
    """Return a writable workspace-local directory for filesystem tests."""

    root = Path("tests_runtime") / test_name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build_test_settings(root: Path) -> AppSettings:
    """Create isolated settings for workflow tests."""

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


def _build_request() -> UserRequest:
    """Create a stable user request fixture for workflow tests."""

    return UserRequest(
        target_platform="bilibili",
        target_duration=30,
        edit_style="powerful",
        language="zh",
        need_burn_subtitle=True,
    )


def _build_editing_plan(task_id: str) -> EditingPlan:
    """Create a minimal editing plan for workflow tests."""

    return EditingPlan(
        task_id=task_id,
        target_duration=10,
        total_duration=3.0,
        clips=[
            EditingClip(
                clip_id="clip_01",
                source_start=0.0,
                source_end=3.0,
                duration=3.0,
                purpose="hook",
                text="Opening hook.",
                subtitle="Opening hook.",
                score=0.9,
                reason="Strong opener.",
            )
        ],
        editing_notes=["Workflow test plan."],
    )


def test_load_settings_reads_app_name() -> None:
    """Ensure the new settings loader can read the project configuration."""

    settings = load_settings()
    assert settings.app_name == "clip-pilot"


def test_build_task_result_includes_execution_output_paths() -> None:
    """Ensure task results expose the saved execution artifact paths."""

    settings = _build_test_settings(_workspace_root("workflow_task_result"))
    storage = TaskStorage(settings)
    task_paths = storage.create_task_paths("task123", "source.mp4")
    context = TaskContext(task_id="task123", request=_build_request(), paths=task_paths)
    video_info = VideoInfo(
        duration_seconds=300.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
        file_size_mb=120.0,
        source_path=str(task_paths.source_video_path),
    )
    execution_report = ExecutionReport(
        task_id="task123",
        status="completed",
        clip_results=[],
        final_video_path=str(task_paths.final_video_path),
        subtitle_path=str(task_paths.subtitle_path),
        burned_video_path=str(task_paths.burned_video_path),
        warnings=[],
        errors=[],
    )

    task_result = _build_task_result(
        storage=storage,
        context=context,
        video_info=video_info,
        execution_report=execution_report,
    )

    assert task_result.execution_report_path == "outputs/tasks/task123/plan/execution_report.json"
    assert task_result.final_video_path == "outputs/tasks/task123/final/final_video.mp4"
    assert task_result.subtitle_path == "outputs/tasks/task123/final/subtitles.srt"
    assert task_result.burned_video_path == "outputs/tasks/task123/final/final_video_burned.mp4"
    assert task_result.project_state_path == "outputs/tasks/task123/project_state.json"


def test_process_execution_report_registers_generated_artifacts(monkeypatch) -> None:
    """Ensure workflow tracks execution outputs in the task artifact inventory."""

    settings = _build_test_settings(_workspace_root("workflow_execution_artifacts"))
    storage = TaskStorage(settings)
    task_paths = storage.create_task_paths("task123", "source.mp4")
    context = TaskContext(task_id="task123", request=_build_request(), paths=task_paths)
    editing_plan = _build_editing_plan("task123")

    task_paths.final_video_path.write_bytes(b"video")
    task_paths.subtitle_path.write_text("1\n00:00:00,000 --> 00:00:03,000\nHi\n", encoding="utf-8")
    task_paths.burned_video_path.write_bytes(b"burned")

    report = ExecutionReport(
        task_id="task123",
        status="completed",
        clip_results=[],
        final_video_path=str(task_paths.final_video_path),
        subtitle_path=str(task_paths.subtitle_path),
        burned_video_path=str(task_paths.burned_video_path),
        warnings=[],
        errors=[],
    )

    monkeypatch.setattr("clippilot.core.workflow.execute_editing_plan", lambda **kwargs: report)

    saved_report = _process_execution_report(
        storage=storage,
        context=context,
        editing_plan=editing_plan,
    )
    manifest_path = storage.save_artifact_manifest(context)
    manifest = storage.load_artifact_manifest("task123")

    assert saved_report.final_video_path == str(task_paths.final_video_path)
    assert manifest_path.exists()
    assert {artifact.name for artifact in manifest.artifacts} >= {
        "execution_report",
        "final_video",
        "subtitle_file",
        "burned_video",
    }


def test_process_review_report_saves_real_review_output(monkeypatch) -> None:
    """Ensure workflow persists the structured review report from the real review agent entrypoint."""

    settings = _build_test_settings(_workspace_root("workflow_review_report"))
    storage = TaskStorage(settings)
    task_paths = storage.create_task_paths("task123", "source.mp4")
    context = TaskContext(task_id="task123", request=_build_request(), paths=task_paths)
    editing_plan = _build_editing_plan("task123")
    video_info = VideoInfo(
        duration_seconds=300.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
        file_size_mb=120.0,
        source_path=str(task_paths.source_video_path),
    )
    execution_report = ExecutionReport(
        task_id="task123",
        status="completed",
        clip_results=[],
        final_video_path=str(task_paths.final_video_path),
        subtitle_path=str(task_paths.subtitle_path),
        burned_video_path=None,
        warnings=[],
        errors=[],
    )
    review_report = ReviewReport(
        task_id="task123",
        passed=True,
        score=0.9,
        checks=[ReviewCheck(name="final_video_exists", passed=True, severity="critical", message="ok")],
        suggestions=[],
    )

    monkeypatch.setattr("clippilot.core.workflow.review_task_output", lambda **kwargs: review_report)

    saved_report = _process_review_report(
        storage=storage,
        context=context,
        video_info=video_info,
        editing_plan=editing_plan,
        execution_report=execution_report,
    )

    assert saved_report.score == 0.9
    assert task_paths.review_report_path.exists()
    assert any(artifact.name == "review_report" for artifact in context.artifacts)


def test_process_video_understanding_persists_timeline_and_project_state() -> None:
    """Ensure the video understanding step writes timeline output into the shared project state."""

    settings = _build_test_settings(_workspace_root("workflow_video_understanding"))
    storage = TaskStorage(settings)
    task_paths = storage.create_task_paths("task123", "source.mp4")
    project_state = ProjectState(
        project_id="task123",
        user_request=_build_request(),
        paths=ProjectPaths(raw_video="outputs/tasks/task123/input/source.mp4"),
        fine_grained_units=[
            FineGrainedUnit(
                unit_id="unit_001",
                start=0.0,
                end=5.0,
                duration=5.0,
                text="Opening scene.",
                has_pause_after=True,
                keywords=["opening"],
            ),
            FineGrainedUnit(
                unit_id="unit_002",
                start=5.4,
                end=10.0,
                duration=4.6,
                text="Second scene.",
                has_pause_before=True,
                keywords=["second"],
            ),
        ],
    )
    context = TaskContext(
        task_id="task123",
        request=_build_request(),
        paths=task_paths,
        project_state=project_state,
    )
    video_info = VideoInfo(
        duration_seconds=300.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
        file_size_mb=120.0,
        source_path=str(task_paths.source_video_path),
    )
    transcript = TranscriptResult(
        video_id="task123",
        segments=[
            TranscriptSegment(start=0.0, end=5.0, text="Opening scene."),
            TranscriptSegment(start=5.0, end=10.0, text="Second scene."),
        ],
        full_text="Opening scene. Second scene.",
        provider="mock",
    )

    timeline, highlight_candidates = _process_video_understanding(
        storage=storage,
        context=context,
        video_info=video_info,
        transcript=transcript,
    )
    loaded_state = storage.load_project_state("task123")

    assert timeline.provider == "qwen"
    assert task_paths.timeline_path.exists()
    assert loaded_state.timeline is not None
    assert loaded_state.paths.timeline == "outputs/tasks/task123/understanding/timeline.json"
    assert loaded_state.highlight_candidates_llm is not None
    assert len(highlight_candidates) == 2
