from pathlib import Path

from clippilot.agents import review_agent
from clippilot.agents.review_agent import review_task_output
from clippilot.schemas.execution_report import ExecutionReport
from clippilot.schemas.user_request import UserRequest
from clippilot.schemas.video_info import VideoInfo


def _workspace_root(test_name: str) -> Path:
    """Return a writable workspace-local directory for filesystem tests."""

    root = Path("tests_runtime") / test_name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build_request(language: str = "zh") -> UserRequest:
    """Create a stable user request fixture for review tests."""

    return UserRequest(
        target_platform="bilibili",
        target_duration=30,
        edit_style="powerful",
        language=language,
        need_burn_subtitle=True,
    )


def _build_video_info(root: Path) -> VideoInfo:
    """Create source video metadata for review tests."""

    return VideoInfo(
        duration_seconds=120.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
        file_size_mb=120.0,
        source_path=str(root / "source.mp4"),
    )


def _build_editing_plan(subtitle_text: str = "正常字幕") -> dict:
    """Create a raw editing plan payload that the reviewer can inspect."""

    return {
        "task_id": "task123",
        "target_duration": 30,
        "total_duration": 5.0,
        "clips": [
            {
                "clip_id": "clip_01",
                "source_start": 0.0,
                "source_end": 5.0,
                "duration": 5.0,
                "purpose": "hook",
                "text": subtitle_text,
                "subtitle": subtitle_text,
                "score": 0.9,
                "reason": "Strong opener.",
            }
        ],
    }


def _build_execution_report(root: Path, final_exists: bool = True, subtitle_exists: bool = True) -> ExecutionReport:
    """Create execution outputs for review tests."""

    final_video_path = root / "final_video.mp4"
    subtitle_path = root / "subtitles.srt"

    if final_exists:
        final_video_path.write_bytes(b"video")
    if subtitle_exists:
        subtitle_path.write_text("1\n00:00:00,000 --> 00:00:05,000\nsubtitle\n", encoding="utf-8")

    return ExecutionReport(
        task_id="task123",
        status="completed",
        clip_results=[],
        final_video_path=str(final_video_path),
        subtitle_path=str(subtitle_path),
        burned_video_path=None,
        warnings=[],
        errors=[],
    )


def test_review_fails_when_final_video_is_missing(monkeypatch) -> None:
    """Ensure review fails hard when the final video artifact does not exist."""

    root = _workspace_root("review_missing_final_video")
    execution_report = _build_execution_report(root, final_exists=False, subtitle_exists=True)

    monkeypatch.setattr(review_agent, "extract_video_info", lambda path: _build_video_info(root))

    report = review_task_output(
        user_request=_build_request(),
        original_video_info=_build_video_info(root),
        editing_plan=_build_editing_plan(),
        execution_report=execution_report,
    )

    assert report.passed is False
    assert any(check.name == "final_video_exists" and not check.passed for check in report.checks)


def test_review_fails_when_editing_plan_is_empty(monkeypatch) -> None:
    """Ensure an empty editing plan produces a failed review instead of crashing."""

    root = _workspace_root("review_empty_plan")
    execution_report = _build_execution_report(root, final_exists=True, subtitle_exists=True)

    monkeypatch.setattr(
        review_agent,
        "extract_video_info",
        lambda path: VideoInfo(
            duration_seconds=30.0,
            width=1920,
            height=1080,
            fps=30.0,
            has_audio=True,
            file_size_mb=10.0,
            source_path=str(path),
        ),
    )

    report = review_task_output(
        user_request=_build_request(),
        original_video_info=_build_video_info(root),
        editing_plan={"task_id": "task123", "target_duration": 30, "total_duration": 0.0, "clips": []},
        execution_report=execution_report,
    )

    assert report.passed is False
    assert any(check.name == "editing_plan_has_clips" and not check.passed for check in report.checks)


def test_review_adds_warning_suggestion_for_long_subtitles(monkeypatch) -> None:
    """Ensure long subtitle lines create a warning check and readability suggestion."""

    root = _workspace_root("review_long_subtitle")
    execution_report = _build_execution_report(root, final_exists=True, subtitle_exists=True)

    monkeypatch.setattr(
        review_agent,
        "extract_video_info",
        lambda path: VideoInfo(
            duration_seconds=30.0,
            width=1920,
            height=1080,
            fps=30.0,
            has_audio=True,
            file_size_mb=10.0,
            source_path=str(path),
        ),
    )

    report = review_task_output(
        user_request=_build_request(language="zh"),
        original_video_info=_build_video_info(root),
        editing_plan=_build_editing_plan(subtitle_text="这是一条非常非常长的中文字幕这是一条非常非常长的中文字幕这是一条非常非常长的中文字幕"),
        execution_report=execution_report,
    )

    assert any(check.severity == "warning" and not check.passed for check in report.checks)
    assert any("Split long subtitle lines" in suggestion for suggestion in report.suggestions)
