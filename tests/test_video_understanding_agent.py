from pathlib import Path

from clippilot.agents.video_understanding_agent import (
    analyze_video_understanding,
    build_qwen_video_understanding_request,
)
from clippilot.schemas.project_state import FineGrainedUnit
from clippilot.schemas.transcript import TranscriptResult, TranscriptSegment
from clippilot.schemas.user_request import UserRequest
from clippilot.schemas.video_info import VideoInfo


def _build_request() -> UserRequest:
    """Create a stable request fixture for video understanding tests."""

    return UserRequest(
        target_platform="bilibili",
        target_duration=30,
        edit_style="healing",
        language="zh",
        need_burn_subtitle=True,
    )


def _build_transcript() -> TranscriptResult:
    """Create a transcript fixture for video understanding tests."""

    return TranscriptResult(
        video_id="task123",
        segments=[
            TranscriptSegment(start=0.0, end=5.0, text="Opening scene with the main point."),
            TranscriptSegment(start=5.0, end=10.0, text="Follow-up explanation."),
        ],
        full_text="Opening scene with the main point. Follow-up explanation.",
        provider="mock",
    )


def _build_video_info() -> VideoInfo:
    """Create a source video metadata fixture for video understanding tests."""

    return VideoInfo(
        duration_seconds=300.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
        file_size_mb=100.0,
        source_path="outputs/tasks/task123/input/source.mp4",
    )


def _build_fine_grained_units() -> list[FineGrainedUnit]:
    """Create fine-grained transcript-backed units for video understanding tests."""

    return [
        FineGrainedUnit(
            unit_id="unit_001",
            start=0.0,
            end=5.0,
            duration=5.0,
            text="Opening scene with the main point.",
            has_pause_after=True,
            keywords=["opening", "point"],
        ),
        FineGrainedUnit(
            unit_id="unit_002",
            start=5.4,
            end=10.0,
            duration=4.6,
            text="Follow-up explanation.",
            has_pause_before=True,
            keywords=["followup", "explanation"],
        ),
    ]


def test_build_qwen_video_understanding_request_contains_expected_contract() -> None:
    """Ensure the future Qwen request payload is shaped consistently."""

    payload = build_qwen_video_understanding_request(
        video_path=Path("outputs/tasks/task123/input/source.mp4"),
        user_request=_build_request(),
        transcript=_build_transcript(),
        video_info=_build_video_info(),
        fine_grained_units=_build_fine_grained_units(),
    )

    assert payload["provider"] == "qwen"
    assert payload["model"]
    assert payload["messages"][1]["content"]["video_path"].endswith("source.mp4")
    assert payload["messages"][1]["content"]["transcript_preview"][0]["text"].startswith("Opening scene")
    assert payload["messages"][1]["content"]["fine_grained_units"][0]["unit_id"] == "unit_001"


def test_analyze_video_understanding_returns_stubbed_timeline_and_candidates() -> None:
    """Ensure the current placeholder implementation produces both coarse and fine outputs."""

    content_analysis, timeline, highlight_candidates = analyze_video_understanding(
        video_path=Path("outputs/tasks/task123/input/source.mp4"),
        user_request=_build_request(),
        video_info=_build_video_info(),
        transcript=_build_transcript(),
        fine_grained_units=_build_fine_grained_units(),
    )

    assert content_analysis.summary
    assert timeline.provider == "qwen"
    assert timeline.generation_mode == "stub"
    assert len(timeline.segments) == 2
    assert len(highlight_candidates) == 2
    assert highlight_candidates[0].transcript_unit_ids == ["unit_001"]
