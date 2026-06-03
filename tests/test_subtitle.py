from pathlib import Path
import shutil
import subprocess

import pytest

from clippilot.schemas.editing_plan import EditingClip, EditingPlan
from clippilot.tools.subtitle import (
    burn_subtitles_to_video,
    generate_srt_from_editing_plan,
    seconds_to_srt_time,
)


def _workspace_root(test_name: str) -> Path:
    """Return a writable workspace-local directory for filesystem tests."""

    root = Path("tests_runtime") / test_name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ffmpeg_available() -> bool:
    """Return whether ffmpeg is available in the current test environment."""

    return shutil.which("ffmpeg") is not None


def _create_sample_video(output_path: Path) -> None:
    """Create a small integration-test video with ffmpeg test sources."""

    command = [
        shutil.which("ffmpeg") or "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=320x240:rate=24",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=660:sample_rate=44100",
        "-t",
        "6",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def _build_editing_plan() -> EditingPlan:
    """Create a simple editing plan fixture for subtitle tests."""

    return EditingPlan(
        task_id="task123",
        target_duration=15,
        total_duration=12.0,
        clips=[
            EditingClip(
                clip_id="clip_01",
                source_start=5.0,
                source_end=9.0,
                duration=4.0,
                purpose="hook",
                text="先说结论，这个方法最重要。",
                subtitle="先说结论，这个方法最重要。",
                score=0.91,
                reason="Strong hook.",
            ),
            EditingClip(
                clip_id="clip_02",
                source_start=20.0,
                source_end=28.0,
                duration=8.0,
                purpose="core_point",
                text="Why this method works in real scenarios.",
                subtitle="Why this method works in real scenarios.",
                score=0.84,
                reason="Clear supporting explanation.",
            ),
        ],
        editing_notes=["Generated for subtitle tests."],
    )


def test_seconds_to_srt_time_formats_zero() -> None:
    """Ensure 0 seconds is formatted as the SRT zero timestamp."""

    assert seconds_to_srt_time(0) == "00:00:00,000"


def test_seconds_to_srt_time_formats_fractional_seconds() -> None:
    """Ensure fractional second values keep millisecond precision in SRT output."""

    assert seconds_to_srt_time(65.5) == "00:01:05,500"


def test_generate_srt_from_editing_plan_uses_new_timeline() -> None:
    """Ensure subtitles are mapped onto the new merged-video timeline, not source-video offsets."""

    root = _workspace_root("subtitle_srt")
    output_srt = root / "subtitles" / "subtitles.srt"
    result = generate_srt_from_editing_plan(_build_editing_plan(), str(output_srt))
    content = output_srt.read_text(encoding="utf-8")
    blocks = [block for block in content.strip().split("\n\n") if block.strip()]

    assert result.success is True
    assert result.subtitle_count == 2
    assert output_srt.exists()
    assert blocks[0].splitlines()[1] == "00:00:00,000 --> 00:00:04,000"
    assert blocks[1].splitlines()[1] == "00:00:04,000 --> 00:00:12,000"


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg is not installed")
def test_burn_subtitles_to_video_creates_output() -> None:
    """Ensure ffmpeg can burn a generated SRT file into a rendered video output."""

    root = _workspace_root("subtitle_burn")
    source_video = root / "source.mp4"
    subtitle_path = root / "subtitles.srt"
    burned_video_path = root / "final_with_subtitles.mp4"

    _create_sample_video(source_video)
    generate_srt_from_editing_plan(_build_editing_plan(), str(subtitle_path))
    result = burn_subtitles_to_video(
        input_video_path=str(source_video),
        subtitle_path=str(subtitle_path),
        output_video_path=str(burned_video_path),
    )

    assert result.success is True
    assert burned_video_path.exists()
    assert Path(result.burned_video_path) == burned_video_path
