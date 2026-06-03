from pathlib import Path
import shutil
import subprocess

import pytest

from clippilot.tools.video_cut import cut_video_clip, cut_video_segment


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
        "sine=frequency=440:sample_rate=44100",
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


def test_cut_video_segment_returns_predicted_output_path() -> None:
    """Ensure clip output naming stays deterministic."""

    output = cut_video_segment(Path("demo.mp4"), 1.0, 3.5)
    assert output.name == "demo_1.00_3.50.mp4"


def test_cut_video_clip_rejects_invalid_time_range() -> None:
    """Ensure the real cut API reports invalid time ranges as structured failures."""

    result = cut_video_clip(
        input_video_path="demo.mp4",
        output_clip_path="clip.mp4",
        start_time=5.0,
        end_time=3.0,
    )
    assert result.success is False
    assert result.duration == 0.0
    assert result.error is not None


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg is not installed")
def test_cut_video_clip_creates_output_file_with_ffmpeg() -> None:
    """Ensure ffmpeg-based cutting creates a playable clip file."""

    root = _workspace_root("video_cut")
    source_video = root / "source.mp4"
    output_clip = root / "clips" / "clip_01.mp4"
    _create_sample_video(source_video)

    result = cut_video_clip(
        input_video_path=str(source_video),
        output_clip_path=str(output_clip),
        start_time=1.0,
        end_time=4.0,
    )

    assert result.success is True
    assert output_clip.exists()
    assert result.duration == 3.0
    assert Path(result.clip_path) == output_clip
