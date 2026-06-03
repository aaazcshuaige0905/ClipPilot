from pathlib import Path
import shutil
import subprocess

import pytest

from clippilot.core.exceptions import ClipPilotProcessingError
from clippilot.tools.video_cut import cut_video_clip
from clippilot.tools.video_merge import merge_video_clips, merge_video_segments


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
        "8",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def test_merge_video_segments_returns_requested_output_path() -> None:
    """Ensure the lightweight path helper still preserves the chosen output destination."""

    output_path = Path("merged.mp4")
    assert merge_video_segments([], output_path) == output_path


def test_merge_video_clips_rejects_empty_clip_list() -> None:
    """Ensure the real merge API fails fast when no input clips are provided."""

    with pytest.raises(ClipPilotProcessingError):
        merge_video_clips([], "merged.mp4")


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg is not installed")
def test_merge_video_clips_creates_final_video_with_ffmpeg() -> None:
    """Ensure ffmpeg concat merge creates the requested final video."""

    root = _workspace_root("video_merge")
    source_video = root / "source.mp4"
    clips_dir = root / "clips"
    clip_a = clips_dir / "clip_a.mp4"
    clip_b = clips_dir / "clip_b.mp4"
    final_video = root / "final_video.mp4"

    _create_sample_video(source_video)

    cut_result_a = cut_video_clip(str(source_video), str(clip_a), 0.0, 3.0)
    cut_result_b = cut_video_clip(str(source_video), str(clip_b), 3.0, 6.0)
    assert cut_result_a.success is True
    assert cut_result_b.success is True

    merge_result = merge_video_clips(
        clip_paths=[str(clip_a), str(clip_b)],
        output_video_path=str(final_video),
    )

    assert merge_result.success is True
    assert merge_result.clip_count == 2
    assert final_video.exists()
    assert Path(merge_result.final_video_path) == final_video
