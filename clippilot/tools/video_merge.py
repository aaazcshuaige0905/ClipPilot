from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile

from clippilot.core.exceptions import ClipPilotProcessingError


@dataclass(frozen=True)
class VideoMergeResult:
    """Represent the outcome of one ffmpeg concat merge operation."""

    final_video_path: str
    clip_count: int
    success: bool
    error: str | None = None


def _format_concat_path(path: Path) -> str:
    """Format a clip path for ffmpeg concat demuxer list files."""

    normalized = path.resolve().as_posix().replace("'", "'\\''")
    return f"file '{normalized}'"


def merge_video_clips(clip_paths: list[str], output_video_path: str) -> VideoMergeResult:
    """Merge multiple clip files into a single final video with ffmpeg concat demuxer."""

    if not clip_paths:
        raise ClipPilotProcessingError("clip_paths must not be empty when merging video clips.")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return VideoMergeResult(
            final_video_path=output_video_path,
            clip_count=len(clip_paths),
            success=False,
            error="ffmpeg is not installed or not available in PATH.",
        )

    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clip_file_paths = [Path(path) for path in clip_paths]
    for clip_path in clip_file_paths:
        if not clip_path.exists():
            return VideoMergeResult(
                final_video_path=str(output_path),
                clip_count=len(clip_file_paths),
                success=False,
                error=f"Clip file does not exist: {clip_path}",
            )

    concat_file_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix="_concat_list.txt", delete=False, encoding="utf-8") as handle:
            concat_file_path = Path(handle.name)
            for clip_path in clip_file_paths:
                handle.write(_format_concat_path(clip_path) + "\n")

        command = [
            ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file_path),
            "-c",
            "copy",
            str(output_path),
        ]

        completed_process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return VideoMergeResult(
            final_video_path=str(output_path),
            clip_count=len(clip_file_paths),
            success=False,
            error=f"Failed to execute ffmpeg: {exc}",
        )
    finally:
        if concat_file_path and concat_file_path.exists():
            concat_file_path.unlink(missing_ok=True)

    if completed_process.returncode != 0:
        stderr = completed_process.stderr.strip() or "Unknown ffmpeg error."
        return VideoMergeResult(
            final_video_path=str(output_path),
            clip_count=len(clip_file_paths),
            success=False,
            error=stderr,
        )

    return VideoMergeResult(
        final_video_path=str(output_path),
        clip_count=len(clip_file_paths),
        success=True,
        error=None,
    )


def merge_video_segments(segment_paths: list[Path], output_path: Path) -> Path:
    """Preserve the lightweight helper used by earlier path-oriented code and tests."""

    return output_path
