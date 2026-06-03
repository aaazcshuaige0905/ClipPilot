from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess


@dataclass(frozen=True)
class VideoCutResult:
    """Represent the outcome of one ffmpeg-based video cut operation."""

    clip_path: str
    duration: float
    success: bool
    error: str | None = None


def _build_predicted_clip_path(video_path: Path, start: float, end: float) -> Path:
    """Build a deterministic clip file name from a source file and time range."""

    return video_path.with_name(f"{video_path.stem}_{start:.2f}_{end:.2f}{video_path.suffix}")


def _validate_cut_range(start_time: float, end_time: float) -> str | None:
    """Validate clip timing inputs and return a human-readable error when invalid."""

    if start_time < 0:
        return "start_time must be greater than or equal to 0."
    if end_time <= start_time:
        return "end_time must be greater than start_time."
    return None


def cut_video_clip(
    input_video_path: str,
    output_clip_path: str,
    start_time: float,
    end_time: float,
) -> VideoCutResult:
    """Cut one source-video range into a standalone clip by calling ffmpeg."""

    validation_error = _validate_cut_range(start_time=start_time, end_time=end_time)
    if validation_error:
        return VideoCutResult(
            clip_path=output_clip_path,
            duration=0.0,
            success=False,
            error=validation_error,
        )

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return VideoCutResult(
            clip_path=output_clip_path,
            duration=0.0,
            success=False,
            error="ffmpeg is not installed or not available in PATH.",
        )

    input_path = Path(input_video_path)
    output_path = Path(output_clip_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-ss",
        str(start_time),
        "-to",
        str(end_time),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    try:
        completed_process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return VideoCutResult(
            clip_path=str(output_path),
            duration=0.0,
            success=False,
            error=f"Failed to execute ffmpeg: {exc}",
        )

    if completed_process.returncode != 0:
        stderr = completed_process.stderr.strip() or "Unknown ffmpeg error."
        return VideoCutResult(
            clip_path=str(output_path),
            duration=0.0,
            success=False,
            error=stderr,
        )

    return VideoCutResult(
        clip_path=str(output_path),
        duration=round(end_time - start_time, 2),
        success=True,
        error=None,
    )


def cut_video_segment(video_path: Path, start: float, end: float) -> Path:
    """Return the deterministic output path that matches the clip naming rule."""

    return _build_predicted_clip_path(video_path, start, end)
