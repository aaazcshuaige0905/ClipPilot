from pathlib import Path

from clippilot.core.exceptions import ClipPilotProcessingError, ClipPilotValidationError
from clippilot.schemas.video_info import VideoInfo
from clippilot.storage.path_manager import AppSettings, load_settings


def _load_video_file_clip():
    """Load moviepy's VideoFileClip only when metadata extraction actually runs."""

    try:
        from moviepy import VideoFileClip as clip_class
    except ImportError:  # pragma: no cover
        try:
            from moviepy.video.io.VideoFileClip import VideoFileClip as clip_class
        except ImportError as exc:  # pragma: no cover
            raise ClipPilotProcessingError(
                "moviepy is not installed. Install dependencies before extracting video metadata."
            ) from exc
    return clip_class


def validate_supported_extension(file_name: str, allowed_extensions: set[str]) -> str:
    """Validate the source extension and return a normalized suffix."""

    suffix = Path(file_name).suffix.lower()
    if suffix not in allowed_extensions:
        supported_formats = ", ".join(sorted(ext.lstrip(".") for ext in allowed_extensions))
        raise ClipPilotValidationError(
            f"Unsupported video format. Only {supported_formats} files are allowed."
        )
    return suffix


def extract_video_info(video_path: Path) -> VideoInfo:
    """Extract video metadata from a saved video file using moviepy."""

    try:
        video_file_clip = _load_video_file_clip()
        with video_file_clip(str(video_path)) as video_clip:
            duration_seconds = round(float(video_clip.duration or 0.0), 2)
            width = int(video_clip.w)
            height = int(video_clip.h)
            fps = round(float(video_clip.fps or 0.0), 2)
            has_audio = video_clip.audio is not None
    except Exception as exc:
        raise ClipPilotProcessingError(
            "Failed to read video metadata. Please upload a valid MP4, MOV, or MKV file."
        ) from exc

    file_size_mb = round(video_path.stat().st_size / (1024 * 1024), 2)

    return VideoInfo(
        duration_seconds=duration_seconds,
        width=width,
        height=height,
        fps=fps,
        has_audio=has_audio,
        file_size_mb=file_size_mb,
        source_path=str(video_path),
    )


def validate_video_duration(video_info: VideoInfo, settings: AppSettings | None = None) -> None:
    """Validate that a source video duration is inside the configured supported range."""

    active_settings = settings or load_settings()
    if (
        video_info.duration_seconds < active_settings.min_video_duration_seconds
        or video_info.duration_seconds > active_settings.max_video_duration_seconds
    ):
        raise ClipPilotValidationError(
            f"Video duration must be between {active_settings.min_video_duration_seconds} "
            f"and {active_settings.max_video_duration_seconds} seconds."
        )
