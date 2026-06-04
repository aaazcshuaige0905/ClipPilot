from pathlib import Path

from clippilot.core.exceptions import ClipPilotProcessingError


def _load_video_file_clip():
    """Load moviepy's VideoFileClip lazily so import failures stay local to audio extraction."""

    try:
        from moviepy import VideoFileClip as clip_class
    except ImportError:  # pragma: no cover
        try:
            from moviepy.video.io.VideoFileClip import VideoFileClip as clip_class
        except ImportError as exc:  # pragma: no cover
            raise ClipPilotProcessingError(
                "moviepy is not installed. Install dependencies before extracting audio."
            ) from exc
    return clip_class


def extract_audio_track(video_path: Path, output_audio_path: Path | None = None) -> Path:
    """Extract the source audio track into a WAV file and return its path."""

    target_path = output_audio_path or video_path.with_suffix(".wav")
    if output_audio_path is None:
        return target_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        video_file_clip = _load_video_file_clip()
        with video_file_clip(str(video_path)) as video_clip:
            if video_clip.audio is None:
                raise ClipPilotProcessingError("The uploaded video does not contain an audio track.")
            video_clip.audio.write_audiofile(str(target_path), logger=None)
    except ClipPilotProcessingError:
        raise
    except Exception as exc:
        raise ClipPilotProcessingError("Failed to extract the audio track from the source video.") from exc

    return target_path
