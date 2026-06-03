from pathlib import Path


def extract_audio_track(video_path: Path) -> Path:
    """Return the future audio extraction target path for a source video."""

    return video_path.with_suffix(".wav")
