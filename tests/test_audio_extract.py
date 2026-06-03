from pathlib import Path

from clippilot.tools.audio_extract import extract_audio_track


def test_extract_audio_track_returns_wav_path() -> None:
    """Ensure audio extraction path generation stays stable."""

    assert extract_audio_track(Path("demo.mp4")).suffix == ".wav"
