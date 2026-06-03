from pathlib import Path

from clippilot.core.exceptions import ClipPilotProcessingError
from clippilot.schemas.transcript import TranscriptResult, TranscriptSegment
from clippilot.storage.path_manager import AppSettings, load_settings


def _load_video_file_clip():
    """Load moviepy's VideoFileClip only when transcription needs video duration."""

    try:
        from moviepy import VideoFileClip as clip_class
    except ImportError:  # pragma: no cover
        try:
            from moviepy.video.io.VideoFileClip import VideoFileClip as clip_class
        except ImportError as exc:  # pragma: no cover
            raise ClipPilotProcessingError(
                "moviepy is not installed. Install dependencies before running transcription."
            ) from exc
    return clip_class


def _get_video_duration_seconds(video_path: Path) -> float:
    """Read the total duration of a video file in seconds."""

    try:
        video_file_clip = _load_video_file_clip()
        with video_file_clip(str(video_path)) as video_clip:
            return round(float(video_clip.duration or 0.0), 2)
    except Exception as exc:
        raise ClipPilotProcessingError("Failed to open the video for ASR transcription.") from exc


def mock_transcribe(video_path: Path, task_id: str, language: str = "zh") -> TranscriptResult:
    """Generate deterministic mock transcript segments for local workflow testing."""

    duration_seconds = _get_video_duration_seconds(video_path)
    segment_span = 12.0
    segment_count = max(1, min(12, int(duration_seconds // 30) or 1))
    segments: list[TranscriptSegment] = []

    for index in range(segment_count):
        start = round(index * segment_span, 2)
        end = round(min(duration_seconds, start + segment_span), 2)
        if start >= duration_seconds:
            break

        text = (
            f"[mock-{language}] Segment {index + 1} for task {task_id}. "
            f"This placeholder text simulates ASR output from {start:.2f}s to {end:.2f}s."
        )
        segments.append(TranscriptSegment(start=start, end=end, text=text))

    full_text = " ".join(segment.text for segment in segments)
    return TranscriptResult(video_id=task_id, segments=segments, full_text=full_text, provider="mock")


def whisper_transcribe(
    video_path: Path,
    task_id: str,
    language: str = "zh",
    model_name: str = "base",
) -> TranscriptResult:
    """Run transcription with the Whisper package if it is installed."""

    try:
        import whisper
    except ImportError as exc:
        raise ClipPilotProcessingError(
            "Whisper ASR provider is not installed. Use mock mode or install whisper."
        ) from exc

    model = whisper.load_model(model_name)
    result = model.transcribe(str(video_path), language=language)
    segments = [
        TranscriptSegment(
            start=round(float(segment["start"]), 2),
            end=round(float(segment["end"]), 2),
            text=segment["text"].strip(),
        )
        for segment in result.get("segments", [])
    ]
    return TranscriptResult(
        video_id=task_id,
        segments=segments,
        full_text=result.get("text", "").strip(),
        provider="whisper",
    )


def faster_whisper_transcribe(
    video_path: Path,
    task_id: str,
    language: str = "zh",
    model_name: str = "base",
) -> TranscriptResult:
    """Run transcription with faster-whisper if it is installed."""

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ClipPilotProcessingError(
            "faster-whisper ASR provider is not installed. Use mock mode or install faster-whisper."
        ) from exc

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    raw_segments, _ = model.transcribe(str(video_path), language=language)
    segments = [
        TranscriptSegment(
            start=round(float(segment.start), 2),
            end=round(float(segment.end), 2),
            text=segment.text.strip(),
        )
        for segment in raw_segments
    ]
    full_text = " ".join(segment.text for segment in segments).strip()
    return TranscriptResult(
        video_id=task_id,
        segments=segments,
        full_text=full_text,
        provider="faster-whisper",
    )


def transcribe_video(
    video_path: Path,
    task_id: str,
    language: str = "zh",
    provider: str | None = None,
    settings: AppSettings | None = None,
) -> TranscriptResult:
    """Transcribe a source video with the configured provider and return a structured transcript."""

    active_settings = settings or load_settings()
    active_provider = (provider or active_settings.asr_provider).strip().lower()

    if active_provider == "mock":
        return mock_transcribe(video_path=video_path, task_id=task_id, language=language)
    if active_provider == "whisper":
        return whisper_transcribe(
            video_path=video_path,
            task_id=task_id,
            language=language,
            model_name=active_settings.whisper_model,
        )
    if active_provider in {"faster-whisper", "faster_whisper"}:
        return faster_whisper_transcribe(
            video_path=video_path,
            task_id=task_id,
            language=language,
            model_name=active_settings.whisper_model,
        )

    raise ClipPilotProcessingError(f"Unsupported ASR provider: {active_provider}.")
