import os
from pathlib import Path

from clippilot.core.exceptions import ClipPilotProcessingError
from clippilot.schemas.transcript import TranscriptResult, TranscriptSegment
from clippilot.storage.path_manager import AppSettings, load_settings


def _load_video_file_clip():
    """Load moviepy's VideoFileClip only when transcription needs media duration."""

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


def _load_audio_file_clip():
    """Load moviepy's AudioFileClip only when transcription needs audio duration."""

    try:
        from moviepy import AudioFileClip as clip_class
    except ImportError:  # pragma: no cover
        try:
            from moviepy.audio.io.AudioFileClip import AudioFileClip as clip_class
        except ImportError as exc:  # pragma: no cover
            raise ClipPilotProcessingError(
                "moviepy is not installed. Install dependencies before running transcription."
            ) from exc
    return clip_class


def _get_media_duration_seconds(media_path: Path) -> float:
    """Read the total duration of a video or audio file in seconds."""

    audio_suffixes = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
    try:
        if media_path.suffix.lower() in audio_suffixes:
            audio_file_clip = _load_audio_file_clip()
            with audio_file_clip(str(media_path)) as audio_clip:
                return round(float(audio_clip.duration or 0.0), 2)

        video_file_clip = _load_video_file_clip()
        with video_file_clip(str(media_path)) as video_clip:
            return round(float(video_clip.duration or 0.0), 2)
    except Exception as exc:
        raise ClipPilotProcessingError("Failed to open the media source for ASR transcription.") from exc


def _normalize_asr_language(language: str | None) -> str | None:
    """Map user-facing language labels into Whisper-compatible language codes."""

    normalized = (language or "").strip().lower()
    if not normalized or normalized in {"auto", "automatic", "detect", "auto-detect"}:
        return None

    aliases = {
        "zh": "zh",
        "zh-cn": "zh",
        "zh-hans": "zh",
        "cn": "zh",
        "chinese": "zh",
        "中文": "zh",
        "en": "en",
        "en-us": "en",
        "en-gb": "en",
        "english": "en",
        "英语": "en",
        "ja": "ja",
        "japanese": "ja",
        "jp": "ja",
        "ko": "ko",
        "korean": "ko",
        "fr": "fr",
        "french": "fr",
        "de": "de",
        "german": "de",
        "es": "es",
        "spanish": "es",
    }
    return aliases.get(normalized, normalized)


def build_transcript_result(
    *,
    video_id: str,
    provider: str,
    segments: list[TranscriptSegment] | None = None,
    full_text: str = "",
    status: str = "completed",
    warnings: list[str] | None = None,
) -> TranscriptResult:
    """Build one normalized transcript payload, including empty/optional ASR outcomes."""

    normalized_segments = segments or []
    normalized_full_text = full_text.strip()
    normalized_warnings = list(warnings or [])
    normalized_status = status

    if normalized_status == "completed" and (not normalized_segments or not normalized_full_text):
        normalized_status = "no_speech"
        normalized_segments = []
        normalized_full_text = ""
        normalized_warnings.append("ASR completed without any usable speech segments.")

    return TranscriptResult(
        video_id=video_id,
        segments=normalized_segments,
        full_text=normalized_full_text,
        provider=provider,
        status=normalized_status,
        warnings=normalized_warnings,
    )


def build_skipped_transcript(
    *,
    video_id: str,
    provider: str,
    status: str,
    warning: str,
) -> TranscriptResult:
    """Build one empty transcript payload for no-audio or skipped-ASR scenarios."""

    return build_transcript_result(
        video_id=video_id,
        provider=provider,
        segments=[],
        full_text="",
        status=status,
        warnings=[warning],
    )


def mock_transcribe(video_path: Path, task_id: str, language: str = "zh") -> TranscriptResult:
    """Generate deterministic mock transcript segments for local workflow testing."""

    duration_seconds = _get_media_duration_seconds(video_path)
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
    return build_transcript_result(
        video_id=task_id,
        provider="mock",
        segments=segments,
        full_text=full_text,
        status="completed",
    )


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
    normalized_language = _normalize_asr_language(language)
    transcribe_kwargs = {"language": normalized_language} if normalized_language else {}
    result = model.transcribe(str(video_path), **transcribe_kwargs)
    segments = [
        TranscriptSegment(
            start=round(float(segment["start"]), 2),
            end=round(float(segment["end"]), 2),
            text=segment["text"].strip(),
        )
        for segment in result.get("segments", [])
        if segment.get("text", "").strip() and float(segment["end"]) > float(segment["start"])
    ]
    return build_transcript_result(
        video_id=task_id,
        provider="whisper",
        segments=segments,
        full_text=result.get("text", "").strip(),
        status="completed",
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

    normalized_language = _normalize_asr_language(language)
    device = os.getenv("CLIP_PILOT_FASTER_WHISPER_DEVICE", "cpu").strip() or "cpu"
    compute_type = os.getenv("CLIP_PILOT_FASTER_WHISPER_COMPUTE_TYPE", "int8").strip() or "int8"
    beam_size = int(os.getenv("CLIP_PILOT_FASTER_WHISPER_BEAM_SIZE", "5"))
    vad_filter = (
        os.getenv("CLIP_PILOT_FASTER_WHISPER_VAD_FILTER", "true").strip().lower()
        in {"1", "true", "yes", "on"}
    )

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    transcribe_kwargs = {
        "language": normalized_language,
        "beam_size": beam_size,
        "vad_filter": vad_filter,
    }
    if normalized_language is None:
        transcribe_kwargs.pop("language")
    raw_segments, _ = model.transcribe(str(video_path), **transcribe_kwargs)
    segments = [
        TranscriptSegment(
            start=round(float(segment.start), 2),
            end=round(float(segment.end), 2),
            text=segment.text.strip(),
        )
        for segment in raw_segments
        if segment.text.strip() and float(segment.end) > float(segment.start)
    ]
    full_text = " ".join(segment.text for segment in segments).strip()
    return build_transcript_result(
        video_id=task_id,
        provider="faster-whisper",
        segments=segments,
        full_text=full_text,
        status="completed",
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
