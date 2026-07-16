from pathlib import Path

from clippilot.core.exceptions import ClipPilotValidationError
from clippilot.schemas.editing_plan import HighlightCandidatesResult
from clippilot.schemas.transcript import TranscriptResult


def validate_transcript_not_empty(transcript: TranscriptResult) -> None:
    """Ensure that a transcript is structurally valid, while allowing optional ASR outcomes."""

    if transcript.status not in {"completed", "no_audio", "no_speech", "failed"}:
        raise ClipPilotValidationError(f"Transcript status is unsupported: {transcript.status}")

    if transcript.status == "completed":
        if not transcript.segments:
            raise ClipPilotValidationError("Transcript does not contain any segments.")
        if not transcript.full_text.strip():
            raise ClipPilotValidationError("Transcript full_text is empty.")

    for segment in transcript.segments:
        if segment.end <= segment.start:
            raise ClipPilotValidationError("Transcript contains a segment with invalid time ordering.")


def validate_candidates_not_empty(candidates: HighlightCandidatesResult) -> None:
    """Ensure that highlight generation produced at least one candidate clip."""

    if not candidates.candidates:
        raise ClipPilotValidationError("No highlight candidates were generated.")

    for candidate in candidates.candidates:
        if candidate.end <= candidate.start:
            raise ClipPilotValidationError("A highlight candidate has invalid time ordering.")
        if not 0.0 <= candidate.score <= 1.0:
            raise ClipPilotValidationError("A highlight candidate score is outside the valid range.")


def validate_output_file_exists(path: Path) -> None:
    """Ensure that an expected artifact was actually written to disk."""

    if not path.exists():
        raise ClipPilotValidationError(f"Expected output file does not exist: {path}")
