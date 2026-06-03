import pytest

from clippilot.core.exceptions import ClipPilotValidationError
from clippilot.harness.validators import validate_transcript_not_empty
from clippilot.schemas.transcript import TranscriptResult


def test_validate_transcript_not_empty_raises_for_empty_transcript() -> None:
    """Ensure transcript validation rejects empty transcripts."""

    with pytest.raises(ClipPilotValidationError):
        validate_transcript_not_empty(
            TranscriptResult(video_id="task123", segments=[], full_text="", provider="mock")
        )
