from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """Represent a timestamped ASR transcript segment."""

    start: float = Field(..., description="Segment start time in seconds.")
    end: float = Field(..., description="Segment end time in seconds.")
    text: str = Field(..., description="Transcribed text for the segment.")


class TranscriptResult(BaseModel):
    """Represent a transcript payload generated from a source video."""

    video_id: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
    full_text: str = ""
    provider: str
    status: str = Field(default="completed", description="completed, no_audio, no_speech, or failed.")
    warnings: list[str] = Field(default_factory=list)

    def has_content(self) -> bool:
        """Return whether the transcript contains usable timed text content."""

        return bool(self.full_text.strip()) and any(segment.text.strip() for segment in self.segments)
