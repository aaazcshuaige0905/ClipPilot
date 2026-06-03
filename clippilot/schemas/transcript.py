from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """Represent a timestamped ASR transcript segment."""

    start: float = Field(..., description="Segment start time in seconds.")
    end: float = Field(..., description="Segment end time in seconds.")
    text: str = Field(..., description="Transcribed text for the segment.")


class TranscriptResult(BaseModel):
    """Represent a transcript payload generated from a source video."""

    video_id: str
    segments: list[TranscriptSegment]
    full_text: str
    provider: str
