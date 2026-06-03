from pydantic import BaseModel, Field


class VideoInfo(BaseModel):
    """Represent extracted metadata for a source video."""

    duration_seconds: float = Field(..., description="Video duration in seconds.")
    width: int = Field(..., description="Video width in pixels.")
    height: int = Field(..., description="Video height in pixels.")
    fps: float = Field(..., description="Frames per second.")
    has_audio: bool = Field(..., description="Whether the video contains an audio track.")
    file_size_mb: float = Field(..., description="Video file size in MB.")
    source_path: str = Field(..., description="Absolute path to the source video.")
