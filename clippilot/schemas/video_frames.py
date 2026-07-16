from pydantic import BaseModel, Field


class SampledFrame(BaseModel):
    """Represent one sampled visual frame saved for video-understanding input."""

    frame_id: str = Field(..., min_length=1)
    timestamp: float = Field(..., ge=0.0)
    image_path: str = Field(..., min_length=1)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    note: str | None = None


class SampledFramesResult(BaseModel):
    """Represent the sampled frame bundle generated from one source video."""

    video_id: str = Field(..., min_length=1)
    strategy: str = Field(..., min_length=1)
    frame_count: int = Field(..., ge=0)
    frames: list[SampledFrame] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
