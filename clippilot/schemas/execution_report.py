from pydantic import BaseModel, Field


class ExecutionClipResult(BaseModel):
    """Represent the execution result of one planned clip cut step."""

    clip_id: str
    source_start: float = Field(default=0.0, ge=0.0)
    source_end: float = Field(default=0.0, ge=0.0)
    clip_path: str
    duration: float = Field(default=0.0, ge=0.0)
    success: bool
    error: str | None = None


class ExecutionReport(BaseModel):
    """Represent the real execution output of the video assembly stage."""

    task_id: str
    status: str
    clip_results: list[ExecutionClipResult] = Field(default_factory=list)
    final_video_path: str | None = None
    subtitle_path: str | None = None
    burned_video_path: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
