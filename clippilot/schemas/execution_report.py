from pydantic import BaseModel, Field


class ExecutionItemResult(BaseModel):
    """Represent the execution result of one planned timeline-item render step."""

    item_id: str
    source_start: float = Field(default=0.0, ge=0.0)
    source_end: float = Field(default=0.0, ge=0.0)
    item_path: str
    duration: float = Field(default=0.0, ge=0.0)
    success: bool
    error: str | None = None


class ExecutionReport(BaseModel):
    """Represent the real execution output of the video assembly stage."""

    task_id: str
    status: str
    item_results: list[ExecutionItemResult] = Field(default_factory=list)
    final_video_path: str | None = None
    subtitle_path: str | None = None
    burned_video_path: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
