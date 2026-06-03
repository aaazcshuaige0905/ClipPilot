from pydantic import BaseModel, Field


class ReviewCheck(BaseModel):
    """Represent one structured quality check inside the review stage."""

    name: str = Field(..., min_length=1)
    passed: bool
    severity: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class ReviewReport(BaseModel):
    """Represent the quality review output for exported task artifacts."""

    task_id: str
    passed: bool = False
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    checks: list[ReviewCheck] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
