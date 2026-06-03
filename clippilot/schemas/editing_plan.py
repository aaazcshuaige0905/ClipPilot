from pydantic import BaseModel, Field, model_validator


class HighlightCandidate(BaseModel):
    """Represent a candidate highlight clip with heuristic score and explanation."""

    start: float = Field(..., description="Candidate start time in seconds.")
    end: float = Field(..., description="Candidate end time in seconds.")
    text: str = Field(..., min_length=1, description="Merged transcript text for the candidate.")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized candidate score.")
    reason: str = Field(..., min_length=1, description="Short explanation of the candidate score.")


class HighlightCandidatesResult(BaseModel):
    """Represent all ranked highlight candidates for a single task."""

    video_id: str
    candidates: list[HighlightCandidate]


class EditingClip(BaseModel):
    """Represent one executable clip item in the editing timeline."""

    clip_id: str
    source_start: float = Field(..., ge=0.0, description="Clip start time in the source video.")
    source_end: float = Field(..., gt=0.0, description="Clip end time in the source video.")
    duration: float = Field(..., gt=0.0, description="Clip duration in seconds.")
    purpose: str = Field(..., min_length=1, description="Purpose of the clip in the final short video.")
    text: str = Field(..., min_length=1, description="Primary transcript text of the clip.")
    subtitle: str = Field(..., min_length=1, description="Subtitle text to display for the clip.")
    score: float = Field(..., ge=0.0, le=1.0, description="Selection score used by the planner.")
    reason: str = Field(..., min_length=1, description="Reason why the planner selected this clip.")

    @model_validator(mode="after")
    def validate_timing(self) -> "EditingClip":
        """Ensure the clip keeps a valid start/end range and aligned duration."""

        if self.source_end <= self.source_start:
            raise ValueError("source_end must be greater than source_start.")
        computed_duration = round(self.source_end - self.source_start, 2)
        if computed_duration < 3.0:
            raise ValueError("Each clip must be at least 3 seconds long.")
        if computed_duration > 20.0:
            raise ValueError("Each clip must not exceed 20 seconds.")
        self.duration = computed_duration
        return self


class EditingPlan(BaseModel):
    """Represent an executable editing timeline generated from transcript and highlight candidates."""

    task_id: str
    target_duration: int = Field(..., gt=0, description="Requested final video duration in seconds.")
    total_duration: float = Field(..., ge=0.0, description="Total selected clip duration in seconds.")
    clips: list[EditingClip] = Field(default_factory=list)
    editing_notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    strategy: str = Field(default="rule_based_timeline_planner")
    source_candidate_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_total_duration(self) -> "EditingPlan":
        """Ensure the plan stays consistent with its clip list and duration target."""

        if not self.clips:
            raise ValueError("EditingPlan must contain at least one clip.")
        computed_total = round(sum(clip.duration for clip in self.clips), 2)
        self.total_duration = computed_total
        if computed_total > self.target_duration + 5:
            raise ValueError("total_duration must not exceed target_duration + 5 seconds.")
        return self
