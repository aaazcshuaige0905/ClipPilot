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


class PlanningBeat(BaseModel):
    """Represent one narrative beat in the final short-video structure."""

    beat_id: str
    order: int = Field(..., ge=1)
    role: str = Field(..., min_length=1, description="Narrative role such as hook, core_point, or ending.")
    goal: str = Field(..., min_length=1, description="What this beat should achieve for the audience.")
    target_duration: float = Field(..., gt=0.0, description="Target duration budget for this beat.")
    summary: str = Field(..., min_length=1, description="Readable summary of the beat content.")
    source_candidate_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class TimelineSourceRef(BaseModel):
    """Represent one source-backed snippet used inside a timeline item."""

    ref_id: str = Field(..., min_length=1)
    ref_type: str = Field(..., min_length=1, description="candidate, segment, or unit.")
    start: float = Field(..., ge=0.0)
    end: float = Field(..., gt=0.0)
    text: str = Field(default="", description="Optional transcript text for debugging and review.")

    @model_validator(mode="after")
    def validate_range(self) -> "TimelineSourceRef":
        """Ensure the source reference uses a valid time range."""

        if self.end <= self.start:
            raise ValueError("TimelineSourceRef end must be greater than start.")
        return self


class TimelineItem(BaseModel):
    """Represent one planner-selected assembly item before executor-specific rendering."""

    item_id: str
    beat_id: str = Field(..., min_length=1)
    purpose: str = Field(..., min_length=1)
    assembly_mode: str = Field(default="single_cut", min_length=1)
    source_start: float = Field(..., ge=0.0, description="Earliest source timestamp covered by this item.")
    source_end: float = Field(..., gt=0.0, description="Latest source timestamp covered by this item.")
    duration: float = Field(..., gt=0.0, description="Planned visible duration for this item.")
    candidate_ids: list[str] = Field(default_factory=list)
    source_unit_ids: list[str] = Field(default_factory=list)
    source_refs: list[TimelineSourceRef] = Field(default_factory=list)
    text: str = Field(..., min_length=1)
    subtitle: str | None = Field(default=None)
    score: float = Field(..., ge=0.0, le=1.0)
    transition: str = Field(default="straight_cut", min_length=1)
    reason: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_item(self) -> "TimelineItem":
        """Ensure the item keeps a coherent range and duration."""

        if self.source_end <= self.source_start:
            raise ValueError("TimelineItem source_end must be greater than source_start.")

        if self.source_refs:
            self.source_start = round(min(reference.start for reference in self.source_refs), 2)
            self.source_end = round(max(reference.end for reference in self.source_refs), 2)
            if self.assembly_mode == "montage":
                computed_duration = round(sum(reference.end - reference.start for reference in self.source_refs), 2)
            else:
                computed_duration = round(self.source_end - self.source_start, 2)
        else:
            computed_duration = round(self.source_end - self.source_start, 2)

        if computed_duration <= 0:
            raise ValueError("TimelineItem duration must be positive.")

        self.duration = computed_duration
        return self


class EditingPlan(BaseModel):
    """Represent an executable editing timeline generated from transcript and highlight candidates."""

    task_id: str
    plan_version: int = Field(default=1, ge=1)
    target_duration: int = Field(..., gt=0, description="Requested final video duration in seconds.")
    total_duration: float = Field(..., ge=0.0, description="Total selected clip duration in seconds.")
    beats: list[PlanningBeat] = Field(default_factory=list)
    timeline_items: list[TimelineItem] = Field(default_factory=list)
    editing_notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    strategy: str = Field(default="llm_montage_planner")
    generation_mode: str = Field(default="stub")
    source_candidate_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_total_duration(self) -> "EditingPlan":
        """Ensure the plan stays consistent with its clip list and duration target."""

        if not self.timeline_items:
            raise ValueError("EditingPlan must contain at least one timeline item.")

        computed_total = round(sum(item.duration for item in self.timeline_items), 2)
        self.total_duration = computed_total
        if computed_total > self.target_duration + 5:
            raise ValueError("total_duration must not exceed target_duration + 5 seconds.")
        return self
