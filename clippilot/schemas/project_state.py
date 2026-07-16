from typing import Any

from pydantic import BaseModel, Field, model_validator

from clippilot.rag.schemas import RetrievedContext
from clippilot.schemas.transcript import TranscriptResult
from clippilot.schemas.user_request import UserRequest
from clippilot.schemas.video_frames import SampledFramesResult
from clippilot.schemas.video_info import VideoInfo


class ProjectPaths(BaseModel):
    """Represent the main artifact paths tracked in the global workflow state."""

    raw_video: str
    audio: str | None = None
    transcript: str | None = None
    sampled_frames: str | None = None
    content_analysis: str | None = None
    timeline: str | None = None
    llm_candidates: str | None = None
    retrieved_context: str | None = None
    final_video: str | None = None


class FineGrainedUnit(BaseModel):
    """Represent one precise editable unit derived from transcript timing boundaries."""

    unit_id: str
    start: float = Field(..., ge=0.0)
    end: float = Field(..., gt=0.0)
    duration: float = Field(..., gt=0.0)
    text: str = Field(..., min_length=1)
    previous_gap_seconds: float = Field(default=0.0, ge=0.0)
    next_gap_seconds: float = Field(default=0.0, ge=0.0)
    has_pause_before: bool = False
    has_pause_after: bool = False
    scene_cut_before: bool = False
    scene_cut_after: bool = False
    energy_score: float | None = Field(default=None, ge=0.0, le=1.0)
    emphasis_score: float | None = Field(default=None, ge=0.0, le=1.0)
    keywords: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_duration(self) -> "FineGrainedUnit":
        """Ensure the editable unit keeps a consistent duration value."""

        computed_duration = round(self.end - self.start, 2)
        if computed_duration <= 0:
            raise ValueError("FineGrainedUnit duration must be positive.")
        self.duration = computed_duration
        return self


class TimelineSegment(BaseModel):
    """Represent one coarse-grained semantic segment on the source-video timeline."""

    segment_id: str
    start: float = Field(..., ge=0.0)
    end: float = Field(..., gt=0.0)
    duration: float = Field(..., gt=0.0)
    title: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    transcript_excerpt: str = Field(..., min_length=1)
    source_unit_ids: list[str] = Field(default_factory=list)
    importance: float = Field(..., ge=0.0, le=1.0)
    hook_score: float = Field(default=0.0, ge=0.0, le=1.0)
    highlight_score: float = Field(default=0.0, ge=0.0, le=1.0)
    emotion: str = Field(default="neutral")
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_segment_duration(self) -> "TimelineSegment":
        """Ensure the semantic segment keeps a consistent duration value."""

        computed_duration = round(self.end - self.start, 2)
        if computed_duration <= 0:
            raise ValueError("TimelineSegment duration must be positive.")
        self.duration = computed_duration
        return self


class HighlightScoreBreakdown(BaseModel):
    """Represent the multi-dimensional score breakdown for one highlight candidate."""

    overall: float = Field(..., ge=0.0, le=1.0)
    hook: float = Field(default=0.0, ge=0.0, le=1.0)
    emotion: float = Field(default=0.0, ge=0.0, le=1.0)
    clarity: float = Field(default=0.0, ge=0.0, le=1.0)
    platform_fit: float = Field(default=0.0, ge=0.0, le=1.0)
    editability: float = Field(default=0.0, ge=0.0, le=1.0)


class TrimPolicy(BaseModel):
    """Represent how a candidate can be safely trimmed during planning."""

    trimmable: bool = True
    min_duration: float = Field(..., gt=0.0)
    ideal_duration: float = Field(..., gt=0.0)
    max_duration: float = Field(..., gt=0.0)
    preferred_start: float = Field(..., ge=0.0)
    preferred_end: float = Field(..., gt=0.0)
    safe_cut_points: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_trim_ranges(self) -> "TrimPolicy":
        """Keep the trim policy internally consistent for the planner."""

        if not (self.min_duration <= self.ideal_duration <= self.max_duration):
            raise ValueError("TrimPolicy must satisfy min_duration <= ideal_duration <= max_duration.")
        if self.preferred_end <= self.preferred_start:
            raise ValueError("TrimPolicy preferred_end must be greater than preferred_start.")
        preferred_duration = round(self.preferred_end - self.preferred_start, 2)
        if preferred_duration < self.min_duration:
            raise ValueError("TrimPolicy preferred duration must be at least min_duration.")
        return self


class LLMHighlightCandidate(BaseModel):
    """Represent one fine-grained highlight candidate returned by video understanding."""

    candidate_id: str
    source_segment_ids: list[str] = Field(default_factory=list)
    transcript_unit_ids: list[str] = Field(default_factory=list)
    source_start: float = Field(..., ge=0.0)
    source_end: float = Field(..., gt=0.0)
    duration: float = Field(..., gt=0.0)
    title: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    transcript_excerpt: str = Field(..., min_length=1)
    highlight_type: str = Field(..., min_length=1)
    semantic_role: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)
    scores: HighlightScoreBreakdown
    must_keep: bool = False
    trim_policy: TrimPolicy
    dependencies: list[str] = Field(default_factory=list)
    redundancy_group: str | None = None
    transition_hint: str = Field(default="straight_cut")
    subtitle_priority: str = Field(default="medium")
    risk_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_candidate_duration(self) -> "LLMHighlightCandidate":
        """Ensure the candidate range and nested trim policy are aligned."""

        computed_duration = round(self.source_end - self.source_start, 2)
        if computed_duration <= 0:
            raise ValueError("LLMHighlightCandidate duration must be positive.")
        self.duration = computed_duration
        if self.trim_policy.max_duration > self.duration + 1e-6:
            raise ValueError("TrimPolicy max_duration cannot exceed the source candidate duration.")
        return self


class VideoTimeline(BaseModel):
    """Represent the coarse-grained timeline extracted by the video understanding agent."""

    provider: str
    model: str
    overview: str = Field(..., min_length=1)
    segments: list[TimelineSegment] = Field(default_factory=list)
    generation_mode: str = Field(default="stub")
    warnings: list[str] = Field(default_factory=list)


class ContentAnalysis(BaseModel):
    """Represent high-level semantic understanding of the source video."""

    summary: str = Field(..., min_length=1)
    narrative_flow: list[str] = Field(default_factory=list)
    main_topics: list[str] = Field(default_factory=list)
    hook_candidates: list[str] = Field(default_factory=list)
    tone: str = Field(default="unknown")
    recommended_structure: list[str] = Field(default_factory=list)
    pacing: str = Field(default="medium")


class ProjectState(BaseModel):
    """Represent the task-wide mutable state shared across agents."""

    project_id: str
    user_request: UserRequest
    paths: ProjectPaths
    video_metadata: VideoInfo | None = None
    transcript: TranscriptResult | None = None
    fine_grained_units: list[FineGrainedUnit] | None = None
    sampled_frames: SampledFramesResult | None = None
    content_analysis: ContentAnalysis | None = None
    retrieved_context: RetrievedContext | None = None
    timeline: VideoTimeline | None = None
    highlight_candidates_llm: list[LLMHighlightCandidate] | None = None
    validation_result: dict[str, Any] | None = None
    human_review: dict[str, Any] | None = None
    render_output: dict[str, Any] | None = None
    qc_report: dict[str, Any] | None = None
    copywriting: dict[str, Any] | None = None
    status: str = "CREATED"
    current_step: str = "created"
    errors: list[str] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)
