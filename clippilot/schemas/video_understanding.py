from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from clippilot.schemas.project_state import HighlightScoreBreakdown, TimelineSegment


class VideoWindow(BaseModel):
    """Represent one source-aligned proxy-video window used for local understanding."""

    window_id: str = Field(..., min_length=1)
    order: int = Field(..., ge=1)
    source_start: float = Field(..., ge=0.0)
    source_end: float = Field(..., gt=0.0)
    duration: float = Field(..., gt=0.0)
    proxy_video_path: str = Field(..., min_length=1)
    source_unit_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_window(self) -> "VideoWindow":
        if self.source_end <= self.source_start:
            raise ValueError("VideoWindow source_end must be greater than source_start.")
        self.duration = round(self.source_end - self.source_start, 2)
        return self


class WindowVisualEvent(BaseModel):
    """Represent one visually grounded event found inside a video window."""

    event_id: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    relative_start: float | None = Field(default=None, ge=0.0)
    relative_end: float | None = Field(default=None, ge=0.0)
    source_unit_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class CoarseHighlightCandidate(BaseModel):
    """Represent a high-recall candidate identified during one window pass."""

    candidate_id: str = Field(..., min_length=1)
    window_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    transcript_excerpt: str = Field(..., min_length=1)
    candidate_type: str = Field(default="semantic_highlight", min_length=1)
    semantic_role: str = Field(default="supporting_point", min_length=1)
    source_unit_ids: list[str] = Field(default_factory=list)
    evidence_event_ids: list[str] = Field(default_factory=list)
    source_start: float = Field(..., ge=0.0)
    source_end: float = Field(..., gt=0.0)
    scores: HighlightScoreBreakdown
    dependencies: list[str] = Field(default_factory=list)
    redundancy_group: str | None = None
    generation_source: str = Field(default="llm_window")
    risk_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_range(self) -> "CoarseHighlightCandidate":
        if self.source_end <= self.source_start:
            raise ValueError("CoarseHighlightCandidate source_end must be greater than source_start.")
        return self


class WindowUnderstandingResult(BaseModel):
    """Represent the normalized result of one independent window-understanding call."""

    window_id: str = Field(..., min_length=1)
    order: int = Field(..., ge=1)
    source_start: float = Field(..., ge=0.0)
    source_end: float = Field(..., gt=0.0)
    summary: str = Field(..., min_length=1)
    topics: list[str] = Field(default_factory=list)
    visual_events: list[WindowVisualEvent] = Field(default_factory=list)
    timeline_segments: list[TimelineSegment] = Field(default_factory=list)
    coarse_candidates: list[CoarseHighlightCandidate] = Field(default_factory=list)
    continues_from_previous: bool = False
    continues_to_next: bool = False
    generation_mode: str = Field(default="llm_video_window_v1", min_length=1)
    warnings: list[str] = Field(default_factory=list)


class CandidateCluster(BaseModel):
    """Represent duplicate or overlapping coarse candidates merged globally."""

    cluster_id: str = Field(..., min_length=1)
    canonical_candidate_id: str = Field(..., min_length=1)
    source_candidate_ids: list[str] = Field(default_factory=list)


class GlobalUnderstandingResult(BaseModel):
    """Represent the text-only aggregation of every window result."""

    summary: str = Field(..., min_length=1)
    narrative_flow: list[str] = Field(default_factory=list)
    main_topics: list[str] = Field(default_factory=list)
    chapters: list[TimelineSegment] = Field(default_factory=list)
    candidate_clusters: list[CandidateCluster] = Field(default_factory=list)
    refinement_candidate_ids: list[str] = Field(default_factory=list)
    reserve_candidate_ids: list[str] = Field(default_factory=list)
    coverage_warnings: list[str] = Field(default_factory=list)
    generation_mode: str = Field(default="llm_window_aggregation_v1")


class CandidateRefinementGroup(BaseModel):
    """Represent one padded source range used to refine nearby candidates together."""

    group_id: str = Field(..., min_length=1)
    source_start: float = Field(..., ge=0.0)
    source_end: float = Field(..., gt=0.0)
    candidate_ids: list[str] = Field(default_factory=list)
    proxy_video_path: str | None = None


class RefinedCandidateOption(BaseModel):
    """Represent one validated short, ideal, or extended trim option."""

    option_id: str = Field(..., min_length=1)
    option_type: Literal["short", "ideal", "extended"]
    start_boundary_id: str = Field(..., min_length=1)
    end_boundary_id: str = Field(..., min_length=1)
    source_unit_ids: list[str] = Field(default_factory=list)
    source_start: float = Field(..., ge=0.0)
    source_end: float = Field(..., gt=0.0)
    duration: float = Field(..., gt=0.0)

    @model_validator(mode="after")
    def validate_option(self) -> "RefinedCandidateOption":
        if self.source_end <= self.source_start:
            raise ValueError("RefinedCandidateOption source_end must be greater than source_start.")
        self.duration = round(self.source_end - self.source_start, 2)
        return self


class RefinedHighlightCandidate(BaseModel):
    """Represent one coarse candidate after focused visual and boundary validation."""

    candidate_id: str = Field(..., min_length=1)
    options: list[RefinedCandidateOption] = Field(default_factory=list)
    scores: HighlightScoreBreakdown
    semantic_complete: bool = False
    visual_complete: bool = False
    requires_previous_context: bool = False
    requires_next_context: bool = False
    dependencies: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    generation_mode: str = Field(default="llm_candidate_refinement_v1")
