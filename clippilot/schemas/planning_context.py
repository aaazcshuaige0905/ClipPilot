from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from clippilot.rag.schemas import utc_timestamp


class RequestSummary(BaseModel):
    """Represent the compressed user-request summary for planner prompts."""

    target_platform: str = Field(..., min_length=1)
    target_duration: int = Field(..., gt=0)
    edit_style: str = Field(..., min_length=1)
    language: str = Field(..., min_length=1)
    need_burn_subtitle: bool
    instruction: str = Field(..., min_length=1)


class MemorySummary(BaseModel):
    """Represent the planner-memory subset that downstream agents should consume."""

    active_constraints: dict[str, Any] = Field(default_factory=dict)
    planning_focus: list[str] = Field(default_factory=list)
    open_issues: list[str] = Field(default_factory=list)
    locked_candidate_ids: list[str] = Field(default_factory=list)
    forbidden_candidate_ids: list[str] = Field(default_factory=list)
    locked_quotes: list[str] = Field(default_factory=list)
    latest_instruction: str | None = None
    latest_revision_summary: str | None = None


class CandidateUnitPreview(BaseModel):
    """Represent one fine-grained transcript unit preview attached to a candidate."""

    unit_id: str = Field(..., min_length=1)
    start: float = Field(..., ge=0.0)
    end: float = Field(..., gt=0.0)
    text: str = Field(..., min_length=1)
    has_pause_before: bool = False
    has_pause_after: bool = False


class CandidatePreview(BaseModel):
    """Represent one compressed highlight candidate preview for planner prompting."""

    candidate_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    transcript_excerpt: str = Field(..., min_length=1)
    source_start: float = Field(..., ge=0.0)
    source_end: float = Field(..., gt=0.0)
    semantic_role: str = Field(..., min_length=1)
    highlight_type: str = Field(..., min_length=1)
    overall_score: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., min_length=1)
    source_segment_ids: list[str] = Field(default_factory=list)
    related_units: list[CandidateUnitPreview] = Field(default_factory=list)
    trim_guidance: dict[str, Any] = Field(default_factory=dict)


class TimelineSegmentPreview(BaseModel):
    """Represent one compressed timeline segment preview for planner context."""

    segment_id: str = Field(..., min_length=1)
    start: float = Field(..., ge=0.0)
    end: float = Field(..., gt=0.0)
    title: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    importance: float = Field(..., ge=0.0, le=1.0)
    hook_score: float = Field(..., ge=0.0, le=1.0)
    relation: str = Field(default="context", min_length=1)


class RetrievedRulePreview(BaseModel):
    """Represent one retrieved rule reduced for downstream planner consumption."""

    chunk_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    strategy_type: str = Field(..., min_length=1)
    priority: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class ContextRisk(BaseModel):
    """Represent one compressed planning risk or unresolved issue."""

    risk_type: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class CompressedPlanningContext(BaseModel):
    """Represent the compact planner-facing context built from local task state."""

    request_summary: RequestSummary
    memory_summary: MemorySummary
    top_candidates: list[CandidatePreview] = Field(default_factory=list)
    timeline_segments: list[TimelineSegmentPreview] = Field(default_factory=list)
    retrieved_rules: list[RetrievedRulePreview] = Field(default_factory=list)
    risks: list[ContextRisk] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    generated_at: str = Field(default_factory=utc_timestamp)
