from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from clippilot.rag.schemas import utc_timestamp


class LockedPlanElements(BaseModel):
    """Represent currently locked or forbidden content within one project."""

    must_keep_quotes: list[str] = Field(default_factory=list)
    must_keep_candidate_ids: list[str] = Field(default_factory=list)
    must_keep_unit_ids: list[str] = Field(default_factory=list)
    forbidden_candidate_ids: list[str] = Field(default_factory=list)
    forbidden_unit_ids: list[str] = Field(default_factory=list)


class PlannerInstructionRecord(BaseModel):
    """Represent one user-side instruction captured in short-term planner memory."""

    turn_id: str = Field(..., min_length=1)
    raw_text: str = Field(..., min_length=1)
    parsed_constraints: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_timestamp)


class PlannerRevisionRecord(BaseModel):
    """Represent one planner-produced version update stored in short-term memory."""

    plan_version: int = Field(..., ge=1)
    summary: str = Field(..., min_length=1)
    changed_item_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_timestamp)


class PlannerMemoryEvent(BaseModel):
    """Represent one stage-level memory write applied to planner memory."""

    event_type: str = Field(..., min_length=1)
    stage: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_timestamp)


class PlannerMemory(BaseModel):
    """Represent task-local short-term memory owned by the planner agent."""

    project_id: str = Field(..., min_length=1)
    current_plan_version: int = Field(default=0, ge=0)
    active_constraints: dict[str, Any] = Field(default_factory=dict)
    planning_focus: list[str] = Field(default_factory=list)
    open_issues: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_rule_titles: list[str] = Field(default_factory=list)
    locked_elements: LockedPlanElements = Field(default_factory=LockedPlanElements)
    instruction_history: list[PlannerInstructionRecord] = Field(default_factory=list)
    revision_history: list[PlannerRevisionRecord] = Field(default_factory=list)
    memory_events: list[PlannerMemoryEvent] = Field(default_factory=list)
    planner_working_summary: str = Field(default="Initial planner memory created for this project.", min_length=1)
    created_at: str = Field(default_factory=utc_timestamp)
    updated_at: str = Field(default_factory=utc_timestamp)

    def touch(self) -> None:
        """Refresh the update timestamp after planner memory changes."""

        self.updated_at = utc_timestamp()
