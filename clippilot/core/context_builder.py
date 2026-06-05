from __future__ import annotations

from clippilot.rag.schemas import RetrievedContext
from clippilot.schemas.planner_memory import PlannerMemory
from clippilot.schemas.planning_context import (
    CandidatePreview,
    CandidateUnitPreview,
    CompressedPlanningContext,
    ContextRisk,
    MemorySummary,
    RequestSummary,
    RetrievedRulePreview,
    TimelineSegmentPreview,
)
from clippilot.schemas.project_state import FineGrainedUnit, LLMHighlightCandidate, VideoTimeline
from clippilot.schemas.user_request import UserRequest


def _compact_text(text: str, limit: int = 140) -> str:
    """Collapse whitespace and shorten prompt-facing content."""

    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 3]}..."


def _request_instruction(user_request: UserRequest) -> str:
    """Build a single compact request instruction for planner prompts."""

    subtitle_mode = "burned subtitles" if user_request.need_burn_subtitle else "soft subtitles"
    return (
        f"Create a {user_request.target_duration}-second video for {user_request.target_platform} "
        f"using a {user_request.edit_style} style in {user_request.language} with {subtitle_mode}."
    )


def _build_request_summary(user_request: UserRequest) -> RequestSummary:
    """Compress the user request into one planner-friendly summary object."""

    return RequestSummary(
        target_platform=user_request.target_platform,
        target_duration=user_request.target_duration,
        edit_style=user_request.edit_style,
        language=user_request.language,
        need_burn_subtitle=user_request.need_burn_subtitle,
        instruction=_request_instruction(user_request),
    )


def _build_memory_summary(planner_memory: PlannerMemory | None) -> MemorySummary:
    """Extract the planner-memory subset that matters for downstream planning."""

    if planner_memory is None:
        return MemorySummary()

    latest_instruction = planner_memory.instruction_history[-1].raw_text if planner_memory.instruction_history else None
    latest_revision_summary = planner_memory.revision_history[-1].summary if planner_memory.revision_history else None
    return MemorySummary(
        active_constraints=dict(planner_memory.active_constraints),
        planning_focus=[_compact_text(item, limit=100) for item in planner_memory.planning_focus[:5]],
        open_issues=[_compact_text(item, limit=120) for item in planner_memory.open_issues[:4]],
        locked_candidate_ids=planner_memory.locked_elements.must_keep_candidate_ids[:6],
        forbidden_candidate_ids=planner_memory.locked_elements.forbidden_candidate_ids[:6],
        locked_quotes=[_compact_text(item, limit=100) for item in planner_memory.locked_elements.must_keep_quotes[:3]],
        latest_instruction=_compact_text(latest_instruction, limit=150) if latest_instruction else None,
        latest_revision_summary=_compact_text(latest_revision_summary, limit=150) if latest_revision_summary else None,
    )


def _build_candidate_previews(
    ranked_candidates: list[LLMHighlightCandidate],
    fine_grained_units: list[FineGrainedUnit],
    limit: int = 6,
) -> list[CandidatePreview]:
    """Select and compress the strongest LLM candidates for prompt use."""

    units_by_id = {unit.unit_id: unit for unit in fine_grained_units}
    previews: list[CandidatePreview] = []

    for candidate in ranked_candidates[:limit]:
        related_units: list[CandidateUnitPreview] = []
        for unit_id in candidate.transcript_unit_ids[:4]:
            unit = units_by_id.get(unit_id)
            if unit is None:
                continue
            related_units.append(
                CandidateUnitPreview(
                    unit_id=unit.unit_id,
                    start=unit.start,
                    end=unit.end,
                    text=_compact_text(unit.text, limit=90),
                    has_pause_before=unit.has_pause_before,
                    has_pause_after=unit.has_pause_after,
                )
            )

        previews.append(
            CandidatePreview(
                candidate_id=candidate.candidate_id,
                title=_compact_text(candidate.title, limit=80),
                summary=_compact_text(candidate.summary, limit=120),
                transcript_excerpt=_compact_text(candidate.transcript_excerpt, limit=120),
                source_start=candidate.source_start,
                source_end=candidate.source_end,
                semantic_role=candidate.semantic_role,
                highlight_type=candidate.highlight_type,
                overall_score=round(candidate.scores.overall, 4),
                reason=_compact_text(candidate.reason, limit=120),
                source_segment_ids=candidate.source_segment_ids[:4],
                related_units=related_units,
                trim_guidance={
                    "preferred_start": candidate.trim_policy.preferred_start,
                    "preferred_end": candidate.trim_policy.preferred_end,
                    "ideal_duration": candidate.trim_policy.ideal_duration,
                    "max_duration": candidate.trim_policy.max_duration,
                },
            )
        )

    return previews


def _build_timeline_previews(
    timeline: VideoTimeline | None,
    candidate_previews: list[CandidatePreview],
    limit: int = 6,
) -> list[TimelineSegmentPreview]:
    """Select timeline segments that best support the compressed candidate set."""

    if timeline is None or not timeline.segments:
        return []

    linked_segment_ids = {
        segment_id
        for candidate in candidate_previews
        for segment_id in candidate.source_segment_ids
        if segment_id
    }
    ordered_segments = sorted(
        timeline.segments,
        key=lambda segment: (
            0 if segment.segment_id in linked_segment_ids else 1,
            -segment.importance,
            -segment.hook_score,
            segment.start,
        ),
    )

    previews: list[TimelineSegmentPreview] = []
    for segment in ordered_segments[:limit]:
        relation = "candidate_linked" if segment.segment_id in linked_segment_ids else "high_importance"
        previews.append(
            TimelineSegmentPreview(
                segment_id=segment.segment_id,
                start=segment.start,
                end=segment.end,
                title=_compact_text(segment.title, limit=80),
                summary=_compact_text(segment.summary, limit=120),
                importance=segment.importance,
                hook_score=segment.hook_score,
                relation=relation,
            )
        )
    return previews


def _rule_priority_rank(priority: str) -> int:
    """Convert textual rule priority into a stable sorting rank."""

    ranks = {"must": 0, "should": 1, "could": 2}
    return ranks.get(priority.strip().lower(), 3)


def _build_retrieved_rule_previews(
    retrieved_context: RetrievedContext | None,
    planner_memory: PlannerMemory | None,
    limit: int = 4,
) -> list[RetrievedRulePreview]:
    """Select the most relevant retrieved rules for prompt injection."""

    if retrieved_context is None or not retrieved_context.chunks:
        return []

    memory_order = {
        chunk_id: index
        for index, chunk_id in enumerate(planner_memory.retrieved_chunk_ids)
    } if planner_memory is not None else {}

    ordered_chunks = sorted(
        retrieved_context.chunks,
        key=lambda chunk: (
            memory_order.get(chunk.chunk_id, 999),
            _rule_priority_rank(chunk.priority),
            -chunk.final_score,
            chunk.title,
        ),
    )

    previews: list[RetrievedRulePreview] = []
    for chunk in ordered_chunks[:limit]:
        previews.append(
            RetrievedRulePreview(
                chunk_id=chunk.chunk_id,
                title=_compact_text(chunk.title, limit=80),
                strategy_type=chunk.strategy_type,
                priority=chunk.priority,
                text=_compact_text(chunk.text, limit=160),
            )
        )
    return previews


def _build_risks(
    planner_memory: PlannerMemory | None,
    candidate_previews: list[CandidatePreview],
    retrieved_rules: list[RetrievedRulePreview],
) -> list[ContextRisk]:
    """Build a small risk digest so planners know what still needs care."""

    risks: list[ContextRisk] = []
    if planner_memory is not None:
        for issue in planner_memory.open_issues[:4]:
            risks.append(ContextRisk(risk_type="open_issue", message=_compact_text(issue, limit=140)))

    if not candidate_previews:
        risks.append(ContextRisk(risk_type="candidate_gap", message="No transcript-backed candidate previews are available."))
    if not retrieved_rules:
        risks.append(ContextRisk(risk_type="rule_gap", message="No retrieved strategy rules are available for this planning pass."))

    return risks[:6]


def build_compressed_planning_context(
    *,
    user_request: UserRequest,
    ranked_candidates: list[LLMHighlightCandidate],
    fine_grained_units: list[FineGrainedUnit] | None = None,
    timeline: VideoTimeline | None = None,
    retrieved_context: RetrievedContext | None = None,
    planner_memory: PlannerMemory | None = None,
) -> CompressedPlanningContext:
    """Compile raw planning state into a compact planner-facing context packet."""

    candidate_previews = _build_candidate_previews(
        ranked_candidates=ranked_candidates,
        fine_grained_units=fine_grained_units or [],
    )
    timeline_previews = _build_timeline_previews(
        timeline=timeline,
        candidate_previews=candidate_previews,
    )
    retrieved_rules = _build_retrieved_rule_previews(
        retrieved_context=retrieved_context,
        planner_memory=planner_memory,
    )
    risks = _build_risks(
        planner_memory=planner_memory,
        candidate_previews=candidate_previews,
        retrieved_rules=retrieved_rules,
    )

    return CompressedPlanningContext(
        request_summary=_build_request_summary(user_request),
        memory_summary=_build_memory_summary(planner_memory),
        top_candidates=candidate_previews,
        timeline_segments=timeline_previews,
        retrieved_rules=retrieved_rules,
        risks=risks,
        stats={
            "candidate_preview_count": len(candidate_previews),
            "timeline_segment_count": len(timeline_previews),
            "retrieved_rule_count": len(retrieved_rules),
            "risk_count": len(risks),
            "input_candidate_count": len(ranked_candidates),
        },
    )
