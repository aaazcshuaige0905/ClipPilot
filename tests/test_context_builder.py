from clippilot.agents.planner_agent import _build_qwen_planner_request, build_editing_plan
from clippilot.core.context_builder import build_compressed_planning_context
from clippilot.core.memory_manager import build_initial_planner_memory
from clippilot.rag.schemas import RetrievedChunk, RetrievedContext, RetrievalQuery
from clippilot.schemas.editing_plan import HighlightCandidate, HighlightCandidatesResult
from clippilot.schemas.project_state import (
    FineGrainedUnit,
    HighlightScoreBreakdown,
    LLMHighlightCandidate,
    TimelineSegment,
    TrimPolicy,
    VideoTimeline,
)
from clippilot.schemas.user_request import UserRequest
from clippilot.schemas.video_info import VideoInfo
from clippilot.schemas.transcript import TranscriptResult, TranscriptSegment


def _build_user_request() -> UserRequest:
    """Create a reusable user request fixture for context-builder tests."""

    return UserRequest(
        target_platform="bilibili",
        target_duration=30,
        edit_style="powerful",
        language="zh",
        need_burn_subtitle=True,
    )


def _build_video_info() -> VideoInfo:
    """Create reusable source video metadata for planner tests."""

    return VideoInfo(
        duration_seconds=320.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
        file_size_mb=128.0,
        source_path="outputs/tasks/task123/input/source.mp4",
    )


def _build_transcript() -> TranscriptResult:
    """Create a transcript fixture for planner fallback and note generation."""

    return TranscriptResult(
        video_id="task123",
        provider="mock",
        full_text="Opening claim. Supporting proof. Final takeaway.",
        segments=[
            TranscriptSegment(start=0.0, end=4.0, text="Opening claim."),
            TranscriptSegment(start=4.0, end=8.0, text="Supporting proof."),
            TranscriptSegment(start=8.0, end=12.0, text="Final takeaway."),
        ],
    )


def _build_fine_grained_units() -> list[FineGrainedUnit]:
    """Create fine-grained units that can be referenced by compressed candidates."""

    return [
        FineGrainedUnit(
            unit_id="unit_001",
            start=0.0,
            end=3.0,
            duration=3.0,
            text="Opening claim with urgency.",
            has_pause_after=True,
            keywords=["opening", "claim"],
        ),
        FineGrainedUnit(
            unit_id="unit_002",
            start=3.3,
            end=6.0,
            duration=2.7,
            text="Supporting proof with numbers.",
            has_pause_before=True,
            keywords=["proof", "numbers"],
        ),
        FineGrainedUnit(
            unit_id="unit_003",
            start=8.0,
            end=11.5,
            duration=3.5,
            text="Final takeaway and CTA.",
            has_pause_before=True,
            keywords=["takeaway", "cta"],
        ),
    ]


def _build_llm_candidates() -> list[LLMHighlightCandidate]:
    """Create transcript-backed highlight candidates for compression tests."""

    return [
        LLMHighlightCandidate(
            candidate_id="cand_hook",
            source_segment_ids=["seg_01"],
            transcript_unit_ids=["unit_001", "unit_002"],
            source_start=0.0,
            source_end=6.0,
            duration=6.0,
            title="Fast opener",
            summary="Start with the strongest payoff.",
            transcript_excerpt="Opening claim with urgency. Supporting proof with numbers.",
            highlight_type="insight",
            semantic_role="opening_hook",
            reason="High hook potential and clear structure.",
            scores=HighlightScoreBreakdown(
                overall=0.91,
                hook=0.94,
                emotion=0.65,
                clarity=0.88,
                platform_fit=0.9,
                editability=0.86,
            ),
            trim_policy=TrimPolicy(
                min_duration=3.0,
                ideal_duration=5.5,
                max_duration=6.0,
                preferred_start=0.0,
                preferred_end=5.5,
            ),
            risk_flags=["needs_precise_cut"],
        ),
        LLMHighlightCandidate(
            candidate_id="cand_ending",
            source_segment_ids=["seg_03"],
            transcript_unit_ids=["unit_003"],
            source_start=8.0,
            source_end=11.5,
            duration=3.5,
            title="CTA close",
            summary="Close with a direct takeaway.",
            transcript_excerpt="Final takeaway and CTA.",
            highlight_type="takeaway",
            semantic_role="ending",
            reason="Strong closer aligned with platform pacing.",
            scores=HighlightScoreBreakdown(
                overall=0.83,
                hook=0.35,
                emotion=0.5,
                clarity=0.86,
                platform_fit=0.84,
                editability=0.82,
            ),
            trim_policy=TrimPolicy(
                min_duration=2.5,
                ideal_duration=3.5,
                max_duration=3.5,
                preferred_start=8.0,
                preferred_end=11.5,
            ),
        ),
    ]


def _build_timeline() -> VideoTimeline:
    """Create a semantic timeline for compressed-context selection tests."""

    return VideoTimeline(
        provider="qwen",
        model="stub",
        overview="A three-part narrative with a hook, proof, and CTA.",
        segments=[
            TimelineSegment(
                segment_id="seg_01",
                start=0.0,
                end=6.0,
                duration=6.0,
                title="Opening setup",
                summary="Strong claim and immediate payoff.",
                transcript_excerpt="Opening claim with urgency.",
                source_unit_ids=["unit_001", "unit_002"],
                importance=0.95,
                hook_score=0.92,
            ),
            TimelineSegment(
                segment_id="seg_02",
                start=6.0,
                end=8.0,
                duration=2.0,
                title="Bridge",
                summary="A short bridge into the conclusion.",
                transcript_excerpt="Brief bridge.",
                source_unit_ids=[],
                importance=0.4,
                hook_score=0.2,
            ),
            TimelineSegment(
                segment_id="seg_03",
                start=8.0,
                end=11.5,
                duration=3.5,
                title="Ending CTA",
                summary="Direct actionable close.",
                transcript_excerpt="Final takeaway and CTA.",
                source_unit_ids=["unit_003"],
                importance=0.82,
                hook_score=0.45,
            ),
        ],
    )


def _build_retrieved_context() -> RetrievedContext:
    """Create retrieved rules for planner-context compression tests."""

    return RetrievedContext(
        query=RetrievalQuery(text="platform bilibili | duration 30 seconds"),
        chunks=[
            RetrievedChunk(
                chunk_id="rule_hook",
                text="Lead with a payoff in the first three seconds.",
                title="Hook Rule",
                source_file="platform_rules.md",
                knowledge_type="rule",
                strategy_type="hook_rule",
                platform="bilibili",
                language="zh",
                style="powerful",
                duration_band="15_30",
                stage="planning",
                priority="must",
                final_score=0.9,
                retrieval_sources=["dense"],
            ),
            RetrievedChunk(
                chunk_id="rule_pacing",
                text="Keep middle support sections compact and avoid repetition.",
                title="Pacing Rule",
                source_file="editing_templates.md",
                knowledge_type="template",
                strategy_type="pacing_rule",
                platform="bilibili",
                language="zh",
                style="powerful",
                duration_band="15_30",
                stage="planning",
                priority="should",
                final_score=0.7,
                retrieval_sources=["bm25"],
            ),
        ],
    )


def _build_candidates_result() -> HighlightCandidatesResult:
    """Create rule-based candidates so build_editing_plan can still receive this input."""

    return HighlightCandidatesResult(
        video_id="task123",
        candidates=[
            HighlightCandidate(
                start=0.0,
                end=6.0,
                text="Opening claim with urgency. Supporting proof with numbers.",
                score=0.92,
                reason="Strong opening clip.",
            ),
            HighlightCandidate(
                start=8.0,
                end=11.5,
                text="Final takeaway and CTA.",
                score=0.81,
                reason="Concise closer.",
            ),
        ],
    )


def test_build_compressed_planning_context_collects_structured_summaries() -> None:
    """Ensure the context builder compiles request, memory, candidate, rule, and risk summaries."""

    user_request = _build_user_request()
    planner_memory = build_initial_planner_memory("task123", user_request)
    planner_memory.locked_elements.must_keep_candidate_ids.append("cand_hook")
    planner_memory.open_issues.append("Need a sharper CTA.")
    planner_memory.retrieved_chunk_ids = ["rule_pacing", "rule_hook"]

    compressed_context = build_compressed_planning_context(
        user_request=user_request,
        ranked_candidates=_build_llm_candidates(),
        fine_grained_units=_build_fine_grained_units(),
        timeline=_build_timeline(),
        retrieved_context=_build_retrieved_context(),
        planner_memory=planner_memory,
    )

    assert compressed_context.request_summary.target_platform == "bilibili"
    assert compressed_context.memory_summary.locked_candidate_ids == ["cand_hook"]
    assert compressed_context.top_candidates[0].candidate_id == "cand_hook"
    assert compressed_context.timeline_segments[0].relation == "candidate_linked"
    assert [rule.chunk_id for rule in compressed_context.retrieved_rules] == ["rule_pacing", "rule_hook"]
    assert compressed_context.risks[0].risk_type == "open_issue"
    assert compressed_context.stats["candidate_preview_count"] == 2


def test_build_qwen_planner_request_uses_compressed_context_payload() -> None:
    """Ensure the planner request consumes the compressed context packet instead of raw state."""

    compressed_context = build_compressed_planning_context(
        user_request=_build_user_request(),
        ranked_candidates=_build_llm_candidates(),
        fine_grained_units=_build_fine_grained_units(),
        timeline=_build_timeline(),
        retrieved_context=_build_retrieved_context(),
        planner_memory=build_initial_planner_memory("task123", _build_user_request()),
    )

    planner_request = _build_qwen_planner_request(
        task_id="task123",
        video_info=_build_video_info(),
        compressed_context=compressed_context,
    )
    payload = planner_request["messages"][1]["content"]

    assert payload["request_summary"]["target_duration"] == 30
    assert "memory_summary" in payload
    assert "timeline_segments" in payload
    assert "top_candidates" in payload
    assert "retrieved_rules" in payload
    assert "context_stats" in payload
    assert "planner_memory" not in payload


def test_build_editing_plan_reports_compressed_context_usage() -> None:
    """Ensure editing-plan notes describe the compressed planning context path."""

    user_request = _build_user_request()
    planner_memory = build_initial_planner_memory("task123", user_request)
    planner_memory.open_issues.append("Need a sharper CTA.")
    plan = build_editing_plan(
        task_id="task123",
        user_request=user_request,
        video_info=_build_video_info(),
        transcript=_build_transcript(),
        candidates=_build_candidates_result(),
        fine_grained_units=_build_fine_grained_units(),
        timeline=_build_timeline(),
        llm_candidates=_build_llm_candidates(),
        retrieved_context=_build_retrieved_context(),
        planner_memory=planner_memory,
    )

    notes_text = " ".join(plan.editing_notes)
    assert plan.timeline_items
    assert "Compressed context includes" in notes_text
    assert "Compressed planning focus" in notes_text
    assert "Compressed context risks" in notes_text
