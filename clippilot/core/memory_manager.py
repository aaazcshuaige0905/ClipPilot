from __future__ import annotations

import re
from typing import Any

from clippilot.core.states import WorkflowStage
from clippilot.rag.schemas import RetrievedContext
from clippilot.schemas.editing_plan import EditingPlan
from clippilot.schemas.planner_memory import (
    PlannerInstructionRecord,
    PlannerMemory,
    PlannerMemoryEvent,
    PlannerRevisionRecord,
)
from clippilot.schemas.review_report import ReviewReport
from clippilot.schemas.user_request import UserRequest


def _compact_text(text: str, limit: int = 140) -> str:
    """Collapse whitespace and keep memory summaries easy to scan."""

    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 3]}..."


def _merge_unique(existing: list[str], additions: list[str], limit: int = 8) -> list[str]:
    """Merge short bullet-like memory values while preserving order."""

    merged: list[str] = []
    for item in [*existing, *additions]:
        normalized = item.strip()
        if not normalized or normalized in merged:
            continue
        merged.append(normalized)
        if len(merged) >= limit:
            break
    return merged


def _constraints_from_request(user_request: UserRequest) -> dict[str, Any]:
    """Convert the initial request into stable planner constraints."""

    return {
        "target_platform": user_request.target_platform,
        "target_duration": user_request.target_duration,
        "edit_style": user_request.edit_style,
        "language": user_request.language,
        "need_burn_subtitle": user_request.need_burn_subtitle,
    }


def _focus_from_constraints(constraints: dict[str, Any]) -> list[str]:
    """Build lightweight planning focus bullets from parsed constraints."""

    focus: list[str] = []
    target_platform = constraints.get("target_platform")
    if target_platform:
        focus.append(f"Optimize for {target_platform}.")
    target_duration = constraints.get("target_duration")
    if target_duration:
        focus.append(f"Keep the cut near {target_duration} seconds.")
    edit_style = constraints.get("edit_style")
    if edit_style:
        focus.append(f"Follow a {edit_style} editing style.")
    language = constraints.get("language")
    if language:
        focus.append(f"Preserve {language} language clarity.")
    if constraints.get("need_burn_subtitle"):
        focus.append("Plan with burned subtitles enabled.")
    return focus


def _set_working_summary(memory: PlannerMemory, summary_parts: list[str]) -> None:
    """Refresh the rolling planner working summary from short summary fragments."""

    parts = [_compact_text(part) for part in summary_parts if part and part.strip()]
    if parts:
        memory.planner_working_summary = " ".join(parts)


def _append_event(
    memory: PlannerMemory,
    *,
    event_type: str,
    stage: str,
    summary: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append one normalized event into planner memory."""

    memory.memory_events.append(
        PlannerMemoryEvent(
            event_type=event_type,
            stage=stage,
            summary=_compact_text(summary, limit=180),
            metadata=metadata or {},
        )
    )


def _request_instruction_text(user_request: UserRequest) -> str:
    """Serialize the initial request into a human-readable memory instruction."""

    subtitle_mode = "burned subtitles" if user_request.need_burn_subtitle else "soft subtitles"
    return (
        f"Create a {user_request.target_duration}-second short video for {user_request.target_platform} "
        f"using a {user_request.edit_style} editing style in {user_request.language}. "
        f"Prefer {subtitle_mode}."
    )


def _extract_quoted_phrases(text: str) -> list[str]:
    """Extract short quoted phrases from English or CJK quote styles."""

    patterns = [
        r'"([^"\n]{1,120})"',
        r"'([^'\n]{1,120})'",
        r"“([^”\n]{1,120})”",
        r"「([^」\n]{1,120})」",
        r"『([^』\n]{1,120})』",
    ]
    phrases: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            normalized = " ".join(match.split()).strip()
            if normalized and normalized not in phrases:
                phrases.append(normalized)
    return phrases


def _extract_candidate_ids(text: str) -> list[str]:
    """Extract planner candidate identifiers from free-form feedback."""

    candidate_ids = re.findall(r"\b(?:cand|candidate)_[a-zA-Z0-9_]+\b", text)
    return list(dict.fromkeys(candidate_ids))


def _extract_duration_constraint(text: str, current_target_duration: int | None) -> int | None:
    """Parse explicit or relative duration feedback into a target duration override."""

    explicit_patterns = [
        r"(?:改成|调整到|控制在|压缩到|缩短到|延长到)\s*(\d+)\s*秒",
        r"(?:to|under|around|about)\s*(\d+)\s*(?:seconds?|secs?|s)\b",
        r"(\d+)\s*(?:seconds?|secs?|s|秒)\b",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return max(3, int(match.group(1)))

    lowered = text.lower()
    if current_target_duration is not None:
        if any(marker in text for marker in ("更短", "短一点", "压缩", "精简")) or any(
            marker in lowered for marker in ("shorter", "trim it down", "make it shorter")
        ):
            return max(3, current_target_duration - 5)
        if any(marker in text for marker in ("更长", "长一点", "展开")) or any(
            marker in lowered for marker in ("longer", "expand it", "make it longer")
        ):
            return current_target_duration + 5
    return None


def _feedback_focus_hints(feedback: str) -> list[str]:
    """Generate short planning-focus hints from revision feedback."""

    hints: list[str] = []
    lowered = feedback.lower()
    if any(marker in feedback for marker in ("开头", "hook", "前3秒")) or "hook" in lowered:
        hints.append("Re-evaluate the opening hook.")
    if any(marker in feedback for marker in ("结尾", "收尾", "cta")) or "ending" in lowered:
        hints.append("Strengthen the ending payoff or CTA.")
    if any(marker in feedback for marker in ("节奏", "更快", "更紧凑")) or any(
        marker in lowered for marker in ("pacing", "faster", "tighter")
    ):
        hints.append("Tighten pacing during replanning.")
    return hints


def build_initial_planner_memory(task_id: str, user_request: UserRequest) -> PlannerMemory:
    """Create planner memory and immediately seed it with the initial request."""

    constraints = _constraints_from_request(user_request)
    memory = PlannerMemory(
        project_id=task_id,
        active_constraints=constraints,
        planning_focus=_focus_from_constraints(constraints),
        planner_working_summary=(
            "Initial planner memory created from the upload request. "
            "No user revisions have been applied yet."
        ),
    )
    record_user_instruction(
        memory,
        turn_id="initial_request",
        raw_text=_request_instruction_text(user_request),
        parsed_constraints=constraints,
        stage=WorkflowStage.CREATED.value,
    )
    _append_event(
        memory,
        event_type="request_initialized",
        stage=WorkflowStage.CREATED.value,
        summary="Planner memory initialized from the upload request.",
        metadata=constraints,
    )
    return memory


def record_user_instruction(
    memory: PlannerMemory,
    *,
    turn_id: str,
    raw_text: str,
    parsed_constraints: dict[str, Any] | None = None,
    stage: str = WorkflowStage.CREATED.value,
) -> None:
    """Record one user-side instruction into planner memory."""

    normalized_constraints = parsed_constraints or {}
    memory.instruction_history.append(
        PlannerInstructionRecord(
            turn_id=turn_id,
            raw_text=_compact_text(raw_text, limit=220),
            parsed_constraints=normalized_constraints,
        )
    )
    if normalized_constraints:
        memory.active_constraints.update(normalized_constraints)
        memory.planning_focus = _merge_unique(
            memory.planning_focus,
            _focus_from_constraints(normalized_constraints),
        )
    _append_event(
        memory,
        event_type="user_instruction",
        stage=stage,
        summary=f"Recorded planner instruction for turn {turn_id}.",
        metadata={"turn_id": turn_id, "parsed_constraints": normalized_constraints},
    )
    _set_working_summary(
        memory,
        [
            f"Planner is operating on task {memory.project_id}.",
            f"Latest instruction: {_compact_text(raw_text, limit=120)}",
            "No revision feedback has been applied yet."
            if not memory.revision_history
            else f"Latest saved plan version: v{memory.current_plan_version}.",
        ],
    )


def apply_revision_feedback(
    memory: PlannerMemory,
    *,
    feedback_text: str,
    turn_id: str,
    stage: str = "revision_requested",
) -> dict[str, Any]:
    """Parse free-form revision feedback and write the structured result into planner memory."""

    current_target_duration = memory.active_constraints.get("target_duration")
    feedback_constraints: dict[str, Any] = {}
    parsed_feedback: dict[str, Any] = {
        "must_keep_candidate_ids": [],
        "forbidden_candidate_ids": [],
        "must_keep_quotes": [],
        "open_issues": [],
        "focus_hints": [],
    }
    lowered = feedback_text.lower()
    candidate_ids = _extract_candidate_ids(feedback_text)
    quoted_phrases = _extract_quoted_phrases(feedback_text)
    duration_override = _extract_duration_constraint(feedback_text, current_target_duration)

    keep_intent = any(marker in feedback_text for marker in ("保留", "留下", "一定要", "必须保留")) or any(
        marker in lowered for marker in ("keep", "must keep", "leave in")
    )
    remove_intent = any(marker in feedback_text for marker in ("不要", "删掉", "去掉", "移除")) or any(
        marker in lowered for marker in ("remove", "drop", "avoid", "cut")
    )

    if duration_override is not None:
        feedback_constraints["target_duration"] = duration_override
    if keep_intent:
        parsed_feedback["must_keep_candidate_ids"] = candidate_ids
        parsed_feedback["must_keep_quotes"] = quoted_phrases
    if remove_intent:
        parsed_feedback["forbidden_candidate_ids"] = candidate_ids
        if quoted_phrases:
            parsed_feedback["open_issues"].extend(
                [f"Avoid quote during replanning: {quote}" for quote in quoted_phrases]
            )

    parsed_feedback["focus_hints"] = _feedback_focus_hints(feedback_text)
    if not candidate_ids and not quoted_phrases and duration_override is None:
        parsed_feedback["open_issues"].append(_compact_text(feedback_text, limit=140))

    if parsed_feedback["must_keep_candidate_ids"]:
        memory.locked_elements.must_keep_candidate_ids = _merge_unique(
            memory.locked_elements.must_keep_candidate_ids,
            parsed_feedback["must_keep_candidate_ids"],
            limit=12,
        )
    if parsed_feedback["forbidden_candidate_ids"]:
        memory.locked_elements.forbidden_candidate_ids = _merge_unique(
            memory.locked_elements.forbidden_candidate_ids,
            parsed_feedback["forbidden_candidate_ids"],
            limit=12,
        )
    if parsed_feedback["must_keep_quotes"]:
        memory.locked_elements.must_keep_quotes = _merge_unique(
            memory.locked_elements.must_keep_quotes,
            parsed_feedback["must_keep_quotes"],
            limit=8,
        )
    if parsed_feedback["open_issues"]:
        memory.open_issues = _merge_unique(memory.open_issues, parsed_feedback["open_issues"], limit=8)
    if parsed_feedback["focus_hints"]:
        memory.planning_focus = _merge_unique(memory.planning_focus, parsed_feedback["focus_hints"], limit=8)

    record_user_instruction(
        memory,
        turn_id=turn_id,
        raw_text=feedback_text,
        parsed_constraints=feedback_constraints,
        stage=stage,
    )
    _append_event(
        memory,
        event_type="revision_feedback",
        stage=stage,
        summary=f"Applied revision feedback from turn {turn_id}.",
        metadata={
            "constraints": feedback_constraints,
            "must_keep_candidate_ids": parsed_feedback["must_keep_candidate_ids"],
            "forbidden_candidate_ids": parsed_feedback["forbidden_candidate_ids"],
            "must_keep_quotes": parsed_feedback["must_keep_quotes"],
            "open_issues": parsed_feedback["open_issues"],
        },
    )
    _set_working_summary(
        memory,
        [
            f"Latest revision request: {_compact_text(feedback_text, limit=120)}",
            (
                f"Target duration updated to {feedback_constraints['target_duration']} seconds."
                if "target_duration" in feedback_constraints
                else f"Current target duration remains {memory.active_constraints.get('target_duration', 'unknown')} seconds."
            ),
            (
                "Planner now has unresolved revision issues to address."
                if memory.open_issues
                else "No unresolved revision issues are recorded."
            ),
        ],
    )
    return {
        "constraints": feedback_constraints,
        **parsed_feedback,
    }


def record_retrieved_context(memory: PlannerMemory, retrieved_context: RetrievedContext) -> None:
    """Record the latest retrieved strategy context into planner memory."""

    chunk_ids = [chunk.chunk_id for chunk in retrieved_context.chunks[:8]]
    titles = [chunk.title for chunk in retrieved_context.chunks[:5]]
    strategy_types = list(dict.fromkeys(chunk.strategy_type for chunk in retrieved_context.chunks))
    strategy_focus = [f"Apply {strategy_type.replace('_', ' ')} guidance." for strategy_type in strategy_types[:3]]

    memory.retrieved_chunk_ids = chunk_ids
    memory.retrieved_rule_titles = titles
    memory.planning_focus = _merge_unique(memory.planning_focus, strategy_focus)

    _append_event(
        memory,
        event_type="retrieved_context",
        stage=WorkflowStage.RAG_CONTEXT_RETRIEVED.value,
        summary=f"Retrieved {len(retrieved_context.chunks)} strategy chunks for planner context.",
        metadata={
            "chunk_ids": chunk_ids,
            "titles": titles,
            "strategy_types": strategy_types[:5],
        },
    )
    if titles:
        _set_working_summary(
            memory,
            [
                f"Retrieved {len(retrieved_context.chunks)} strategy chunks for planning.",
                f"Top rules: {', '.join(titles[:3])}.",
                f"Current target duration: {memory.active_constraints.get('target_duration', 'unknown')} seconds.",
            ],
        )


def record_plan_generation(memory: PlannerMemory, editing_plan: EditingPlan) -> None:
    """Record one generated editing plan into planner memory."""

    selected_candidate_ids = list(
        dict.fromkeys(
            candidate_id
            for item in editing_plan.timeline_items
            for candidate_id in item.candidate_ids
            if candidate_id
        )
    )
    beat_roles = list(dict.fromkeys(beat.role for beat in editing_plan.beats if beat.role))
    beat_focus = [f"Maintain a strong {role} beat." for role in beat_roles[:4]]

    memory.current_plan_version = editing_plan.plan_version
    memory.open_issues = _merge_unique([], list(editing_plan.warnings), limit=6)
    memory.planning_focus = _merge_unique(memory.planning_focus, beat_focus)
    memory.revision_history.append(
        PlannerRevisionRecord(
            plan_version=editing_plan.plan_version,
            summary=(
                f"Generated {len(editing_plan.beats)} beats and {len(editing_plan.timeline_items)} timeline items "
                f"using {editing_plan.generation_mode} mode."
            ),
            changed_item_ids=[item.item_id for item in editing_plan.timeline_items],
            warnings=list(editing_plan.warnings),
        )
    )
    _append_event(
        memory,
        event_type="plan_generated",
        stage=WorkflowStage.EDITING_PLAN_GENERATED.value,
        summary=f"Saved editing plan v{editing_plan.plan_version} into planner memory.",
        metadata={
            "plan_version": editing_plan.plan_version,
            "beat_count": len(editing_plan.beats),
            "timeline_item_count": len(editing_plan.timeline_items),
            "candidate_ids": selected_candidate_ids[:10],
            "warnings": list(editing_plan.warnings),
        },
    )
    _set_working_summary(
        memory,
        [
            f"Plan v{editing_plan.plan_version} generated with {len(editing_plan.beats)} beats "
            f"and {len(editing_plan.timeline_items)} timeline items.",
            f"Selected candidates: {', '.join(selected_candidate_ids[:4]) or 'none recorded'}.",
            "The latest plan includes planner warnings to revisit."
            if editing_plan.warnings
            else "The latest plan generated without planner warnings.",
        ],
    )


def record_review_outcome(memory: PlannerMemory, review_report: ReviewReport) -> None:
    """Record one review-stage outcome into planner memory for future revisions."""

    failed_checks = [check.message for check in review_report.checks if not check.passed]
    open_issues = _merge_unique(failed_checks, review_report.suggestions, limit=8)
    memory.open_issues = open_issues

    _append_event(
        memory,
        event_type="review_completed",
        stage=WorkflowStage.REVIEW_REPORT_GENERATED.value,
        summary=(
            f"Review completed with passed={review_report.passed} and "
            f"score={round(review_report.score, 2)}."
        ),
        metadata={
            "passed": review_report.passed,
            "score": review_report.score,
            "failed_check_count": len(failed_checks),
            "suggestion_count": len(review_report.suggestions),
        },
    )
    _set_working_summary(
        memory,
        [
            f"Latest review status: {'passed' if review_report.passed else 'failed'} "
            f"with score {round(review_report.score, 2)}.",
            f"Outstanding issues: {open_issues[0]}" if open_issues else "No outstanding review issues recorded.",
            f"Latest saved plan version: v{memory.current_plan_version}.",
        ],
    )
