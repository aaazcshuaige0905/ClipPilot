from __future__ import annotations

import os
from itertools import combinations

from clippilot.core.context_builder import build_compressed_planning_context
from clippilot.rag.schemas import RetrievedContext
from clippilot.schemas.editing_plan import (
    EditingPlan,
    HighlightCandidate,
    HighlightCandidatesResult,
    PlanningBeat,
    TimelineItem,
    TimelineSourceRef,
)
from clippilot.schemas.planner_memory import PlannerMemory
from clippilot.schemas.planning_context import CompressedPlanningContext
from clippilot.schemas.project_state import FineGrainedUnit, LLMHighlightCandidate, VideoTimeline
from clippilot.schemas.transcript import TranscriptResult, TranscriptSegment
from clippilot.schemas.user_request import UserRequest
from clippilot.schemas.video_info import VideoInfo

DEFAULT_QWEN_PLANNER_MODEL = "qwen-plus-latest"
DEFAULT_QWEN_PLANNER_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
HOOK_MARKERS = (
    "为什么",
    "先说结论",
    "结论是",
    "最重要",
    "真正",
    "一定要",
    "我认为",
    "核心",
    "关键",
)


def _is_cjk_character(character: str) -> bool:
    """Return whether one character falls inside common CJK Unicode ranges."""

    return "\u3400" <= character <= "\u9fff" or "\uf900" <= character <= "\ufaff"


def _normalize_text(text: str) -> str:
    """Normalize text for duplicate detection across candidates and transcript fallback clips."""

    lowered = text.lower().strip()
    filtered = [
        character
        for character in lowered
        if character.isalnum() or _is_cjk_character(character) or ord(character) > 127
    ]
    return "".join(filtered)


def _compact_text(text: str, limit: int = 90) -> str:
    """Collapse whitespace and trim long excerpts for readable notes and prompts."""

    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 3]}..."


def _candidate_target_duration(candidate: LLMHighlightCandidate) -> float:
    """Return the preferred planning duration for one LLM candidate."""

    return round(min(candidate.duration, candidate.trim_policy.ideal_duration), 2)


def _role_to_goal(role: str) -> str:
    """Translate a beat role into a human-readable editing goal."""

    goals = {
        "hook": "Open with immediate tension, payoff, or a strong claim.",
        "core_point": "Deliver the main informational or emotional value.",
        "supporting_point": "Add context that supports the main point without slowing pacing.",
        "ending": "Close with a summary, reflection, or call to action.",
        "context": "Bridge semantic gaps while keeping the story coherent.",
    }
    return goals.get(role, "Advance the story and keep the audience engaged.")


def _llm_role_for_candidate(candidate: LLMHighlightCandidate, order: int, total: int) -> str:
    """Choose the narrative beat role for one selected LLM candidate."""

    if order == 1 and (candidate.semantic_role == "opening_hook" or candidate.scores.hook >= 0.7):
        return "hook"
    if order == total and candidate.semantic_role == "ending":
        return "ending"
    if candidate.scores.overall >= 0.78:
        return "core_point"
    if candidate.semantic_role == "ending":
        return "ending"
    return "supporting_point"


def _rule_candidate_purpose(candidate: HighlightCandidate, is_first_item: bool) -> str:
    """Choose a readable purpose label for one rule-based timeline item."""

    if is_first_item and _is_rule_hook_candidate(candidate):
        return "hook"
    if candidate.score >= 0.8:
        return "core_point"
    if candidate.score >= 0.65:
        return "supporting_point"
    return "context"


def _is_rule_hook_candidate(candidate: HighlightCandidate) -> bool:
    """Detect whether one rule-based candidate is suitable as the opening hook clip."""

    text = candidate.text.strip()
    return candidate.start <= 20 or any(marker in text for marker in HOOK_MARKERS)


def _deduplicate_candidates(candidates: list[HighlightCandidate]) -> list[HighlightCandidate]:
    """Remove duplicate or near-duplicate candidates by normalized transcript text."""

    seen_texts: set[str] = set()
    unique_candidates: list[HighlightCandidate] = []

    for candidate in candidates:
        normalized_text = _normalize_text(candidate.text) or f"{round(candidate.start, 2)}_{round(candidate.end, 2)}"
        if normalized_text in seen_texts:
            continue
        seen_texts.add(normalized_text)
        unique_candidates.append(candidate)

    return unique_candidates


def _deduplicate_llm_candidates(candidates: list[LLMHighlightCandidate]) -> list[LLMHighlightCandidate]:
    """Remove duplicate LLM candidates by normalized excerpt text."""

    seen_texts: set[str] = set()
    unique_candidates: list[LLMHighlightCandidate] = []

    for candidate in candidates:
        normalized_text = _normalize_text(candidate.transcript_excerpt or candidate.summary) or candidate.candidate_id
        if normalized_text in seen_texts:
            continue
        seen_texts.add(normalized_text)
        unique_candidates.append(candidate)

    return unique_candidates


def _generate_transcript_fallback_candidates(transcript: TranscriptResult) -> list[HighlightCandidate]:
    """Create extra planning candidates from transcript segments when ranked highlights are insufficient."""

    fallback_candidates: list[HighlightCandidate] = []
    seen_ranges: set[tuple[float, float]] = set()
    segments = transcript.segments

    for start_index in range(len(segments)):
        collected: list[TranscriptSegment] = []
        window_start = float(segments[start_index].start)

        for current_index in range(start_index, len(segments)):
            segment = segments[current_index]
            collected.append(segment)
            window_end = float(segment.end)
            duration = round(window_end - window_start, 2)

            if duration > 20.0:
                break
            if duration < 3.0:
                continue

            text = " ".join(item.text.strip() for item in collected if item.text.strip())
            range_key = (round(window_start, 2), round(window_end, 2))
            if not text or range_key in seen_ranges:
                continue

            fallback_candidates.append(
                HighlightCandidate(
                    start=range_key[0],
                    end=range_key[1],
                    text=text,
                    score=0.35,
                    reason="Fallback clip built from transcript because ranked highlights were insufficient.",
                )
            )
            seen_ranges.add(range_key)

    fallback_candidates.sort(key=lambda item: (item.end - item.start, item.start), reverse=True)
    return fallback_candidates


def _apply_strategy_context(candidates: list[HighlightCandidate], retrieved_context: RetrievedContext | None) -> None:
    """Apply a lightweight score bonus from retrieved strategy context to rule-based candidates."""

    if not retrieved_context or not retrieved_context.chunks:
        return

    strategy_types = {chunk.strategy_type for chunk in retrieved_context.chunks}
    for candidate in candidates:
        bonus = 0.0
        duration = round(candidate.end - candidate.start, 2)
        if "hook_rule" in strategy_types and _is_rule_hook_candidate(candidate):
            bonus += 0.08
        if "pacing_rule" in strategy_types and 3.0 <= duration <= 12.0:
            bonus += 0.04
        if "platform_rule" in strategy_types and candidate.start <= 30:
            bonus += 0.03
        candidate.score = round(min(1.0, candidate.score + bonus), 4)


def _apply_llm_strategy_context(
    candidates: list[LLMHighlightCandidate],
    retrieved_context: RetrievedContext | None,
    planner_memory: PlannerMemory | None,
) -> None:
    """Apply lightweight planner and retrieval bonuses to transcript-backed LLM candidates."""

    strategy_types = {chunk.strategy_type for chunk in retrieved_context.chunks} if retrieved_context else set()
    locked_quotes = planner_memory.locked_elements.must_keep_quotes if planner_memory else []
    locked_candidates = set(planner_memory.locked_elements.must_keep_candidate_ids) if planner_memory else set()
    forbidden_candidates = set(planner_memory.locked_elements.forbidden_candidate_ids) if planner_memory else set()

    for candidate in candidates:
        score = candidate.scores.overall
        if candidate.candidate_id in forbidden_candidates:
            score = max(0.0, score - 0.4)
        if candidate.semantic_role == "opening_hook" and "hook_rule" in strategy_types:
            score += 0.08
        if 3.0 <= _candidate_target_duration(candidate) <= 12.0 and "pacing_rule" in strategy_types:
            score += 0.05
        if candidate.candidate_id in locked_candidates:
            score += 0.18
        if locked_quotes and any(quote in candidate.transcript_excerpt for quote in locked_quotes):
            score += 0.16
        candidate.scores.overall = round(min(1.0, max(0.0, score)), 4)


def _rank_rule_candidates(
    candidates: HighlightCandidatesResult,
    transcript: TranscriptResult,
    retrieved_context: RetrievedContext | None = None,
) -> tuple[list[HighlightCandidate], bool]:
    """Create the ranked rule-based candidate pool and indicate whether transcript fallback was used."""

    unique_candidates = _deduplicate_candidates(candidates.candidates)
    _apply_strategy_context(unique_candidates, retrieved_context)
    unique_candidates.sort(key=lambda item: item.score, reverse=True)

    fallback_used = False
    if len(unique_candidates) < 2:
        fallback_used = True
        existing_texts = {_normalize_text(item.text) for item in unique_candidates}
        for fallback_candidate in _generate_transcript_fallback_candidates(transcript):
            normalized_text = _normalize_text(fallback_candidate.text)
            if normalized_text in existing_texts:
                continue
            existing_texts.add(normalized_text)
            unique_candidates.append(fallback_candidate)

    unique_candidates.sort(key=lambda item: item.score, reverse=True)
    return unique_candidates, fallback_used


def _append_fallback_candidates(
    ranked_candidates: list[HighlightCandidate],
    transcript: TranscriptResult,
) -> list[HighlightCandidate]:
    """Append transcript fallback candidates while keeping candidate text unique."""

    existing_texts = {_normalize_text(item.text) for item in ranked_candidates}
    extended_candidates = list(ranked_candidates)

    for fallback_candidate in _generate_transcript_fallback_candidates(transcript):
        normalized_text = _normalize_text(fallback_candidate.text)
        if normalized_text in existing_texts:
            continue
        existing_texts.add(normalized_text)
        extended_candidates.append(fallback_candidate)

    extended_candidates.sort(key=lambda item: item.score, reverse=True)
    return extended_candidates


def _combination_score(selected: list[HighlightCandidate], target_duration: int) -> tuple[float, float, float, float]:
    """Build a sortable score tuple for one rule-based candidate combination."""

    total_duration = round(sum(item.end - item.start for item in selected), 2)
    score_sum = round(sum(item.score for item in selected), 4)
    return (
        abs(total_duration - target_duration),
        0.0 if total_duration <= target_duration else total_duration - target_duration,
        -score_sum,
        -total_duration,
    )


def _find_best_rule_candidates(ranked_candidates: list[HighlightCandidate], target_duration: int) -> list[HighlightCandidate]:
    """Choose the best rule-based candidate subset while keeping total duration under the hard cap."""

    if not ranked_candidates:
        return []

    upper_bound = target_duration + 5
    hook_candidates = [candidate for candidate in ranked_candidates if _is_rule_hook_candidate(candidate)]
    first_clip_pool = hook_candidates or ranked_candidates[:1]

    best_selection: list[HighlightCandidate] = []
    best_sort_key: tuple[float, float, float, float] | None = None

    for first_clip in first_clip_pool:
        remaining = [candidate for candidate in ranked_candidates if candidate != first_clip]
        first_duration = round(first_clip.end - first_clip.start, 2)
        if first_duration > upper_bound:
            continue

        current_best = [first_clip]
        current_key = _combination_score(current_best, target_duration)

        for subset_size in range(len(remaining) + 1):
            for subset in combinations(remaining, subset_size):
                selection = [first_clip, *subset]
                total_duration = round(sum(item.end - item.start for item in selection), 2)
                if total_duration > upper_bound:
                    continue
                sort_key = _combination_score(selection, target_duration)
                if sort_key < current_key:
                    current_best = selection
                    current_key = sort_key

        if best_sort_key is None or current_key < best_sort_key:
            best_selection = current_best
            best_sort_key = current_key

    return best_selection


def _build_qwen_planner_request(
    task_id: str,
    video_info: VideoInfo,
    compressed_context: CompressedPlanningContext,
) -> dict:
    """Build the future structured LLM planner request using compressed local context."""

    model_name = os.getenv("CLIP_PILOT_QWEN_PLANNER_MODEL", DEFAULT_QWEN_PLANNER_MODEL).strip() or DEFAULT_QWEN_PLANNER_MODEL
    base_url = os.getenv("CLIP_PILOT_QWEN_PLANNER_BASE_URL", DEFAULT_QWEN_PLANNER_BASE_URL).strip() or DEFAULT_QWEN_PLANNER_BASE_URL

    return {
        "provider": "qwen",
        "base_url": base_url,
        "model": model_name,
        "task_id": task_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a short-video planner. Build narrative beats and timeline items from structured "
                    "candidate clips, compressed local editing context, and platform strategy rules."
                ),
            },
            {
                "role": "user",
                "content": {
                    "instruction": (
                        "Return structured JSON with beats and timeline_items. Prefer concise, punchy pacing. "
                        "Use the compressed request, memory, candidate, timeline, and rule context instead of the full transcript."
                    ),
                    "task_id": task_id,
                    "video_metadata": {
                        "duration_seconds": video_info.duration_seconds,
                        "width": video_info.width,
                        "height": video_info.height,
                    },
                    "request_summary": compressed_context.request_summary.model_dump(),
                    "memory_summary": compressed_context.memory_summary.model_dump(),
                    "timeline_segments": [segment.model_dump() for segment in compressed_context.timeline_segments],
                    "top_candidates": [candidate.model_dump() for candidate in compressed_context.top_candidates],
                    "retrieved_rules": [rule.model_dump() for rule in compressed_context.retrieved_rules],
                    "risks": [risk.model_dump() for risk in compressed_context.risks],
                    "context_stats": compressed_context.stats,
                },
            },
        ],
        "response_format": {
            "type": "json_schema",
            "name": "editing_plan_beats",
        },
    }


def _rank_llm_candidates(
    llm_candidates: list[LLMHighlightCandidate],
    retrieved_context: RetrievedContext | None,
    planner_memory: PlannerMemory | None,
) -> list[LLMHighlightCandidate]:
    """Rank transcript-backed LLM candidates with local context bonuses."""

    ranked_candidates = _deduplicate_llm_candidates(llm_candidates)
    _apply_llm_strategy_context(ranked_candidates, retrieved_context, planner_memory)
    ranked_candidates.sort(
        key=lambda item: (
            item.scores.overall,
            item.scores.hook,
            item.scores.editability,
            -item.source_start,
        ),
        reverse=True,
    )
    return ranked_candidates


def _select_llm_candidates(
    ranked_candidates: list[LLMHighlightCandidate],
    target_duration: int,
    planner_memory: PlannerMemory | None,
) -> list[LLMHighlightCandidate]:
    """Pick a compact set of LLM candidates that can form the short-video structure."""

    if not ranked_candidates:
        return []

    upper_bound = target_duration + 5
    lower_bound = max(5.0, target_duration - 6.0)
    locked_candidate_ids = set(planner_memory.locked_elements.must_keep_candidate_ids) if planner_memory else set()
    forbidden_candidate_ids = set(planner_memory.locked_elements.forbidden_candidate_ids) if planner_memory else set()

    selected: list[LLMHighlightCandidate] = []
    selected_ids: set[str] = set()
    total_duration = 0.0
    used_redundancy_groups: set[str] = set()

    def try_add(candidate: LLMHighlightCandidate) -> bool:
        nonlocal total_duration

        if candidate.candidate_id in selected_ids or candidate.candidate_id in forbidden_candidate_ids:
            return False
        if candidate.redundancy_group and candidate.redundancy_group in used_redundancy_groups and not candidate.must_keep:
            return False

        candidate_duration = _candidate_target_duration(candidate)
        if selected and total_duration + candidate_duration > upper_bound and not candidate.must_keep:
            return False

        selected.append(candidate)
        selected_ids.add(candidate.candidate_id)
        total_duration = round(total_duration + candidate_duration, 2)
        if candidate.redundancy_group:
            used_redundancy_groups.add(candidate.redundancy_group)
        return True

    for candidate in ranked_candidates:
        if candidate.candidate_id in locked_candidate_ids:
            try_add(candidate)

    opening_hook = next(
        (candidate for candidate in ranked_candidates if candidate.semantic_role == "opening_hook"),
        ranked_candidates[0],
    )
    try_add(opening_hook)

    core_candidates = [
        candidate
        for candidate in ranked_candidates
        if candidate.candidate_id not in selected_ids and candidate.semantic_role != "ending"
    ]
    for candidate in core_candidates:
        try_add(candidate)
        if total_duration >= lower_bound and len(selected) >= 2:
            break

    ending_candidate = next(
        (
            candidate
            for candidate in sorted(ranked_candidates, key=lambda item: item.source_start, reverse=True)
            if candidate.semantic_role == "ending" and candidate.candidate_id not in selected_ids
        ),
        None,
    )
    if ending_candidate is not None and (total_duration < lower_bound or len(selected) < 3 or ending_candidate.must_keep):
        try_add(ending_candidate)

    if total_duration < lower_bound:
        for candidate in ranked_candidates:
            if candidate.candidate_id in selected_ids:
                continue
            if try_add(candidate) and total_duration >= lower_bound:
                break

    ordered_selection = sorted(selected, key=lambda item: item.source_start)
    opening = next((candidate for candidate in ordered_selection if candidate.semantic_role == "opening_hook"), None)
    ending = next((candidate for candidate in reversed(ordered_selection) if candidate.semantic_role == "ending"), None)
    opening_id = opening.candidate_id if opening is not None else None
    ending_id = ending.candidate_id if ending is not None else None
    middle = [candidate for candidate in ordered_selection if candidate.candidate_id not in {opening_id, ending_id}]

    final_selection: list[LLMHighlightCandidate] = []
    if opening is not None:
        final_selection.append(opening)
    final_selection.extend(middle)
    if ending is not None and ending not in final_selection:
        final_selection.append(ending)

    return final_selection or ordered_selection


def _build_source_refs_for_candidate(
    candidate: LLMHighlightCandidate,
    fine_grained_units: list[FineGrainedUnit] | None,
) -> tuple[list[TimelineSourceRef], str]:
    """Build timeline source refs for one selected LLM candidate."""

    units_by_id = {unit.unit_id: unit for unit in (fine_grained_units or [])}
    ordered_units = [units_by_id[unit_id] for unit_id in candidate.transcript_unit_ids if unit_id in units_by_id]

    if not ordered_units:
        return [
            TimelineSourceRef(
                ref_id=candidate.candidate_id,
                ref_type="candidate",
                start=round(candidate.trim_policy.preferred_start, 2),
                end=round(candidate.trim_policy.preferred_end, 2),
                text=candidate.transcript_excerpt,
            )
        ], "continuous_trim"

    if (
        candidate.semantic_role == "opening_hook"
        and len(ordered_units) >= 3
        and round(ordered_units[-1].start - ordered_units[0].end, 2) >= 0.5
        and round(ordered_units[-1].end - ordered_units[0].start, 2) <= 6.0
    ):
        return [
            TimelineSourceRef(
                ref_id=ordered_units[0].unit_id,
                ref_type="unit",
                start=ordered_units[0].start,
                end=ordered_units[0].end,
                text=ordered_units[0].text,
            ),
            TimelineSourceRef(
                ref_id=ordered_units[-1].unit_id,
                ref_type="unit",
                start=ordered_units[-1].start,
                end=ordered_units[-1].end,
                text=ordered_units[-1].text,
            ),
        ], "montage"

    preferred_units = [
        unit
        for unit in ordered_units
        if unit.start >= candidate.trim_policy.preferred_start - 0.01 and unit.end <= candidate.trim_policy.preferred_end + 0.01
    ]
    active_units = preferred_units or ordered_units
    return [
        TimelineSourceRef(
            ref_id=unit.unit_id,
            ref_type="unit",
            start=unit.start,
            end=unit.end,
            text=unit.text,
        )
        for unit in active_units
    ], "continuous_trim"


def _build_llm_beats_and_items(
    selected_candidates: list[LLMHighlightCandidate],
    fine_grained_units: list[FineGrainedUnit] | None,
) -> tuple[list[PlanningBeat], list[TimelineItem]]:
    """Build richer beats and timeline items from the selected LLM candidate set."""

    beats: list[PlanningBeat] = []
    timeline_items: list[TimelineItem] = []
    total = len(selected_candidates)

    for index, candidate in enumerate(selected_candidates, start=1):
        role = _llm_role_for_candidate(candidate, order=index, total=total)
        beat_id = f"beat_{index:02d}"
        beat = PlanningBeat(
            beat_id=beat_id,
            order=index,
            role=role,
            goal=_role_to_goal(role),
            target_duration=_candidate_target_duration(candidate),
            summary=_compact_text(candidate.summary or candidate.transcript_excerpt, limit=120),
            source_candidate_ids=[candidate.candidate_id],
            notes=[candidate.reason],
        )
        beats.append(beat)

        source_refs, assembly_mode = _build_source_refs_for_candidate(candidate, fine_grained_units=fine_grained_units)
        text = " ".join(reference.text.strip() for reference in source_refs if reference.text.strip()) or candidate.transcript_excerpt
        timeline_items.append(
            TimelineItem(
                item_id=f"item_{index:02d}",
                beat_id=beat_id,
                purpose=role,
                assembly_mode=assembly_mode,
                source_start=round(min(reference.start for reference in source_refs), 2),
                source_end=round(max(reference.end for reference in source_refs), 2),
                duration=_candidate_target_duration(candidate),
                candidate_ids=[candidate.candidate_id],
                source_unit_ids=[reference.ref_id for reference in source_refs if reference.ref_type == "unit"],
                source_refs=source_refs,
                text=text.strip(),
                subtitle=_compact_text(text.strip() or candidate.transcript_excerpt, limit=80),
                score=round(candidate.scores.overall, 2),
                transition=candidate.transition_hint,
                reason=candidate.reason,
            )
        )

    return beats, timeline_items


def _build_rule_based_richer_plan(
    task_id: str,
    user_request: UserRequest,
    video_info: VideoInfo,
    transcript: TranscriptResult,
    candidates: HighlightCandidatesResult,
    retrieved_context: RetrievedContext | None,
) -> EditingPlan:
    """Fallback to the existing rule-based candidate flow and wrap it in richer plan structures."""

    ranked_candidates, fallback_used = _rank_rule_candidates(
        candidates=candidates,
        transcript=transcript,
        retrieved_context=retrieved_context,
    )
    selected_candidates = _find_best_rule_candidates(ranked_candidates=ranked_candidates, target_duration=user_request.target_duration)
    selected_duration = round(sum(item.end - item.start for item in selected_candidates), 2)

    if selected_duration < max(3.0, user_request.target_duration - 8):
        expanded_candidates = _append_fallback_candidates(ranked_candidates=ranked_candidates, transcript=transcript)
        if len(expanded_candidates) > len(ranked_candidates):
            fallback_used = True
            ranked_candidates = expanded_candidates
            selected_candidates = _find_best_rule_candidates(
                ranked_candidates=ranked_candidates,
                target_duration=user_request.target_duration,
            )

    beats: list[PlanningBeat] = []
    timeline_items: list[TimelineItem] = []
    for index, candidate in enumerate(selected_candidates, start=1):
        item_purpose = _rule_candidate_purpose(candidate=candidate, is_first_item=index == 1)
        item_text = candidate.text.strip()
        item_duration = round(candidate.end - candidate.start, 2)
        beat_id = f"beat_{index:02d}"
        beats.append(
            PlanningBeat(
                beat_id=beat_id,
                order=index,
                role=item_purpose,
                goal=_role_to_goal(item_purpose),
                target_duration=item_duration,
                summary=_compact_text(item_text, limit=120),
                notes=[candidate.reason],
            )
        )
        timeline_items.append(
            TimelineItem(
                item_id=f"item_{index:02d}",
                beat_id=beat_id,
                purpose=item_purpose,
                assembly_mode="single_cut",
                source_start=round(candidate.start, 2),
                source_end=round(candidate.end, 2),
                duration=item_duration,
                text=item_text,
                subtitle=item_text,
                score=round(candidate.score, 2),
                transition="straight_cut",
                reason=candidate.reason,
                source_refs=[
                    TimelineSourceRef(
                        ref_id=f"rule_candidate_{index:02d}",
                        ref_type="candidate",
                        start=round(candidate.start, 2),
                        end=round(candidate.end, 2),
                        text=item_text,
                    )
                ],
            )
        )

    notes = [
        f"Target platform: {user_request.target_platform}.",
        f"Requested duration: {user_request.target_duration} seconds.",
        f"Source video duration: {round(video_info.duration_seconds, 2)} seconds.",
        f"Selected {len(timeline_items)} timeline items from {len(candidates.candidates)} ranked candidates.",
        "Planner used rule-based fallback because transcript-backed LLM planning inputs were unavailable.",
    ]
    warnings: list[str] = []
    if retrieved_context and retrieved_context.chunks:
        top_titles = ", ".join(chunk.title for chunk in retrieved_context.chunks[:3])
        notes.append(f"Planner retrieved {len(retrieved_context.chunks)} strategy chunks: {top_titles}.")
    if fallback_used:
        warning = "Highlight candidates were insufficient, so transcript fallback clips were added to complete the timeline."
        warnings.append(warning)
        notes.append(warning)

    return EditingPlan(
        task_id=task_id,
        plan_version=1,
        target_duration=user_request.target_duration,
        total_duration=round(sum(item.duration for item in timeline_items), 2),
        beats=beats,
        timeline_items=timeline_items,
        editing_notes=notes,
        warnings=warnings,
        strategy="rule_based_executable_timeline",
        generation_mode="rule_fallback",
        source_candidate_count=len(candidates.candidates),
    )


def build_editing_plan(
    task_id: str,
    user_request: UserRequest,
    video_info: VideoInfo,
    transcript: TranscriptResult,
    candidates: HighlightCandidatesResult,
    fine_grained_units: list[FineGrainedUnit] | None = None,
    timeline: VideoTimeline | None = None,
    llm_candidates: list[LLMHighlightCandidate] | None = None,
    retrieved_context: RetrievedContext | None = None,
    planner_memory: PlannerMemory | None = None,
) -> EditingPlan:
    """Build a richer editing plan from transcript-backed candidates and local planner memory."""

    if not llm_candidates:
        return _build_rule_based_richer_plan(
            task_id=task_id,
            user_request=user_request,
            video_info=video_info,
            transcript=transcript,
            candidates=candidates,
            retrieved_context=retrieved_context,
        )

    ranked_llm_candidates = _rank_llm_candidates(
        llm_candidates=llm_candidates,
        retrieved_context=retrieved_context,
        planner_memory=planner_memory,
    )
    compressed_context = build_compressed_planning_context(
        user_request=user_request,
        ranked_candidates=ranked_llm_candidates,
        fine_grained_units=fine_grained_units or [],
        timeline=timeline,
        retrieved_context=retrieved_context,
        planner_memory=planner_memory,
    )
    planner_request = _build_qwen_planner_request(
        task_id=task_id,
        video_info=video_info,
        compressed_context=compressed_context,
    )
    selected_candidates = _select_llm_candidates(
        ranked_candidates=ranked_llm_candidates,
        target_duration=user_request.target_duration,
        planner_memory=planner_memory,
    )
    if not selected_candidates:
        return _build_rule_based_richer_plan(
            task_id=task_id,
            user_request=user_request,
            video_info=video_info,
            transcript=transcript,
            candidates=candidates,
            retrieved_context=retrieved_context,
        )

    beats, timeline_items = _build_llm_beats_and_items(
        selected_candidates=selected_candidates,
        fine_grained_units=fine_grained_units,
    )

    notes = [
        f"Target platform: {user_request.target_platform}.",
        f"Requested duration: {user_request.target_duration} seconds.",
        f"Source video duration: {round(video_info.duration_seconds, 2)} seconds.",
        f"Planner built {len(beats)} beats and {len(timeline_items)} timeline items from {len(llm_candidates)} transcript-backed candidates.",
        (
            "Planner request contract prepared for future Qwen execution using a compressed planning context "
            "with request, memory, candidate, timeline, rule, and risk summaries."
        ),
        (
            "Compressed context includes "
            f"{compressed_context.stats['candidate_preview_count']} candidate previews, "
            f"{compressed_context.stats['timeline_segment_count']} timeline segments, "
            f"and {compressed_context.stats['retrieved_rule_count']} retrieved rules."
        ),
        f"Prepared planner provider/model: {planner_request['provider']}/{planner_request['model']}.",
    ]
    if retrieved_context and retrieved_context.chunks:
        top_titles = ", ".join(chunk.title for chunk in retrieved_context.chunks[:3])
        notes.append(f"Planner retrieved {len(retrieved_context.chunks)} strategy chunks: {top_titles}.")
    if planner_memory is not None and planner_memory.active_constraints:
        notes.append(
            "Planner memory active constraints: "
            + ", ".join(f"{key}={value}" for key, value in planner_memory.active_constraints.items())
            + "."
        )
    if compressed_context.memory_summary.planning_focus:
        notes.append("Compressed planning focus: " + ", ".join(compressed_context.memory_summary.planning_focus) + ".")
    if compressed_context.risks:
        notes.append(
            "Compressed context risks: "
            + ", ".join(risk.message for risk in compressed_context.risks[:2])
            + "."
        )

    return EditingPlan(
        task_id=task_id,
        plan_version=max(1, (planner_memory.current_plan_version + 1) if planner_memory is not None else 1),
        target_duration=user_request.target_duration,
        total_duration=round(sum(item.duration for item in timeline_items), 2),
        beats=beats,
        timeline_items=timeline_items,
        editing_notes=notes,
        warnings=[],
        strategy="llm_montage_planner",
        generation_mode="stub",
        source_candidate_count=len(candidates.candidates),
    )
