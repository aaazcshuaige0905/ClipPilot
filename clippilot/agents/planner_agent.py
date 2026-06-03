from itertools import combinations

from clippilot.schemas.editing_plan import (
    EditingClip,
    EditingPlan,
    HighlightCandidate,
    HighlightCandidatesResult,
)
from clippilot.schemas.transcript import TranscriptResult, TranscriptSegment
from clippilot.schemas.user_request import UserRequest
from clippilot.schemas.video_info import VideoInfo

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


def _normalize_text(text: str) -> str:
    """Normalize text for duplicate detection across candidates and transcript fallback clips."""

    lowered = text.lower().strip()
    filtered = [character for character in lowered if character.isalnum()]
    return "".join(filtered)


def _is_hook_candidate(candidate: HighlightCandidate) -> bool:
    """Detect whether a candidate is suitable as the opening hook clip."""

    text = candidate.text.strip()
    return candidate.start <= 20 or any(marker in text for marker in HOOK_MARKERS)


def _clip_purpose(candidate: HighlightCandidate, is_first_clip: bool) -> str:
    """Choose a readable purpose label for the selected clip."""

    if is_first_clip and _is_hook_candidate(candidate):
        return "hook"
    if candidate.score >= 0.8:
        return "core_point"
    if candidate.score >= 0.65:
        return "supporting_point"
    return "context"


def _candidate_to_clip(candidate: HighlightCandidate, clip_index: int, is_first_clip: bool) -> EditingClip:
    """Convert one highlight candidate into an executable editing clip."""

    text = candidate.text.strip()
    return EditingClip(
        clip_id=f"clip_{clip_index:02d}",
        source_start=round(candidate.start, 2),
        source_end=round(candidate.end, 2),
        duration=round(candidate.end - candidate.start, 2),
        purpose=_clip_purpose(candidate, is_first_clip=is_first_clip),
        text=text,
        subtitle=text,
        score=round(candidate.score, 2),
        reason=candidate.reason,
    )


def _deduplicate_candidates(candidates: list[HighlightCandidate]) -> list[HighlightCandidate]:
    """Remove duplicate or near-duplicate candidates by normalized transcript text."""

    seen_texts: set[str] = set()
    unique_candidates: list[HighlightCandidate] = []

    for candidate in candidates:
        normalized_text = _normalize_text(candidate.text)
        if not normalized_text or normalized_text in seen_texts:
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


def _rank_candidates(
    candidates: HighlightCandidatesResult,
    transcript: TranscriptResult,
) -> tuple[list[HighlightCandidate], bool]:
    """Create the ranked candidate pool and indicate whether transcript fallback was used."""

    unique_candidates = _deduplicate_candidates(candidates.candidates)
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


def _combination_score(
    selected: list[HighlightCandidate],
    target_duration: int,
) -> tuple[float, float, float, float]:
    """Build a sortable score tuple for one candidate combination."""

    total_duration = round(sum(item.end - item.start for item in selected), 2)
    score_sum = round(sum(item.score for item in selected), 4)
    return (
        abs(total_duration - target_duration),
        0.0 if total_duration <= target_duration else total_duration - target_duration,
        -score_sum,
        -total_duration,
    )


def _find_best_plan_candidates(
    ranked_candidates: list[HighlightCandidate],
    target_duration: int,
) -> list[HighlightCandidate]:
    """Choose the best candidate subset while keeping total duration under the hard cap."""

    if not ranked_candidates:
        return []

    upper_bound = target_duration + 5
    hook_candidates = [candidate for candidate in ranked_candidates if _is_hook_candidate(candidate)]
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


def _build_editing_notes(
    user_request: UserRequest,
    video_info: VideoInfo,
    clips: list[EditingClip],
    fallback_used: bool,
    source_candidate_count: int,
) -> tuple[list[str], list[str]]:
    """Build user-facing editing notes and warnings for the generated plan."""

    notes = [
        f"Target platform: {user_request.target_platform}.",
        f"Requested duration: {user_request.target_duration} seconds.",
        f"Source video duration: {round(video_info.duration_seconds, 2)} seconds.",
        f"Selected {len(clips)} clips from {source_candidate_count} ranked candidates.",
    ]
    warnings: list[str] = []

    if clips and clips[0].purpose == "hook":
        notes.append("The first clip is optimized as a hook-style opening segment.")

    if fallback_used:
        warning = "Highlight candidates were insufficient, so transcript fallback clips were added to complete the timeline."
        warnings.append(warning)
        notes.append(warning)

    return notes, warnings


def build_editing_plan(
    task_id: str,
    user_request: UserRequest,
    video_info: VideoInfo,
    transcript: TranscriptResult,
    candidates: HighlightCandidatesResult,
) -> EditingPlan:
    """Build an executable timeline plan from ranked highlights and transcript fallback clips."""

    ranked_candidates, fallback_used = _rank_candidates(candidates=candidates, transcript=transcript)
    selected_candidates = _find_best_plan_candidates(ranked_candidates=ranked_candidates, target_duration=user_request.target_duration)
    selected_duration = round(sum(item.end - item.start for item in selected_candidates), 2)

    if selected_duration < max(3.0, user_request.target_duration - 8):
        expanded_candidates = _append_fallback_candidates(ranked_candidates=ranked_candidates, transcript=transcript)
        if len(expanded_candidates) > len(ranked_candidates):
            fallback_used = True
            ranked_candidates = expanded_candidates
            selected_candidates = _find_best_plan_candidates(
                ranked_candidates=ranked_candidates,
                target_duration=user_request.target_duration,
            )

    clips: list[EditingClip] = []
    for index, candidate in enumerate(selected_candidates, start=1):
        clips.append(_candidate_to_clip(candidate=candidate, clip_index=index, is_first_clip=index == 1))

    notes, warnings = _build_editing_notes(
        user_request=user_request,
        video_info=video_info,
        clips=clips,
        fallback_used=fallback_used,
        source_candidate_count=len(candidates.candidates),
    )

    return EditingPlan(
        task_id=task_id,
        target_duration=user_request.target_duration,
        total_duration=round(sum(clip.duration for clip in clips), 2),
        clips=clips,
        editing_notes=notes,
        warnings=warnings,
        strategy="rule_based_executable_timeline",
        source_candidate_count=len(candidates.candidates),
    )
