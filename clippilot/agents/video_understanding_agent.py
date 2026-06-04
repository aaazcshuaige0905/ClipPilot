import os
from pathlib import Path

from clippilot.schemas.project_state import (
    ContentAnalysis,
    FineGrainedUnit,
    HighlightScoreBreakdown,
    LLMHighlightCandidate,
    TimelineSegment,
    TrimPolicy,
    VideoTimeline,
)
from clippilot.schemas.transcript import TranscriptResult
from clippilot.schemas.user_request import UserRequest
from clippilot.schemas.video_info import VideoInfo

DEFAULT_QWEN_VIDEO_MODEL = "qwen-vl-max-latest"
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def build_qwen_video_understanding_request(
    video_path: Path,
    user_request: UserRequest,
    transcript: TranscriptResult,
    video_info: VideoInfo,
    fine_grained_units: list[FineGrainedUnit],
) -> dict:
    """Build the future OpenAI-compatible request payload for Qwen video understanding."""

    model_name = os.getenv("CLIP_PILOT_QWEN_VIDEO_MODEL", DEFAULT_QWEN_VIDEO_MODEL).strip() or DEFAULT_QWEN_VIDEO_MODEL
    base_url = os.getenv("CLIP_PILOT_QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL).strip() or DEFAULT_QWEN_BASE_URL

    transcript_preview = [
        {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        }
        for segment in transcript.segments[:12]
    ]
    fine_grained_preview = [
        {
            "unit_id": unit.unit_id,
            "start": unit.start,
            "end": unit.end,
            "text": unit.text,
            "has_pause_before": unit.has_pause_before,
            "has_pause_after": unit.has_pause_after,
            "keywords": unit.keywords,
        }
        for unit in fine_grained_units[:20]
    ]
    return {
        "provider": "qwen",
        "base_url": base_url,
        "model": model_name,
        "video_path": str(video_path),
        "metadata": {
            "duration_seconds": video_info.duration_seconds,
            "width": video_info.width,
            "height": video_info.height,
            "fps": video_info.fps,
            "has_audio": video_info.has_audio,
        },
        "user_request": user_request.model_dump(),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a video understanding assistant. Analyze the source video and transcript, "
                    "then return both coarse semantic timeline segments and fine-grained highlight candidates "
                    "that can be trimmed into a final short video."
                ),
            },
            {
                "role": "user",
                "content": {
                    "instruction": (
                        "Return structured JSON containing content_analysis, timeline_segments, and "
                        "highlight_candidates. Each highlight candidate must reference fine-grained unit IDs "
                        "and include a trim policy with min, ideal, and max durations."
                    ),
                    "target_platform": user_request.target_platform,
                    "target_duration": user_request.target_duration,
                    "edit_style": user_request.edit_style,
                    "language": user_request.language,
                    "video_path": str(video_path),
                    "transcript_preview": transcript_preview,
                    "fine_grained_units": fine_grained_preview,
                },
            },
        ],
        "response_format": {
            "type": "json_schema",
            "name": "video_understanding_timeline",
        },
    }


def _clip_text(text: str, limit: int = 120) -> str:
    """Trim transcript text so stub outputs stay readable."""

    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 3]}..."


def _classify_emotion(text: str, emphasis_score: float | None) -> str:
    """Infer a lightweight emotion label for stub outputs."""

    lowered = text.lower()
    if any(keyword in lowered for keyword in ("suddenly", "really", "must", "important", "finally")):
        return "emotional"
    if (emphasis_score or 0.0) >= 0.75:
        return "intense"
    return "warm"


def _semantic_role_for_index(index: int, total_segments: int) -> str:
    """Choose a stable semantic role based on the candidate position in the story."""

    if index == 1:
        return "opening_hook"
    if index == total_segments:
        return "ending"
    return "core_point"


def _trim_policy_for_units(units: list[FineGrainedUnit]) -> TrimPolicy:
    """Build a trim policy from the precise transcript-backed units."""

    preferred_start = round(units[0].start, 2)
    preferred_end = round(units[-1].end, 2)
    full_duration = round(preferred_end - preferred_start, 2)

    if len(units) >= 3:
        preferred_start = round(units[1].start, 2)
        preferred_end = round(units[-2].end, 2)
    preferred_duration = round(preferred_end - preferred_start, 2)
    min_duration = round(max(3.0, min(preferred_duration, full_duration, 5.0)), 2)
    ideal_duration = round(max(min_duration, min(full_duration, max(6.0, preferred_duration))), 2)

    safe_cut_points: list[float] = []
    for unit in units:
        if unit.has_pause_before or unit.unit_id == units[0].unit_id:
            safe_cut_points.append(round(unit.start, 2))
        if unit.has_pause_after or unit.unit_id == units[-1].unit_id:
            safe_cut_points.append(round(unit.end, 2))

    unique_cut_points = sorted({point for point in safe_cut_points if preferred_start <= point <= preferred_end})
    if preferred_start not in unique_cut_points:
        unique_cut_points.insert(0, preferred_start)
    if preferred_end not in unique_cut_points:
        unique_cut_points.append(preferred_end)

    return TrimPolicy(
        trimmable=len(units) > 1,
        min_duration=min_duration,
        ideal_duration=ideal_duration,
        max_duration=full_duration,
        preferred_start=preferred_start,
        preferred_end=preferred_end,
        safe_cut_points=unique_cut_points,
    )


def _group_fine_grained_units(fine_grained_units: list[FineGrainedUnit]) -> list[list[FineGrainedUnit]]:
    """Group adjacent fine-grained units into coarse semantic chunks."""

    if not fine_grained_units:
        return []

    groups: list[list[FineGrainedUnit]] = []
    current_group: list[FineGrainedUnit] = []
    current_duration = 0.0

    for unit in fine_grained_units:
        if not current_group:
            current_group = [unit]
            current_duration = unit.duration
            continue

        projected_duration = round(current_duration + unit.duration, 2)
        should_split = unit.has_pause_before or projected_duration >= 15.0 or len(current_group) >= 3
        if should_split:
            groups.append(current_group)
            current_group = [unit]
            current_duration = unit.duration
            continue

        current_group.append(unit)
        current_duration = projected_duration

    if current_group:
        groups.append(current_group)
    return groups


def _build_stub_timeline_segments(fine_grained_units: list[FineGrainedUnit]) -> list[TimelineSegment]:
    """Build coarse semantic timeline segments from fine-grained transcript-backed units."""

    grouped_units = _group_fine_grained_units(fine_grained_units)
    timeline_segments: list[TimelineSegment] = []
    total_segments = max(1, len(grouped_units))

    for index, units in enumerate(grouped_units, start=1):
        start = round(units[0].start, 2)
        end = round(units[-1].end, 2)
        text = " ".join(unit.text.strip() for unit in units if unit.text.strip())
        excerpt = _clip_text(text)
        highlight_score = round(max(0.45, 0.95 - ((index - 1) * 0.08)), 2)
        hook_score = round(0.9 if index == 1 else max(0.35, highlight_score - 0.12), 2)
        timeline_segments.append(
            TimelineSegment(
                segment_id=f"timeline_{index:02d}",
                start=start,
                end=end,
                duration=round(end - start, 2),
                title=f"Scene {index}",
                summary=excerpt,
                transcript_excerpt=excerpt,
                source_unit_ids=[unit.unit_id for unit in units],
                importance=highlight_score,
                hook_score=hook_score,
                highlight_score=highlight_score,
                emotion=_classify_emotion(text, units[0].emphasis_score),
                tags=["transcript_stub", _semantic_role_for_index(index, total_segments)],
            )
        )
    return timeline_segments


def _build_stub_highlight_candidates(timeline_segments: list[TimelineSegment], units_by_id: dict[str, FineGrainedUnit]) -> list[LLMHighlightCandidate]:
    """Build fine-grained highlight candidates aligned to precise unit boundaries."""

    highlight_candidates: list[LLMHighlightCandidate] = []
    total_segments = max(1, len(timeline_segments))

    for index, segment in enumerate(timeline_segments, start=1):
        units = [units_by_id[unit_id] for unit_id in segment.source_unit_ids if unit_id in units_by_id]
        if not units:
            continue

        semantic_role = _semantic_role_for_index(index, total_segments)
        trim_policy = _trim_policy_for_units(units)
        highlight_type = "emotion_peak" if semantic_role == "opening_hook" else "quote" if semantic_role == "ending" else "core_point"
        overall_score = segment.highlight_score
        clarity_score = round(min(0.96, 0.55 + (len(segment.transcript_excerpt) / 220.0)), 2)
        editability_score = round(0.92 if trim_policy.trimmable else 0.7, 2)

        highlight_candidates.append(
            LLMHighlightCandidate(
                candidate_id=f"hl_{index:03d}",
                source_segment_ids=[segment.segment_id],
                transcript_unit_ids=[unit.unit_id for unit in units],
                source_start=segment.start,
                source_end=segment.end,
                duration=segment.duration,
                title=segment.title,
                summary=segment.summary,
                transcript_excerpt=segment.transcript_excerpt,
                highlight_type=highlight_type,
                semantic_role=semantic_role,
                reason=(
                    "Stubbed candidate derived from transcript-backed fine-grained units and semantic grouping. "
                    "Replace this with live Qwen reasoning once the API call is wired."
                ),
                scores=HighlightScoreBreakdown(
                    overall=overall_score,
                    hook=segment.hook_score,
                    emotion=round(min(1.0, overall_score + 0.05), 2),
                    clarity=clarity_score,
                    platform_fit=round(min(0.95, overall_score + 0.02), 2),
                    editability=editability_score,
                ),
                must_keep=semantic_role == "opening_hook",
                trim_policy=trim_policy,
                redundancy_group=highlight_type,
                transition_hint="hard_cut" if semantic_role == "opening_hook" else "fade_or_soft_cut" if semantic_role == "ending" else "straight_cut",
                subtitle_priority="high" if semantic_role in {"opening_hook", "ending"} else "medium",
            )
        )

    return highlight_candidates


def analyze_video_understanding(
    video_path: Path,
    user_request: UserRequest,
    video_info: VideoInfo,
    transcript: TranscriptResult,
    fine_grained_units: list[FineGrainedUnit],
) -> tuple[ContentAnalysis, VideoTimeline, list[LLMHighlightCandidate]]:
    """Return transcript-backed understanding outputs and preserve the future Qwen request contract."""

    request_payload = build_qwen_video_understanding_request(
        video_path=video_path,
        user_request=user_request,
        transcript=transcript,
        video_info=video_info,
        fine_grained_units=fine_grained_units,
    )
    timeline_segments = _build_stub_timeline_segments(fine_grained_units)
    units_by_id = {unit.unit_id: unit for unit in fine_grained_units}
    highlight_candidates = _build_stub_highlight_candidates(timeline_segments, units_by_id)
    narrative_flow = [segment.summary for segment in timeline_segments[:5]]
    main_topics = sorted(
        {
            user_request.target_platform,
            user_request.edit_style,
            user_request.language,
            *(keyword for unit in fine_grained_units[:6] for keyword in unit.keywords[:2]),
        }
    )
    hook_candidates = [candidate.transcript_excerpt for candidate in highlight_candidates[:3]]

    content_analysis = ContentAnalysis(
        summary=(
            "Stubbed video understanding generated from fine-grained transcript units while the Qwen video model API "
            "integration is being prepared."
        ),
        narrative_flow=narrative_flow,
        main_topics=main_topics,
        hook_candidates=hook_candidates,
        tone=user_request.edit_style,
        recommended_structure=["opening_hook", "core_point", "ending"],
        pacing="medium",
    )
    timeline = VideoTimeline(
        provider=request_payload["provider"],
        model=request_payload["model"],
        overview=(
            "Transcript-backed timeline generated from fine-grained editable units through the prepared Qwen "
            "interface contract."
        ),
        segments=timeline_segments,
        generation_mode="stub",
        warnings=[
            "Qwen video understanding SDK/API call is not wired yet. Current output is transcript-backed stub data."
        ],
    )
    return content_analysis, timeline, highlight_candidates
