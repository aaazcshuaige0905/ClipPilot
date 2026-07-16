import json
import os
from pathlib import Path
import re
from math import ceil
from typing import Any

from clippilot.core.exceptions import ClipPilotProcessingError
from clippilot.llm.providers import MultimodalProviderError, OpenAICompatibleMultimodalProvider
from clippilot.llm.serializers import build_multimodal_frame_messages, build_multimodal_video_messages
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
from clippilot.schemas.video_frames import SampledFramesResult
from clippilot.schemas.video_info import VideoInfo
from clippilot.schemas.video_understanding import (
    CandidateCluster,
    CandidateRefinementGroup,
    CoarseHighlightCandidate,
    GlobalUnderstandingResult,
    RefinedCandidateOption,
    RefinedHighlightCandidate,
    VideoWindow,
    WindowUnderstandingResult,
    WindowVisualEvent,
)
from clippilot.storage.path_manager import AppSettings
from clippilot.tools.video_segmenter import _render_proxy_window

DEFAULT_QWEN_VIDEO_MODEL = "qwen3.7-plus"
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class _StageRepairError(ClipPilotProcessingError):
    """Carry both provider responses when a stage and its one repair attempt are invalid."""

    def __init__(self, cause: Exception, primary_response: dict[str, Any], repair_response: dict[str, Any]) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.primary_response = primary_response
        self.repair_response = repair_response

VIDEO_UNDERSTANDING_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["content_analysis", "timeline_segments", "highlight_candidates"],
    "properties": {
        "content_analysis": {
            "oneOf": [
                {"type": "string"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "summary",
                        "narrative_flow",
                        "main_topics",
                        "hook_candidates",
                        "tone",
                        "recommended_structure",
                        "pacing",
                    ],
                    "properties": {
                        "summary": {"type": "string"},
                        "narrative_flow": {"type": "array", "items": {"type": "string"}},
                        "main_topics": {"type": "array", "items": {"type": "string"}},
                        "hook_candidates": {"type": "array", "items": {"type": "string"}},
                        "tone": {"type": "string"},
                        "recommended_structure": {"type": "array", "items": {"type": "string"}},
                        "pacing": {"type": "string"},
                    },
                },
            ]
        },
        "timeline_segments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "segment_id",
                    "start",
                    "end",
                    "title",
                    "summary",
                    "transcript_excerpt",
                    "source_unit_ids",
                    "importance",
                    "hook_score",
                    "highlight_score",
                    "emotion",
                    "tags",
                ],
                "properties": {
                    "segment_id": {"type": "string"},
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "transcript_excerpt": {"type": "string"},
                    "source_unit_ids": {"type": "array", "items": {"type": "string"}},
                    "importance": {"type": "number"},
                    "hook_score": {"type": "number"},
                    "highlight_score": {"type": "number"},
                    "emotion": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "highlight_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "candidate_id",
                    "source_segment_ids",
                    "transcript_unit_ids",
                    "source_start",
                    "source_end",
                    "title",
                    "summary",
                    "transcript_excerpt",
                    "highlight_type",
                    "semantic_role",
                    "reason",
                    "scores",
                    "must_keep",
                    "trim_policy",
                    "dependencies",
                    "redundancy_group",
                    "transition_hint",
                    "subtitle_priority",
                    "risk_flags",
                ],
                "properties": {
                    "candidate_id": {"type": "string"},
                    "source_segment_ids": {"type": "array", "items": {"type": "string"}},
                    "transcript_unit_ids": {"type": "array", "items": {"type": "string"}},
                    "source_start": {"type": "number"},
                    "source_end": {"type": "number"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "transcript_excerpt": {"type": "string"},
                    "highlight_type": {"type": "string"},
                    "semantic_role": {"type": "string"},
                    "reason": {"type": "string"},
                    "scores": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["overall", "hook", "emotion", "clarity", "platform_fit", "editability"],
                        "properties": {
                            "overall": {"type": "number"},
                            "hook": {"type": "number"},
                            "emotion": {"type": "number"},
                            "clarity": {"type": "number"},
                            "platform_fit": {"type": "number"},
                            "editability": {"type": "number"},
                        },
                    },
                    "must_keep": {"type": "boolean"},
                    "trim_policy": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "trimmable",
                            "min_duration",
                            "ideal_duration",
                            "max_duration",
                            "preferred_start",
                            "preferred_end",
                            "safe_cut_points",
                        ],
                        "properties": {
                            "trimmable": {"type": "boolean"},
                            "min_duration": {"type": "number"},
                            "ideal_duration": {"type": "number"},
                            "max_duration": {"type": "number"},
                            "preferred_start": {"type": "number"},
                            "preferred_end": {"type": "number"},
                            "safe_cut_points": {"type": "array", "items": {"type": "number"}},
                        },
                    },
                    "dependencies": {"type": "array", "items": {"type": "string"}},
                    "redundancy_group": {"type": ["string", "null"]},
                    "transition_hint": {"type": "string"},
                    "subtitle_priority": {"type": "string"},
                    "risk_flags": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}


def _video_understanding_response_format() -> dict[str, Any]:
    """Use the JSON mode that the configured DashScope-compatible endpoint documents."""

    return {"type": "json_object"}


def _video_understanding_contract_instructions() -> str:
    """Return a compact human-readable contract reminder for the model prompt."""

    return (
        "Return JSON only. Do not wrap in markdown. Do not return an empty array for the stage's primary output. "
        "Do not rename fields. Use empty arrays instead of null for optional list fields. "
        "The application validates the result locally against this JSON Schema: "
        f"{json.dumps(VIDEO_UNDERSTANDING_RESPONSE_SCHEMA, ensure_ascii=False)}"
    )


def build_qwen_video_understanding_request(
    video_path: Path,
    user_request: UserRequest,
    transcript: TranscriptResult,
    video_info: VideoInfo,
    fine_grained_units: list[FineGrainedUnit],
    sampled_frames: SampledFramesResult | None = None,
    settings: AppSettings | None = None,
) -> dict:
    """Build the future OpenAI-compatible request payload for Qwen video understanding."""

    model_name = (
        settings.video_understanding_model
        if settings is not None
        else os.getenv("CLIP_PILOT_QWEN_VIDEO_MODEL", DEFAULT_QWEN_VIDEO_MODEL).strip() or DEFAULT_QWEN_VIDEO_MODEL
    )
    base_url = (
        settings.video_understanding_base_url
        if settings is not None
        else os.getenv("CLIP_PILOT_QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL).strip() or DEFAULT_QWEN_BASE_URL
    )
    api_path = settings.video_understanding_api_path if settings is not None else "/chat/completions"
    timeout_seconds = settings.video_understanding_timeout_seconds if settings is not None else 180

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
    sampled_frame_preview = [
        {
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "image_path": frame.image_path,
            "width": frame.width,
            "height": frame.height,
            "note": frame.note,
        }
        for frame in (sampled_frames.frames if sampled_frames is not None else [])
    ]
    return {
        "provider": "qwen",
        "base_url": base_url,
        "api_path": api_path,
        "model": model_name,
        "timeout_seconds": timeout_seconds,
        "stage": "timeline",
        "max_tokens": (
            settings.video_understanding_timeline_max_tokens if settings is not None else 7000
        ),
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
                        "Return JSON containing content_analysis and timeline_segments. This is stage 1 of 2: "
                        "focus only on factual visual/transcript understanding and a coherent semantic timeline. "
                        "Return 3-12 timeline segments when the video evidence permits. The "
                        "highlight_candidates field may be an empty array in this stage."
                    ),
                    "output_contract": _video_understanding_contract_instructions(),
                    "target_platform": user_request.target_platform,
                    "target_duration": user_request.target_duration,
                    "edit_style": user_request.edit_style,
                    "language": user_request.language,
                    "video_path": str(video_path),
                    "transcript_preview": transcript_preview,
                    "fine_grained_units": fine_grained_preview,
                    "sampled_frames": sampled_frame_preview,
                },
            },
        ],
        "response_format": _video_understanding_response_format(),
    }


def _build_highlight_stage_request(
    base_request: dict[str, Any],
    content_analysis: ContentAnalysis,
    timeline_segments: list[TimelineSegment],
    settings: AppSettings,
) -> dict[str, Any]:
    """Build stage 2 while retaining every sampled frame from the base multimodal request."""

    request_payload = dict(base_request)
    messages = [dict(message) for message in base_request["messages"]]
    user_content = dict(messages[1]["content"])
    user_content.update(
        {
            "instruction": (
                "Return JSON containing highlight_candidates selected from the validated stage-1 timeline. "
                "Re-check every attached frame before scoring. Return 3-8 candidates when the evidence permits. "
                "Each candidate must reference existing source_segment_ids and include source_start, source_end, "
                "scores, and a complete trim_policy. content_analysis and timeline_segments may repeat the supplied "
                "validated values, but highlight_candidates must not be empty."
            ),
            "validated_content_analysis": content_analysis.model_dump(),
            "validated_timeline_segments": [segment.model_dump() for segment in timeline_segments],
        }
    )
    messages[0] = {
        "role": "system",
        "content": (
            "You are stage 2 of a video-understanding pipeline. Re-check the complete visual evidence and use the "
            "validated semantic timeline to select precise, editable highlights. Return JSON only."
        ),
    }
    messages[1] = {"role": "user", "content": user_content}
    request_payload.update(
        {
            "stage": "highlights",
            "max_tokens": settings.video_understanding_highlight_max_tokens,
            "messages": messages,
        }
    )
    return request_payload


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


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp one numeric value into a closed interval."""

    return max(lower, min(value, upper))


def _build_trim_policy(
    *,
    source_start: float,
    source_end: float,
    preferred_start: float | None = None,
    preferred_end: float | None = None,
    safe_cut_points: list[float] | None = None,
    requested_min_duration: float | None = None,
    requested_ideal_duration: float | None = None,
    requested_max_duration: float | None = None,
    trimmable: bool = True,
) -> TrimPolicy:
    """Build one internally consistent trim policy, clamping short-window edge cases safely."""

    normalized_source_start = round(float(source_start), 2)
    normalized_source_end = round(float(source_end), 2)
    full_duration = round(normalized_source_end - normalized_source_start, 2)
    if full_duration <= 0:
        raise ClipPilotProcessingError("Trim policy source window must have positive duration.")

    normalized_preferred_start = round(
        _clamp(
            float(preferred_start if preferred_start is not None else normalized_source_start),
            normalized_source_start,
            normalized_source_end,
        ),
        2,
    )
    normalized_preferred_end = round(
        _clamp(
            float(preferred_end if preferred_end is not None else normalized_source_end),
            normalized_preferred_start,
            normalized_source_end,
        ),
        2,
    )
    preferred_duration = round(normalized_preferred_end - normalized_preferred_start, 2)
    if preferred_duration <= 0:
        normalized_preferred_start = normalized_source_start
        normalized_preferred_end = normalized_source_end
        preferred_duration = full_duration

    normalized_safe_cut_points = sorted(
        {
            round(float(point), 2)
            for point in (safe_cut_points or [])
            if normalized_source_start <= round(float(point), 2) <= normalized_source_end
        }
    )
    for boundary in (
        normalized_source_start,
        normalized_preferred_start,
        normalized_preferred_end,
        normalized_source_end,
    ):
        if boundary not in normalized_safe_cut_points:
            normalized_safe_cut_points.append(boundary)
    normalized_safe_cut_points = sorted(set(normalized_safe_cut_points))

    short_window = full_duration <= 3.0 or preferred_duration <= 3.0
    if short_window or not trimmable:
        return TrimPolicy(
            trimmable=False,
            min_duration=full_duration,
            ideal_duration=full_duration,
            max_duration=full_duration,
            preferred_start=normalized_source_start,
            preferred_end=normalized_source_end,
            safe_cut_points=[normalized_source_start, normalized_source_end],
        )

    max_duration = round(
        _clamp(
            float(requested_max_duration if requested_max_duration is not None else full_duration),
            0.01,
            full_duration,
        ),
        2,
    )

    min_target = float(requested_min_duration if requested_min_duration is not None else min(preferred_duration, full_duration, 5.0))
    min_duration = round(_clamp(min_target, 0.01, max_duration), 2)

    if preferred_duration < min_duration:
        normalized_preferred_start = normalized_source_start
        normalized_preferred_end = normalized_source_end
        preferred_duration = full_duration

    ideal_target = float(
        requested_ideal_duration
        if requested_ideal_duration is not None
        else min(full_duration, max(6.0, preferred_duration))
    )
    ideal_duration = round(_clamp(ideal_target, min_duration, max_duration), 2)

    if preferred_duration < min_duration:
        min_duration = preferred_duration
        ideal_duration = round(_clamp(ideal_duration, min_duration, max_duration), 2)

    return TrimPolicy(
        trimmable=True,
        min_duration=min_duration,
        ideal_duration=ideal_duration,
        max_duration=max_duration,
        preferred_start=normalized_preferred_start,
        preferred_end=normalized_preferred_end,
        safe_cut_points=normalized_safe_cut_points,
    )


def _trim_policy_for_units(units: list[FineGrainedUnit]) -> TrimPolicy:
    """Build a trim policy from the precise transcript-backed units."""

    source_start = round(units[0].start, 2)
    source_end = round(units[-1].end, 2)
    preferred_start = source_start
    preferred_end = source_end

    if len(units) >= 3:
        preferred_start = round(units[1].start, 2)
        preferred_end = round(units[-2].end, 2)

    safe_cut_points: list[float] = []
    for unit in units:
        if unit.has_pause_before or unit.unit_id == units[0].unit_id:
            safe_cut_points.append(round(unit.start, 2))
        if unit.has_pause_after or unit.unit_id == units[-1].unit_id:
            safe_cut_points.append(round(unit.end, 2))

    return _build_trim_policy(
        source_start=source_start,
        source_end=source_end,
        preferred_start=preferred_start,
        preferred_end=preferred_end,
        safe_cut_points=safe_cut_points,
        requested_min_duration=5.0,
        requested_ideal_duration=max(6.0, round(preferred_end - preferred_start, 2)),
        requested_max_duration=round(source_end - source_start, 2),
        trimmable=len(units) > 1,
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


def _frame_stub_window_duration(video_duration: float, segment_count: int) -> float:
    """Pick a compact per-candidate duration for frame-only visual fallback planning."""

    if video_duration <= 0:
        return 4.0
    return round(min(10.0, max(4.0, video_duration / max(segment_count * 4, 1))), 2)


def _build_frame_stub_timeline_segments(
    sampled_frames: SampledFramesResult | None,
    video_info: VideoInfo,
) -> list[TimelineSegment]:
    """Build coarse visual-only segments when transcript-backed units are unavailable."""

    duration_seconds = max(0.0, round(video_info.duration_seconds, 2))
    if duration_seconds <= 0:
        return []

    frames = list(sampled_frames.frames) if sampled_frames is not None else []
    if not frames:
        end = round(min(duration_seconds, 6.0), 2)
        return [
            TimelineSegment(
                segment_id="timeline_01",
                start=0.0,
                end=end,
                duration=end,
                title="Visual opening",
                summary="Fallback visual segment created without transcript or sampled-frame coverage.",
                transcript_excerpt="Visual-only fallback segment.",
                source_unit_ids=[],
                importance=0.6,
                hook_score=0.72,
                highlight_score=0.68,
                emotion="neutral",
                tags=["visual_stub", "opening_hook"],
            )
        ]

    segment_count = min(4, max(3, ceil(len(frames) / 10)))
    window_duration = _frame_stub_window_duration(duration_seconds, segment_count)
    last_start = max(0.0, round(duration_seconds - window_duration, 2))
    sampled_indices = sorted({round(index * (len(frames) - 1) / max(segment_count - 1, 1)) for index in range(segment_count)})

    timeline_segments: list[TimelineSegment] = []
    for order, raw_index in enumerate(sampled_indices, start=1):
        frame = frames[int(raw_index)]
        start = round(min(max(0.0, frame.timestamp - (window_duration / 2.0)), last_start), 2)
        end = round(min(duration_seconds, start + window_duration), 2)
        role = _semantic_role_for_index(order, len(sampled_indices))
        summary = f"Visual moment sampled near {frame.timestamp:.2f}s with no transcript available."
        timeline_segments.append(
            TimelineSegment(
                segment_id=f"timeline_{order:02d}",
                start=start,
                end=end,
                duration=round(end - start, 2),
                title=f"Visual scene {order}",
                summary=summary,
                transcript_excerpt=summary,
                source_unit_ids=[],
                importance=round(max(0.5, 0.86 - ((order - 1) * 0.08)), 2),
                hook_score=round(0.84 if order == 1 else 0.46, 2),
                highlight_score=round(max(0.5, 0.86 - ((order - 1) * 0.08)), 2),
                emotion="visual_only",
                tags=["visual_stub", role],
            )
        )
    return timeline_segments


def _build_frame_stub_highlight_candidates(timeline_segments: list[TimelineSegment]) -> list[LLMHighlightCandidate]:
    """Build visual-only highlight candidates from sampled-frame windows."""

    highlight_candidates: list[LLMHighlightCandidate] = []
    total_segments = max(1, len(timeline_segments))

    for index, segment in enumerate(timeline_segments, start=1):
        semantic_role = _semantic_role_for_index(index, total_segments)
        duration = round(segment.end - segment.start, 2)
        highlight_candidates.append(
            LLMHighlightCandidate(
                candidate_id=f"hl_visual_{index:03d}",
                source_segment_ids=[segment.segment_id],
                transcript_unit_ids=[],
                source_start=segment.start,
                source_end=segment.end,
                duration=duration,
                title=segment.title,
                summary=segment.summary,
                transcript_excerpt=segment.transcript_excerpt,
                highlight_type="visual_moment",
                semantic_role=semantic_role,
                reason="Visual-only fallback candidate created because transcript content was unavailable.",
                scores=HighlightScoreBreakdown(
                    overall=segment.highlight_score,
                    hook=segment.hook_score,
                    emotion=round(min(1.0, segment.highlight_score + 0.04), 2),
                    clarity=0.58,
                    platform_fit=round(min(0.92, segment.highlight_score + 0.03), 2),
                    editability=0.82,
                ),
                must_keep=semantic_role == "opening_hook",
                trim_policy=_build_trim_policy(
                    source_start=segment.start,
                    source_end=segment.end,
                    preferred_start=segment.start,
                    preferred_end=segment.end,
                    safe_cut_points=[segment.start, segment.end],
                    requested_min_duration=4.0,
                    requested_ideal_duration=min(duration, 8.0),
                    requested_max_duration=duration,
                    trimmable=True,
                ),
                redundancy_group="visual_stub",
                transition_hint="hard_cut" if semantic_role == "opening_hook" else "straight_cut",
                subtitle_priority="low",
            )
        )

    return highlight_candidates


def _build_stub_response_payload(
    content_analysis: ContentAnalysis,
    timeline: VideoTimeline,
    highlight_candidates: list[LLMHighlightCandidate],
) -> dict[str, Any]:
    """Build a normalized output payload that matches the shared understanding contract."""

    return {
        "content_analysis": content_analysis.model_dump(),
        "timeline_segments": [segment.model_dump() for segment in timeline.segments],
        "highlight_candidates": [candidate.model_dump() for candidate in highlight_candidates],
    }


def _build_video_understanding_artifact_payload(
    *,
    request_payload: dict[str, Any],
    content_analysis: ContentAnalysis,
    timeline: VideoTimeline,
    highlight_candidates: list[LLMHighlightCandidate],
    attempted_live_call: bool,
    used_fallback: bool,
    provider_response: dict[str, Any] | None,
    phase_responses: dict[str, Any] | None = None,
    fallback_stage: str | None = None,
    failure: Exception | None = None,
    skip_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Build one unified persisted payload for both live and fallback understanding paths."""

    warnings = list(timeline.warnings)
    payload: dict[str, Any] = {
        "provider": request_payload["provider"],
        "model": request_payload["model"],
        "generation_mode": timeline.generation_mode,
        "attempted_live_call": attempted_live_call,
        "used_fallback": used_fallback,
        "warnings": warnings,
        "skip_reasons": list(skip_reasons or []),
        "failure": None,
        "provider_response": provider_response,
        "phase_responses": phase_responses or {},
        "fallback_stage": fallback_stage,
        "normalized_output": _build_stub_response_payload(content_analysis, timeline, highlight_candidates),
    }
    if failure is not None:
        payload["failure"] = {
            "type": type(failure).__name__,
            "message": str(failure),
        }
    return payload


def _live_video_understanding_skip_reasons(
    sampled_frames: SampledFramesResult | None,
    settings: AppSettings | None,
) -> list[str]:
    """Explain why the live multimodal branch was skipped instead of attempted."""

    reasons: list[str] = []
    if settings is None:
        reasons.append("settings_missing")
        return reasons
    if not settings.video_understanding_enabled:
        reasons.append("video_understanding_disabled")
    if not settings.video_understanding_api_key:
        reasons.append("video_understanding_api_key_missing")
    if sampled_frames is None:
        reasons.append("sampled_frames_missing")
    elif not sampled_frames.frames:
        reasons.append("sampled_frames_empty")
    return reasons


def _skip_warning_for_live_video_understanding(skip_reasons: list[str]) -> str:
    """Build a readable skip warning for fallback outputs when live reasoning was not attempted."""

    if not skip_reasons:
        return "Live multimodal video understanding was skipped for an unspecified reason."
    return "Live multimodal video understanding was skipped: " + ", ".join(skip_reasons) + "."


def _extract_response_text(raw_response: dict[str, Any]) -> str:
    """Extract the primary assistant text payload from one OpenAI-compatible response body."""

    choices = raw_response.get("choices") or []
    if not choices:
        raise ClipPilotProcessingError("Video-understanding response did not contain any choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        combined = "\n".join(part for part in text_parts if part.strip())
        if combined.strip():
            return combined
    raise ClipPilotProcessingError("Video-understanding response did not contain readable assistant content.")


def _extract_json_payload(text: str) -> dict[str, Any]:
    """Parse a JSON object from raw model text, supporting fenced-code output."""

    stripped = text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.DOTALL)
        if match:
            stripped = match.group(1).strip()
    return json.loads(stripped)


def _coerce_content_analysis(payload: dict[str, Any], fallback_tone: str) -> ContentAnalysis:
    """Normalize model output into the shared ContentAnalysis schema."""

    raw_content_payload = payload.get("content_analysis") or {}
    if isinstance(raw_content_payload, str):
        content_payload: dict[str, Any] = {"summary": raw_content_payload}
    elif isinstance(raw_content_payload, dict):
        content_payload = raw_content_payload
    else:
        raise ClipPilotProcessingError("Video-understanding content_analysis must be a JSON object or summary string.")
    return ContentAnalysis(
        summary=content_payload.get("summary") or "Model-generated visual understanding summary.",
        narrative_flow=content_payload.get("narrative_flow") or [],
        main_topics=content_payload.get("main_topics") or [],
        hook_candidates=content_payload.get("hook_candidates") or [],
        tone=content_payload.get("tone") or fallback_tone,
        recommended_structure=content_payload.get("recommended_structure") or ["opening_hook", "core_point", "ending"],
        pacing=content_payload.get("pacing") or "medium",
    )


def _coerce_timeline_segments(payload: dict[str, Any]) -> list[TimelineSegment]:
    """Normalize model timeline output into validated TimelineSegment models."""

    segments: list[TimelineSegment] = []
    for index, item in enumerate(payload.get("timeline_segments") or [], start=1):
        if not isinstance(item, dict):
            raise ClipPilotProcessingError("Each timeline segment returned by video understanding must be an object.")
        segments.append(
            TimelineSegment(
                segment_id=item.get("segment_id") or f"timeline_{index:02d}",
                start=round(float(item["start"]), 2),
                end=round(float(item["end"]), 2),
                duration=round(float(item["end"]) - float(item["start"]), 2),
                title=item.get("title") or f"Scene {index}",
                summary=item.get("summary") or item.get("transcript_excerpt") or f"Segment {index}",
                transcript_excerpt=item.get("transcript_excerpt") or item.get("summary") or f"Segment {index}",
                source_unit_ids=item.get("source_unit_ids") or [],
                importance=float(item.get("importance", 0.6)),
                hook_score=float(item.get("hook_score", 0.4)),
                highlight_score=float(item.get("highlight_score", item.get("importance", 0.6))),
                emotion=item.get("emotion") or "neutral",
                tags=item.get("tags") or [],
            )
        )
    if not segments:
        raise ClipPilotProcessingError("Video-understanding model response returned no usable timeline segments.")
    return segments


def _coerce_highlight_candidates(payload: dict[str, Any]) -> list[LLMHighlightCandidate]:
    """Normalize model candidate output into validated LLMHighlightCandidate models."""

    candidates: list[LLMHighlightCandidate] = []
    for index, item in enumerate(payload.get("highlight_candidates") or [], start=1):
        if not isinstance(item, dict):
            raise ClipPilotProcessingError("Each highlight candidate returned by video understanding must be an object.")
        scores_payload = item.get("scores") or {}
        trim_payload = item.get("trim_policy") or {}
        if not isinstance(scores_payload, dict):
            raise ClipPilotProcessingError("Highlight candidate scores must be a JSON object.")
        if not isinstance(trim_payload, dict):
            raise ClipPilotProcessingError("Highlight candidate trim_policy must be a JSON object.")
        source_start = item.get("source_start", item.get("start"))
        source_end = item.get("source_end", item.get("end"))
        if source_start is None or source_end is None:
            raise ClipPilotProcessingError(
                "Each highlight candidate returned by video understanding must include source_start/source_end."
            )
        candidates.append(
            LLMHighlightCandidate(
                candidate_id=item.get("candidate_id") or f"hl_{index:03d}",
                source_segment_ids=item.get("source_segment_ids") or [],
                transcript_unit_ids=item.get("transcript_unit_ids") or [],
                source_start=round(float(source_start), 2),
                source_end=round(float(source_end), 2),
                duration=round(float(source_end) - float(source_start), 2),
                title=item.get("title") or f"Highlight {index}",
                summary=item.get("summary") or item.get("transcript_excerpt") or f"Highlight {index}",
                transcript_excerpt=item.get("transcript_excerpt") or item.get("summary") or f"Highlight {index}",
                highlight_type=item.get("highlight_type") or "core_point",
                semantic_role=item.get("semantic_role") or "core_point",
                reason=item.get("reason") or "Model-selected highlight candidate.",
                scores=HighlightScoreBreakdown(
                    overall=float(scores_payload.get("overall", 0.65)),
                    hook=float(scores_payload.get("hook", 0.4)),
                    emotion=float(scores_payload.get("emotion", 0.5)),
                    clarity=float(scores_payload.get("clarity", 0.7)),
                    platform_fit=float(scores_payload.get("platform_fit", 0.7)),
                    editability=float(scores_payload.get("editability", 0.7)),
                ),
                must_keep=bool(item.get("must_keep", False)),
                trim_policy=_build_trim_policy(
                    source_start=float(source_start),
                    source_end=float(source_end),
                    preferred_start=float(trim_payload.get("preferred_start", source_start)),
                    preferred_end=float(trim_payload.get("preferred_end", source_end)),
                    safe_cut_points=trim_payload.get("safe_cut_points") or [],
                    requested_min_duration=float(trim_payload.get("min_duration", 3.0)),
                    requested_ideal_duration=float(trim_payload.get("ideal_duration", 5.0)),
                    requested_max_duration=float(trim_payload.get("max_duration", round(float(source_end) - float(source_start), 2))),
                    trimmable=bool(trim_payload.get("trimmable", True)),
                ),
                dependencies=item.get("dependencies") or [],
                redundancy_group=item.get("redundancy_group"),
                transition_hint=item.get("transition_hint") or "straight_cut",
                subtitle_priority=item.get("subtitle_priority") or "medium",
                risk_flags=item.get("risk_flags") or [],
            )
        )
    if not candidates:
        raise ClipPilotProcessingError("Video-understanding model response returned no usable highlight candidates.")
    return candidates


def _validate_highlight_references(
    candidates: list[LLMHighlightCandidate],
    timeline_segments: list[TimelineSegment],
) -> list[LLMHighlightCandidate]:
    """Reject stage-2 candidates that do not point back to the validated stage-1 timeline."""

    valid_segment_ids = {segment.segment_id for segment in timeline_segments}
    for candidate in candidates:
        unknown_ids = set(candidate.source_segment_ids) - valid_segment_ids
        if unknown_ids:
            raise ClipPilotProcessingError(
                f"Highlight candidate {candidate.candidate_id} references unknown timeline segments: "
                f"{', '.join(sorted(unknown_ids))}."
            )
        if not candidate.source_segment_ids:
            raise ClipPilotProcessingError(
                f"Highlight candidate {candidate.candidate_id} must reference at least one timeline segment."
            )
    return candidates


def _call_multimodal_provider(
    request_payload: dict[str, Any],
    sampled_frames: SampledFramesResult,
    settings: AppSettings,
) -> dict[str, Any]:
    """Serialize sampled frames and execute one multimodal provider request."""

    provider = OpenAICompatibleMultimodalProvider(
        api_key=settings.video_understanding_api_key,
        base_url=request_payload["base_url"],
        api_path=request_payload.get("api_path") or settings.video_understanding_api_path,
        timeout_seconds=int(request_payload.get("timeout_seconds") or settings.video_understanding_timeout_seconds),
    )
    multimodal_messages = build_multimodal_frame_messages(
        request_payload=request_payload,
        sampled_frames=sampled_frames,
        project_root=settings.project_root,
    )
    provider_payload = {
        "model": request_payload["model"],
        "messages": multimodal_messages,
        "temperature": 0.2,
        "max_tokens": int(request_payload.get("max_tokens") or settings.video_understanding_timeline_max_tokens),
        "enable_thinking": False,
        "response_format": request_payload.get("response_format") or _video_understanding_response_format(),
    }
    return provider.create_multimodal_completion(provider_payload)


def _build_repair_request(
    failed_request: dict[str, Any],
    failed_response: dict[str, Any],
    validation_error: Exception,
    settings: AppSettings,
) -> dict[str, Any]:
    """Build one bounded JSON-only repair call without resending image bytes."""

    stage = str(failed_request.get("stage") or "unknown")
    primary_field = "timeline_segments" if stage == "timeline" else "highlight_candidates"
    return {
        **failed_request,
        "stage": f"{stage}_repair",
        "max_tokens": settings.video_understanding_repair_max_tokens,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You repair malformed video-analysis JSON. Return JSON only, preserve useful facts, and satisfy "
                    "the supplied contract. Do not explain your changes."
                ),
            },
            {
                "role": "user",
                "content": {
                    "instruction": (
                        f"Repair the response for stage {stage}. `{primary_field}` must contain at least one usable "
                        "item. Keep timestamps and evidence grounded in the original response."
                    ),
                    "validation_error": str(validation_error),
                    "output_contract": _video_understanding_contract_instructions(),
                    "failed_response": failed_response,
                    "sampled_frames": [],
                },
            },
        ],
    }


def _call_stage_with_one_repair(
    request_payload: dict[str, Any],
    sampled_frames: SampledFramesResult,
    settings: AppSettings,
    parser,
) -> tuple[Any, dict[str, Any], dict[str, Any] | None]:
    """Call a stage and retry once with a compact JSON-repair prompt on validation failure."""

    response = _call_multimodal_provider(request_payload, sampled_frames, settings)
    try:
        return parser(response), response, None
    except (
        ClipPilotProcessingError,
        KeyError,
        ValueError,
        TypeError,
        AttributeError,
        json.JSONDecodeError,
    ) as exc:
        repair_request = _build_repair_request(request_payload, response, exc, settings)
        repair_response = _call_multimodal_provider(repair_request, sampled_frames, settings)
        try:
            return parser(repair_response), response, repair_response
        except (
            ClipPilotProcessingError,
            KeyError,
            ValueError,
            TypeError,
            AttributeError,
            json.JSONDecodeError,
        ) as repair_exc:
            raise _StageRepairError(repair_exc, response, repair_response) from repair_exc


def _should_use_live_video_understanding(
    sampled_frames: SampledFramesResult | None,
    settings: AppSettings | None,
) -> bool:
    """Return whether the current task has enough configuration to use the live multimodal branch."""

    return bool(
        settings is not None
        and settings.video_understanding_enabled
        and settings.video_understanding_api_key
        and sampled_frames is not None
        and sampled_frames.frames
    )


def analyze_video_understanding(
    video_path: Path,
    user_request: UserRequest,
    video_info: VideoInfo,
    transcript: TranscriptResult,
    fine_grained_units: list[FineGrainedUnit],
    sampled_frames: SampledFramesResult | None = None,
    settings: AppSettings | None = None,
) -> tuple[ContentAnalysis, VideoTimeline, list[LLMHighlightCandidate], dict, dict]:
    """Return transcript-backed understanding outputs and preserve the future Qwen request contract."""

    request_payload = build_qwen_video_understanding_request(
        video_path=video_path,
        user_request=user_request,
        transcript=transcript,
        video_info=video_info,
        fine_grained_units=fine_grained_units,
        sampled_frames=sampled_frames,
        settings=settings,
    )

    llm_warning: str | None = None
    llm_failure: Exception | None = None
    provider_response: dict[str, Any] | None = None
    phase_responses: dict[str, Any] = {}
    fallback_stage: str | None = None
    attempted_live_call = False
    skip_reasons = _live_video_understanding_skip_reasons(sampled_frames, settings)
    if _should_use_live_video_understanding(sampled_frames, settings):
        attempted_live_call = True
        try:
            def _parse_timeline(response: dict[str, Any]) -> tuple[ContentAnalysis, list[TimelineSegment]]:
                payload = _extract_json_payload(_extract_response_text(response))
                return (
                    _coerce_content_analysis(payload, user_request.edit_style),
                    _coerce_timeline_segments(payload),
                )

            (content_analysis, timeline_segments), timeline_response, timeline_repair_response = (
                _call_stage_with_one_repair(request_payload, sampled_frames, settings, _parse_timeline)
            )
            phase_responses["timeline"] = timeline_response
            if timeline_repair_response is not None:
                phase_responses["timeline_repair"] = timeline_repair_response

            highlight_request = _build_highlight_stage_request(
                request_payload,
                content_analysis,
                timeline_segments,
                settings,
            )
            stage_failure: Exception | None = None

            def _parse_highlights(response: dict[str, Any]) -> list[LLMHighlightCandidate]:
                payload = _extract_json_payload(_extract_response_text(response))
                return _validate_highlight_references(
                    _coerce_highlight_candidates(payload),
                    timeline_segments,
                )

            try:
                highlight_candidates, highlight_response, highlight_repair_response = _call_stage_with_one_repair(
                    highlight_request,
                    sampled_frames,
                    settings,
                    _parse_highlights,
                )
                provider_response = highlight_repair_response or highlight_response
                phase_responses["highlights"] = highlight_response
                if highlight_repair_response is not None:
                    phase_responses["highlights_repair"] = highlight_repair_response
                used_fallback = False
                warnings: list[str] = []
                generation_mode = "llm_frames_two_stage_v1"
            except (
                ClipPilotProcessingError,
                MultimodalProviderError,
                KeyError,
                ValueError,
                TypeError,
                AttributeError,
                json.JSONDecodeError,
            ) as highlight_exc:
                fallback_stage = "highlights"
                if isinstance(highlight_exc, _StageRepairError):
                    phase_responses["highlights"] = highlight_exc.primary_response
                    phase_responses["highlights_repair"] = highlight_exc.repair_response
                    provider_response = highlight_exc.repair_response
                    highlight_failure = highlight_exc.cause
                else:
                    provider_response = phase_responses.get("timeline_repair") or timeline_response
                    highlight_failure = highlight_exc
                stage_failure = highlight_failure
                units_by_id = {unit.unit_id: unit for unit in fine_grained_units}
                highlight_candidates = _build_stub_highlight_candidates(timeline_segments, units_by_id)
                if not highlight_candidates:
                    highlight_candidates = _build_frame_stub_highlight_candidates(timeline_segments)
                used_fallback = True
                warnings = [
                    "Live timeline succeeded, but highlight generation failed; candidates were derived locally: "
                    f"{highlight_failure}"
                ]
                generation_mode = "llm_timeline_local_candidates_v1"

            timeline = VideoTimeline(
                provider=request_payload["provider"],
                model=request_payload["model"],
                overview=content_analysis.summary,
                segments=timeline_segments,
                generation_mode=generation_mode,
                warnings=warnings,
            )
            raw_response = _build_video_understanding_artifact_payload(
                request_payload=request_payload,
                content_analysis=content_analysis,
                timeline=timeline,
                highlight_candidates=highlight_candidates,
                attempted_live_call=True,
                used_fallback=used_fallback,
                provider_response=provider_response,
                phase_responses=phase_responses,
                fallback_stage=fallback_stage,
                failure=stage_failure,
            )
            return content_analysis, timeline, highlight_candidates, request_payload, raw_response
        except (
            ClipPilotProcessingError,
            MultimodalProviderError,
            KeyError,
            ValueError,
            TypeError,
            AttributeError,
            json.JSONDecodeError,
        ) as exc:
            if isinstance(exc, _StageRepairError):
                provider_response = exc.repair_response
                phase_responses["timeline"] = exc.primary_response
                phase_responses["timeline_repair"] = exc.repair_response
                llm_failure = exc.cause
            else:
                llm_failure = exc
            fallback_stage = "timeline"
            llm_warning = (
                "Live multimodal video understanding failed and fell back to transcript-backed stub data: "
                f"{llm_failure}"
            )

    if fine_grained_units:
        timeline_segments = _build_stub_timeline_segments(fine_grained_units)
        units_by_id = {unit.unit_id: unit for unit in fine_grained_units}
        highlight_candidates = _build_stub_highlight_candidates(timeline_segments, units_by_id)
        fallback_summary = (
            "Stubbed video understanding generated from fine-grained transcript units while the Qwen video model API "
            "integration is being prepared."
        )
        fallback_overview = (
            "Transcript-backed timeline generated from fine-grained editable units through the prepared Qwen "
            "interface contract."
        )
        fallback_generation_mode = "stub_transcript_units_v1"
    else:
        timeline_segments = _build_frame_stub_timeline_segments(sampled_frames, video_info)
        highlight_candidates = _build_frame_stub_highlight_candidates(timeline_segments)
        fallback_summary = (
            "Stubbed visual-only understanding generated from sampled frames because no usable transcript content was available."
        )
        fallback_overview = "Frame-backed timeline generated without transcript support."
        fallback_generation_mode = "stub_visual_frames_v1"
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
        summary=fallback_summary,
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
        overview=fallback_overview,
        segments=timeline_segments,
        generation_mode=fallback_generation_mode,
        warnings=[
            llm_warning
            or _skip_warning_for_live_video_understanding(skip_reasons)
        ],
    )
    raw_response = _build_video_understanding_artifact_payload(
        request_payload=request_payload,
        content_analysis=content_analysis,
        timeline=timeline,
        highlight_candidates=highlight_candidates,
        attempted_live_call=attempted_live_call,
        used_fallback=True,
        provider_response=provider_response,
        phase_responses=phase_responses,
        fallback_stage=fallback_stage,
        failure=llm_failure,
        skip_reasons=skip_reasons,
    )
    return content_analysis, timeline, highlight_candidates, request_payload, raw_response


# Windowed video-understanding pipeline.  The legacy whole-video entry point above is
# intentionally retained as a disabled/live-call fallback while the workflow migrates.


def _units_for_window(
    window: VideoWindow,
    fine_grained_units: list[FineGrainedUnit],
) -> list[FineGrainedUnit]:
    return [
        unit
        for unit in fine_grained_units
        if unit.end > window.source_start + 0.01 and unit.start < window.source_end - 0.01
    ]


def _window_unit_payload(unit: FineGrainedUnit, window: VideoWindow) -> dict[str, Any]:
    return {
        "unit_id": unit.unit_id,
        "source_start": unit.start,
        "source_end": unit.end,
        "relative_start": round(max(0.0, unit.start - window.source_start), 2),
        "relative_end": round(min(window.duration, unit.end - window.source_start), 2),
        "text": unit.text,
        "has_pause_before": unit.has_pause_before,
        "has_pause_after": unit.has_pause_after,
        "keywords": unit.keywords,
    }


def _score_breakdown(payload: object, default_overall: float = 0.65) -> HighlightScoreBreakdown:
    values = payload if isinstance(payload, dict) else {}

    def score(name: str, default: float) -> float:
        try:
            return min(1.0, max(0.0, float(values.get(name, default))))
        except (TypeError, ValueError):
            return default

    return HighlightScoreBreakdown(
        overall=score("overall", default_overall),
        hook=score("hook", 0.5),
        emotion=score("emotion", 0.5),
        clarity=score("clarity", 0.7),
        platform_fit=score("platform_fit", 0.65),
        editability=score("editability", 0.7),
    )


def build_qwen_window_understanding_request(
    window: VideoWindow,
    user_request: UserRequest,
    video_info: VideoInfo,
    fine_grained_units: list[FineGrainedUnit],
    settings: AppSettings,
) -> dict[str, Any]:
    """Build one high-recall, source-aligned request for a short proxy-video window."""

    units = _units_for_window(window, fine_grained_units)
    return {
        "provider": "qwen",
        "base_url": settings.video_understanding_base_url,
        "api_path": settings.video_understanding_api_path,
        "model": settings.video_understanding_model,
        "timeout_seconds": settings.video_understanding_timeout_seconds,
        "stage": "window_understanding",
        "window_id": window.window_id,
        "max_tokens": settings.video_understanding_window_max_tokens,
        "video_fps": settings.video_understanding_window_fps,
        "min_pixels": settings.video_understanding_window_min_pixels,
        "max_pixels": settings.video_understanding_window_max_pixels,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the high-recall window stage of a video-understanding pipeline. Identify factual "
                    "visual events, a local semantic timeline, and every potentially useful highlight. Ground all "
                    "timeline segments and candidates in supplied transcript unit IDs. Do not invent timestamps. "
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": {
                    "instruction": (
                        "Analyze this complete proxy window once. Return summary, topics, visual_events, "
                        "timeline_segments, coarse_candidates, continues_from_previous, and continues_to_next. "
                        "Return zero to five coarse candidates; do not invent candidates merely to fill a quota. "
                        "Candidates must list source_unit_ids from transcript_units. The application derives all "
                        "source times from those IDs. Preserve possible hooks, explanations, visual moments, "
                        "emotional moments, actionable advice, and endings even when they are not final selections."
                    ),
                    "window": window.model_dump(exclude={"proxy_video_path"}),
                    "source_video": {
                        "duration": video_info.duration_seconds,
                        "fps": video_info.fps,
                        "width": video_info.width,
                        "height": video_info.height,
                    },
                    "user_request": user_request.model_dump(),
                    "transcript_units": [_window_unit_payload(unit, window) for unit in units],
                    "output_shape": {
                        "summary": "string",
                        "topics": ["string"],
                        "visual_events": [
                            {
                                "event_id": "string",
                                "description": "string",
                                "relative_start": "number|null",
                                "relative_end": "number|null",
                                "source_unit_ids": ["unit_id"],
                                "confidence": "0..1",
                            }
                        ],
                        "timeline_segments": [
                            {
                                "segment_id": "string",
                                "title": "string",
                                "summary": "string",
                                "transcript_excerpt": "string",
                                "source_unit_ids": ["unit_id"],
                                "relative_start": "number, only when no transcript units exist",
                                "relative_end": "number, only when no transcript units exist",
                                "importance": "0..1",
                                "hook_score": "0..1",
                                "highlight_score": "0..1",
                            }
                        ],
                        "coarse_candidates": [
                            {
                                "candidate_id": "string",
                                "title": "string",
                                "summary": "string",
                                "transcript_excerpt": "string",
                                "candidate_type": "string",
                                "semantic_role": "string",
                                "source_unit_ids": ["unit_id"],
                                "relative_start": "number, only when no transcript units exist",
                                "relative_end": "number, only when no transcript units exist",
                                "evidence_event_ids": ["event_id"],
                                "scores": {
                                    "overall": "0..1",
                                    "hook": "0..1",
                                    "emotion": "0..1",
                                    "clarity": "0..1",
                                    "platform_fit": "0..1",
                                    "editability": "0..1",
                                },
                                "dependencies": ["candidate_id"],
                                "redundancy_group": "string|null",
                                "risk_flags": ["string"],
                            }
                        ],
                        "continues_from_previous": "boolean",
                        "continues_to_next": "boolean",
                    },
                },
            },
        ],
        "response_format": _video_understanding_response_format(),
    }


def _call_window_provider(
    request_payload: dict[str, Any],
    window: VideoWindow | CandidateRefinementGroup,
    settings: AppSettings,
) -> dict[str, Any]:
    provider = OpenAICompatibleMultimodalProvider(
        api_key=settings.video_understanding_api_key,
        base_url=request_payload["base_url"],
        api_path=request_payload.get("api_path") or settings.video_understanding_api_path,
        timeout_seconds=int(request_payload.get("timeout_seconds") or settings.video_understanding_timeout_seconds),
    )
    messages = build_multimodal_video_messages(
        request_payload,
        window,
        project_root=settings.project_root,
        max_encoded_bytes=settings.video_understanding_max_base64_bytes,
    )
    return provider.create_multimodal_completion(
        {
            "model": request_payload["model"],
            "messages": messages,
            "temperature": 0.15,
            "max_tokens": int(request_payload["max_tokens"]),
            "enable_thinking": False,
            "response_format": request_payload["response_format"],
        }
    )


def _call_text_provider(
    request_payload: dict[str, Any],
    settings: AppSettings,
) -> dict[str, Any]:
    provider = OpenAICompatibleMultimodalProvider(
        api_key=settings.video_understanding_api_key,
        base_url=request_payload["base_url"],
        api_path=request_payload.get("api_path") or settings.video_understanding_api_path,
        timeout_seconds=int(request_payload.get("timeout_seconds") or settings.video_understanding_timeout_seconds),
    )
    return provider.create_multimodal_completion(
        {
            "model": request_payload["model"],
            "messages": request_payload["messages"],
            "temperature": float(request_payload.get("temperature", 0.1)),
            "max_tokens": int(request_payload["max_tokens"]),
            "enable_thinking": False,
            "response_format": request_payload.get("response_format") or _video_understanding_response_format(),
        }
    )


def _repair_window_response(
    response: dict[str, Any],
    error: Exception,
    request_payload: dict[str, Any],
    settings: AppSettings,
) -> dict[str, Any]:
    repair_request = {
        "base_url": request_payload["base_url"],
        "api_path": request_payload["api_path"],
        "model": request_payload["model"],
        "timeout_seconds": request_payload["timeout_seconds"],
        "max_tokens": settings.video_understanding_repair_max_tokens,
        "messages": [
            {
                "role": "system",
                "content": "Repair malformed window-understanding JSON. Return JSON only and do not add facts.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "validation_error": str(error),
                        "failed_response": response,
                        "original_output_shape": request_payload["messages"][1]["content"]["output_shape"],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "response_format": _video_understanding_response_format(),
    }
    return _call_text_provider(repair_request, settings)


def _coerce_window_understanding(
    payload: dict[str, Any],
    window: VideoWindow,
    units: list[FineGrainedUnit],
) -> WindowUnderstandingResult:
    units_by_id = {unit.unit_id: unit for unit in units}
    valid_unit_ids = set(units_by_id)
    events: list[WindowVisualEvent] = []
    valid_event_ids: set[str] = set()
    warnings: list[str] = []
    for index, raw in enumerate(payload.get("visual_events") or [], start=1):
        if not isinstance(raw, dict):
            continue
        unit_ids = [unit_id for unit_id in raw.get("source_unit_ids") or [] if unit_id in valid_unit_ids]
        event_id = str(raw.get("event_id") or f"{window.window_id}_event_{index:02d}")
        events.append(
            WindowVisualEvent(
                event_id=event_id,
                description=str(raw.get("description") or "Visual event in the proxy window."),
                relative_start=raw.get("relative_start"),
                relative_end=raw.get("relative_end"),
                source_unit_ids=unit_ids,
                confidence=float(raw.get("confidence", 0.5)),
            )
        )
        valid_event_ids.add(event_id)

    segments: list[TimelineSegment] = []
    for index, raw in enumerate(payload.get("timeline_segments") or [], start=1):
        if not isinstance(raw, dict):
            continue
        unit_ids = [unit_id for unit_id in raw.get("source_unit_ids") or [] if unit_id in valid_unit_ids]
        related_units = [units_by_id[unit_id] for unit_id in unit_ids]
        if related_units:
            start = round(min(unit.start for unit in related_units), 2)
            end = round(max(unit.end for unit in related_units), 2)
        elif not units and raw.get("relative_start") is not None and raw.get("relative_end") is not None:
            start = round(window.source_start + max(0.0, float(raw["relative_start"])), 2)
            end = round(min(window.source_end, window.source_start + float(raw["relative_end"])), 2)
            if end <= start:
                continue
        else:
            continue
        segments.append(
            TimelineSegment(
                segment_id=str(raw.get("segment_id") or f"{window.window_id}_segment_{index:02d}"),
                start=start,
                end=end,
                duration=round(end - start, 2),
                title=str(raw.get("title") or f"Window segment {index}"),
                summary=str(raw.get("summary") or raw.get("transcript_excerpt") or "Window timeline segment."),
                transcript_excerpt=str(
                    raw.get("transcript_excerpt") or " ".join(unit.text for unit in related_units)
                ),
                source_unit_ids=unit_ids,
                importance=float(raw.get("importance", 0.6)),
                hook_score=float(raw.get("hook_score", 0.4)),
                highlight_score=float(raw.get("highlight_score", raw.get("importance", 0.6))),
                emotion=str(raw.get("emotion") or "neutral"),
                tags=raw.get("tags") or [],
            )
        )

    candidates: list[CoarseHighlightCandidate] = []
    for index, raw in enumerate(payload.get("coarse_candidates") or [], start=1):
        if not isinstance(raw, dict):
            continue
        requested_ids = list(dict.fromkeys(raw.get("source_unit_ids") or []))
        unknown_ids = [unit_id for unit_id in requested_ids if unit_id not in valid_unit_ids]
        unit_ids = [unit_id for unit_id in requested_ids if unit_id in valid_unit_ids]
        if unknown_ids:
            warnings.append(f"Discarded unknown unit IDs from candidate {index}: {', '.join(unknown_ids)}")
        related_units = sorted((units_by_id[unit_id] for unit_id in unit_ids), key=lambda unit: unit.start)
        if related_units:
            start = round(related_units[0].start, 2)
            end = round(related_units[-1].end, 2)
        elif not units and raw.get("relative_start") is not None and raw.get("relative_end") is not None:
            start = round(window.source_start + max(0.0, float(raw["relative_start"])), 2)
            end = round(min(window.source_end, window.source_start + float(raw["relative_end"])), 2)
            if end <= start:
                continue
            warnings.append(f"Candidate {index} uses model-estimated visual-only boundaries.")
        else:
            continue
        candidate_id = str(raw.get("candidate_id") or f"{window.window_id}_candidate_{index:02d}")
        if not candidate_id.startswith(window.window_id):
            candidate_id = f"{window.window_id}_{candidate_id}"
        candidates.append(
            CoarseHighlightCandidate(
                candidate_id=candidate_id,
                window_id=window.window_id,
                title=str(raw.get("title") or f"Candidate {index}"),
                summary=str(raw.get("summary") or raw.get("transcript_excerpt") or "Potential highlight."),
                transcript_excerpt=str(
                    raw.get("transcript_excerpt") or " ".join(unit.text for unit in related_units)
                ),
                candidate_type=str(raw.get("candidate_type") or "semantic_highlight"),
                semantic_role=str(raw.get("semantic_role") or "supporting_point"),
                source_unit_ids=[unit.unit_id for unit in related_units],
                evidence_event_ids=[
                    event_id for event_id in raw.get("evidence_event_ids") or [] if event_id in valid_event_ids
                ],
                source_start=start,
                source_end=end,
                scores=_score_breakdown(raw.get("scores")),
                dependencies=raw.get("dependencies") or [],
                redundancy_group=raw.get("redundancy_group"),
                risk_flags=raw.get("risk_flags") or [],
            )
        )

    if not segments and units:
        segments = _window_stub_segments(window, units)
        warnings.append("Model returned no unit-grounded timeline; local transcript segments were used.")
    return WindowUnderstandingResult(
        window_id=window.window_id,
        order=window.order,
        source_start=window.source_start,
        source_end=window.source_end,
        summary=str(payload.get("summary") or f"Understanding for {window.window_id}."),
        topics=payload.get("topics") or [],
        visual_events=events,
        timeline_segments=segments,
        coarse_candidates=candidates,
        continues_from_previous=bool(payload.get("continues_from_previous", False)),
        continues_to_next=bool(payload.get("continues_to_next", False)),
        warnings=warnings,
    )


def _window_stub_segments(window: VideoWindow, units: list[FineGrainedUnit]) -> list[TimelineSegment]:
    segments = _build_stub_timeline_segments(units)
    return [
        segment.model_copy(update={"segment_id": f"{window.window_id}_{segment.segment_id}"})
        for segment in segments
    ]


def _window_fallback(window: VideoWindow, units: list[FineGrainedUnit], warning: str) -> WindowUnderstandingResult:
    segments = _window_stub_segments(window, units)
    units_by_id = {unit.unit_id: unit for unit in units}
    legacy_candidates = _build_stub_highlight_candidates(segments, units_by_id)
    candidates = [
        CoarseHighlightCandidate(
            candidate_id=f"{window.window_id}_{candidate.candidate_id}",
            window_id=window.window_id,
            title=candidate.title,
            summary=candidate.summary,
            transcript_excerpt=candidate.transcript_excerpt,
            candidate_type=candidate.highlight_type,
            semantic_role=candidate.semantic_role,
            source_unit_ids=candidate.transcript_unit_ids,
            source_start=candidate.source_start,
            source_end=candidate.source_end,
            scores=candidate.scores,
            dependencies=candidate.dependencies,
            redundancy_group=candidate.redundancy_group,
            generation_source="local_window_fallback",
            risk_flags=[*candidate.risk_flags, "window_llm_fallback"],
        )
        for candidate in legacy_candidates[:5]
    ]
    return WindowUnderstandingResult(
        window_id=window.window_id,
        order=window.order,
        source_start=window.source_start,
        source_end=window.source_end,
        summary=f"Transcript-backed fallback understanding for {window.window_id}.",
        topics=sorted({keyword for unit in units for keyword in unit.keywords[:2]}),
        timeline_segments=segments,
        coarse_candidates=candidates,
        generation_mode="local_window_fallback_v1",
        warnings=[warning],
    )


def understand_video_window(
    window: VideoWindow,
    user_request: UserRequest,
    video_info: VideoInfo,
    fine_grained_units: list[FineGrainedUnit],
    settings: AppSettings,
) -> tuple[WindowUnderstandingResult, dict[str, Any], dict[str, Any]]:
    """Understand one proxy window, repairing JSON once and falling back locally if needed."""

    units = _units_for_window(window, fine_grained_units)
    request_payload = build_qwen_window_understanding_request(
        window, user_request, video_info, fine_grained_units, settings
    )
    response: dict[str, Any] = {}
    repair_response: dict[str, Any] | None = None
    try:
        response = _call_window_provider(request_payload, window, settings)
        try:
            payload = _extract_json_payload(_extract_response_text(response))
            result = _coerce_window_understanding(payload, window, units)
        except (ClipPilotProcessingError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            repair_response = _repair_window_response(response, exc, request_payload, settings)
            payload = _extract_json_payload(_extract_response_text(repair_response))
            result = _coerce_window_understanding(payload, window, units)
        raw = {"primary_response": response, "repair_response": repair_response, "used_fallback": False}
        return result, request_payload, raw
    except Exception as exc:
        result = _window_fallback(window, units, f"Window understanding failed: {exc}")
        raw = {
            "primary_response": response,
            "repair_response": repair_response,
            "used_fallback": True,
            "error": str(exc),
        }
        return result, request_payload, raw


def _all_coarse_candidates(
    window_results: list[WindowUnderstandingResult],
) -> list[CoarseHighlightCandidate]:
    return [candidate for result in window_results for candidate in result.coarse_candidates]


def _candidate_overlap_score(left: CoarseHighlightCandidate, right: CoarseHighlightCandidate) -> float:
    left_units = set(left.source_unit_ids)
    right_units = set(right.source_unit_ids)
    if left_units or right_units:
        union = left_units | right_units
        unit_overlap = len(left_units & right_units) / len(union) if union else 0.0
    else:
        unit_overlap = 0.0
    intersection = max(0.0, min(left.source_end, right.source_end) - max(left.source_start, right.source_start))
    union_duration = max(left.source_end, right.source_end) - min(left.source_start, right.source_start)
    time_iou = intersection / union_duration if union_duration > 0 else 0.0
    return max(unit_overlap, time_iou)


def _cluster_coarse_candidates(
    candidates: list[CoarseHighlightCandidate],
) -> tuple[list[CoarseHighlightCandidate], list[CandidateCluster]]:
    """Merge duplicate overlap-window candidates while preserving the most complete canonical span."""

    canonical: list[CoarseHighlightCandidate] = []
    cluster_members: list[list[str]] = []
    for candidate in sorted(candidates, key=lambda item: (item.source_start, -item.scores.overall)):
        match_index = next(
            (
                index
                for index, existing in enumerate(canonical)
                if _candidate_overlap_score(existing, candidate) >= 0.6
            ),
            None,
        )
        if match_index is None:
            canonical.append(candidate)
            cluster_members.append([candidate.candidate_id])
            continue
        cluster_members[match_index].append(candidate.candidate_id)
        existing = canonical[match_index]
        existing_span = existing.source_end - existing.source_start
        candidate_span = candidate.source_end - candidate.source_start
        if (candidate_span, candidate.scores.overall) > (existing_span, existing.scores.overall):
            canonical[match_index] = candidate

    clusters = [
        CandidateCluster(
            cluster_id=f"cluster_{index:03d}",
            canonical_candidate_id=candidate.candidate_id,
            source_candidate_ids=members,
        )
        for index, (candidate, members) in enumerate(zip(canonical, cluster_members), start=1)
    ]
    return canonical, clusters


def _local_global_understanding(
    window_results: list[WindowUnderstandingResult],
    user_request: UserRequest,
    *,
    max_candidates: int,
    warning: str | None = None,
) -> GlobalUnderstandingResult:
    candidates, clusters = _cluster_coarse_candidates(_all_coarse_candidates(window_results))
    candidates.sort(
        key=lambda item: (
            item.scores.overall,
            item.scores.hook,
            item.scores.editability,
            -item.source_start,
        ),
        reverse=True,
    )
    shortlist = [candidate.candidate_id for candidate in candidates[:max_candidates]]
    reserve = [candidate.candidate_id for candidate in candidates[max_candidates : max_candidates + 10]]
    chapters = _dedupe_window_segments(window_results)
    return GlobalUnderstandingResult(
        summary=" ".join(result.summary for result in window_results if result.summary)[:1200]
        or "Windowed video understanding completed.",
        narrative_flow=[result.summary for result in window_results if result.summary],
        main_topics=sorted({topic for result in window_results for topic in result.topics}),
        chapters=chapters,
        candidate_clusters=clusters,
        refinement_candidate_ids=shortlist,
        reserve_candidate_ids=reserve,
        coverage_warnings=[warning] if warning else [],
        generation_mode="local_window_aggregation_v1",
    )


def _dedupe_window_segments(window_results: list[WindowUnderstandingResult]) -> list[TimelineSegment]:
    selected: list[TimelineSegment] = []
    seen_unit_sets: list[set[str]] = []
    for segment in sorted(
        (segment for result in window_results for segment in result.timeline_segments),
        key=lambda item: (item.start, item.end),
    ):
        unit_set = set(segment.source_unit_ids)
        duplicate = False
        for existing_units in seen_unit_sets:
            union = unit_set | existing_units
            if union and len(unit_set & existing_units) / len(union) >= 0.8:
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(segment)
        seen_unit_sets.append(unit_set)
    return selected


def build_qwen_global_understanding_request(
    window_results: list[WindowUnderstandingResult],
    user_request: UserRequest,
    video_info: VideoInfo,
    settings: AppSettings,
) -> dict[str, Any]:
    candidates, clusters = _cluster_coarse_candidates(_all_coarse_candidates(window_results))
    return {
        "base_url": settings.video_understanding_base_url,
        "api_path": settings.video_understanding_api_path,
        "model": settings.video_understanding_model,
        "timeout_seconds": settings.video_understanding_timeout_seconds,
        "max_tokens": settings.video_understanding_global_max_tokens,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You aggregate independently analyzed video windows. Preserve coverage and narrative structure, "
                    "deduplicate overlapping candidates, and select candidates for focused refinement. Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "instruction": (
                            "Return summary, narrative_flow, main_topics, refinement_candidate_ids, "
                            "reserve_candidate_ids, and coverage_warnings. Select no more than the configured limit. "
                            "Use only supplied canonical candidate IDs. Balance hooks, explanations, visual moments, "
                            "actionable advice, endings, and chapter coverage instead of selecting by score alone."
                        ),
                        "source_duration": video_info.duration_seconds,
                        "user_request": user_request.model_dump(),
                        "max_refinement_candidates": settings.video_understanding_refinement_max_candidates,
                        "windows": [
                            {
                                "window_id": result.window_id,
                                "source_start": result.source_start,
                                "source_end": result.source_end,
                                "summary": result.summary,
                                "topics": result.topics,
                                "continues_from_previous": result.continues_from_previous,
                                "continues_to_next": result.continues_to_next,
                            }
                            for result in window_results
                        ],
                        "canonical_candidates": [candidate.model_dump() for candidate in candidates],
                        "duplicate_clusters": [cluster.model_dump() for cluster in clusters],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "response_format": _video_understanding_response_format(),
    }


def aggregate_window_understandings(
    window_results: list[WindowUnderstandingResult],
    user_request: UserRequest,
    video_info: VideoInfo,
    settings: AppSettings,
) -> tuple[GlobalUnderstandingResult, dict[str, Any], dict[str, Any]]:
    """Aggregate window results with one text call and a deterministic coverage-preserving fallback."""

    request_payload = build_qwen_global_understanding_request(window_results, user_request, video_info, settings)
    candidates, clusters = _cluster_coarse_candidates(_all_coarse_candidates(window_results))
    valid_ids = {candidate.candidate_id for candidate in candidates}
    response: dict[str, Any] = {}
    try:
        response = _call_text_provider(request_payload, settings)
        payload = _extract_json_payload(_extract_response_text(response))
        shortlist = [
            candidate_id
            for candidate_id in payload.get("refinement_candidate_ids") or []
            if candidate_id in valid_ids
        ][: settings.video_understanding_refinement_max_candidates]
        if not shortlist and candidates:
            raise ClipPilotProcessingError("Global aggregation returned no valid refinement candidate IDs.")
        result = GlobalUnderstandingResult(
            summary=str(payload.get("summary") or "Windowed video understanding completed."),
            narrative_flow=payload.get("narrative_flow") or [result.summary for result in window_results],
            main_topics=payload.get("main_topics") or sorted(
                {topic for result in window_results for topic in result.topics}
            ),
            chapters=_dedupe_window_segments(window_results),
            candidate_clusters=clusters,
            refinement_candidate_ids=list(dict.fromkeys(shortlist)),
            reserve_candidate_ids=[
                candidate_id
                for candidate_id in payload.get("reserve_candidate_ids") or []
                if candidate_id in valid_ids and candidate_id not in shortlist
            ][:10],
            coverage_warnings=payload.get("coverage_warnings") or [],
        )
        return result, request_payload, {"response": response, "used_fallback": False}
    except Exception as exc:
        result = _local_global_understanding(
            window_results,
            user_request,
            max_candidates=settings.video_understanding_refinement_max_candidates,
            warning=f"Global aggregation fell back locally: {exc}",
        )
        return result, request_payload, {"response": response, "used_fallback": True, "error": str(exc)}


def build_candidate_refinement_groups(
    candidates: list[CoarseHighlightCandidate],
    selected_candidate_ids: list[str],
    video_duration: float,
    settings: AppSettings,
) -> list[CandidateRefinementGroup]:
    """Group nearby shortlisted candidates into bounded padded refinement ranges."""

    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    selected = [candidates_by_id[candidate_id] for candidate_id in selected_candidate_ids if candidate_id in candidates_by_id]
    selected.sort(key=lambda item: (item.source_start, item.source_end))
    groups: list[CandidateRefinementGroup] = []
    padding = settings.video_understanding_refinement_padding_seconds
    merge_gap = settings.video_understanding_refinement_merge_gap_seconds
    max_duration = settings.video_understanding_refinement_max_group_duration_seconds
    for candidate in selected:
        padded_start = round(max(0.0, candidate.source_start - padding), 2)
        padded_end = round(min(video_duration, candidate.source_end + padding), 2)
        if groups:
            current = groups[-1]
            merged_end = max(current.source_end, padded_end)
            if padded_start <= current.source_end + merge_gap and merged_end - current.source_start <= max_duration:
                groups[-1] = current.model_copy(
                    update={
                        "source_end": round(merged_end, 2),
                        "candidate_ids": [*current.candidate_ids, candidate.candidate_id],
                    }
                )
                continue
        groups.append(
            CandidateRefinementGroup(
                group_id=f"refinement_{len(groups) + 1:03d}",
                source_start=padded_start,
                source_end=padded_end,
                candidate_ids=[candidate.candidate_id],
            )
        )
    return groups


def _allowed_boundaries(units: list[FineGrainedUnit]) -> dict[str, float]:
    boundaries: dict[str, float] = {}
    for unit in units:
        boundaries[f"boundary_{unit.unit_id}_start"] = round(unit.start, 2)
        boundaries[f"boundary_{unit.unit_id}_end"] = round(unit.end, 2)
    return boundaries


def build_qwen_candidate_refinement_request(
    group: CandidateRefinementGroup,
    candidates: list[CoarseHighlightCandidate],
    units: list[FineGrainedUnit],
    settings: AppSettings,
) -> dict[str, Any]:
    boundaries = _allowed_boundaries(units)
    return {
        "base_url": settings.video_understanding_base_url,
        "api_path": settings.video_understanding_api_path,
        "model": settings.video_understanding_model,
        "timeout_seconds": settings.video_understanding_timeout_seconds,
        "max_tokens": settings.video_understanding_refinement_max_tokens,
        "video_fps": settings.video_understanding_refinement_fps,
        "min_pixels": settings.video_understanding_window_min_pixels,
        "max_pixels": settings.video_understanding_window_max_pixels,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You refine previously discovered highlight candidates using denser video evidence. Select only "
                    "supplied boundary IDs and unit IDs. Return JSON only and never invent timestamps."
                ),
            },
            {
                "role": "user",
                "content": {
                    "instruction": (
                        "For each candidate return semantic_complete, visual_complete, context requirements, scores, "
                        "risk_flags, and up to three options named short, ideal, and extended. Every option must use "
                        "start_boundary_id/end_boundary_id from allowed_boundaries and contiguous source_unit_ids."
                    ),
                    "group": group.model_dump(exclude={"proxy_video_path"}),
                    "candidates": [candidate.model_dump() for candidate in candidates],
                    "context_units": [unit.model_dump() for unit in units],
                    "allowed_boundaries": [
                        {"boundary_id": boundary_id, "timestamp": timestamp}
                        for boundary_id, timestamp in boundaries.items()
                    ],
                    "output_shape": {
                        "refined_candidates": [
                            {
                                "candidate_id": "existing candidate ID",
                                "semantic_complete": "boolean",
                                "visual_complete": "boolean",
                                "requires_previous_context": "boolean",
                                "requires_next_context": "boolean",
                                "scores": "score object",
                                "risk_flags": ["string"],
                                "options": [
                                    {
                                        "option_type": "short|ideal|extended",
                                        "start_boundary_id": "allowed ID",
                                        "end_boundary_id": "allowed ID",
                                        "source_unit_ids": ["unit_id"],
                                    }
                                ],
                            }
                        ]
                    },
                },
            },
        ],
        "response_format": _video_understanding_response_format(),
    }


def _local_refined_candidate(
    candidate: CoarseHighlightCandidate,
    units_by_id: dict[str, FineGrainedUnit],
    *,
    risk: str | None = None,
) -> RefinedHighlightCandidate:
    related = [units_by_id[unit_id] for unit_id in candidate.source_unit_ids if unit_id in units_by_id]
    related.sort(key=lambda unit: unit.start)
    if not related:
        option = RefinedCandidateOption(
            option_id=f"{candidate.candidate_id}_ideal",
            option_type="ideal",
            start_boundary_id=f"boundary_{candidate.candidate_id}_visual_start",
            end_boundary_id=f"boundary_{candidate.candidate_id}_visual_end",
            source_unit_ids=[],
            source_start=candidate.source_start,
            source_end=candidate.source_end,
            duration=round(candidate.source_end - candidate.source_start, 2),
        )
        return RefinedHighlightCandidate(
            candidate_id=candidate.candidate_id,
            options=[option],
            scores=candidate.scores,
            semantic_complete=False,
            visual_complete=True,
            dependencies=candidate.dependencies,
            risk_flags=[*candidate.risk_flags, "visual_only_boundary", *([risk] if risk else [])],
            generation_mode="local_visual_candidate_refinement_v1",
        )
    variants: list[tuple[str, list[FineGrainedUnit]]] = [("short", [related[0]])]
    if len(related) > 1:
        variants.append(("ideal", related))
    else:
        variants.append(("ideal", related))
    variants.append(("extended", related))
    options: list[RefinedCandidateOption] = []
    seen: set[tuple[float, float]] = set()
    for option_type, option_units in variants:
        start = round(option_units[0].start, 2)
        end = round(option_units[-1].end, 2)
        key = (start, end)
        if key in seen and option_type != "ideal":
            continue
        seen.add(key)
        options.append(
            RefinedCandidateOption(
                option_id=f"{candidate.candidate_id}_{option_type}",
                option_type=option_type,
                start_boundary_id=f"boundary_{option_units[0].unit_id}_start",
                end_boundary_id=f"boundary_{option_units[-1].unit_id}_end",
                source_unit_ids=[unit.unit_id for unit in option_units],
                source_start=start,
                source_end=end,
                duration=round(end - start, 2),
            )
        )
    return RefinedHighlightCandidate(
        candidate_id=candidate.candidate_id,
        options=options,
        scores=candidate.scores,
        semantic_complete=True,
        visual_complete=False,
        dependencies=candidate.dependencies,
        risk_flags=[*candidate.risk_flags, *([risk] if risk else [])],
        generation_mode="local_candidate_refinement_v1",
    )


def _coerce_refinement_response(
    payload: dict[str, Any],
    group_candidates: list[CoarseHighlightCandidate],
    units: list[FineGrainedUnit],
) -> list[RefinedHighlightCandidate]:
    candidates_by_id = {candidate.candidate_id: candidate for candidate in group_candidates}
    units_by_id = {unit.unit_id: unit for unit in units}
    boundaries = _allowed_boundaries(units)
    refined: list[RefinedHighlightCandidate] = []
    for raw in payload.get("refined_candidates") or []:
        if not isinstance(raw, dict) or raw.get("candidate_id") not in candidates_by_id:
            continue
        candidate = candidates_by_id[str(raw["candidate_id"])]
        options: list[RefinedCandidateOption] = []
        seen_types: set[str] = set()
        for option_raw in raw.get("options") or []:
            if not isinstance(option_raw, dict):
                continue
            option_type = str(option_raw.get("option_type") or "")
            start_id = str(option_raw.get("start_boundary_id") or "")
            end_id = str(option_raw.get("end_boundary_id") or "")
            unit_ids = [unit_id for unit_id in option_raw.get("source_unit_ids") or [] if unit_id in units_by_id]
            if option_type not in {"short", "ideal", "extended"} or option_type in seen_types:
                continue
            if start_id not in boundaries or end_id not in boundaries or boundaries[end_id] <= boundaries[start_id]:
                continue
            if not unit_ids:
                continue
            seen_types.add(option_type)
            start = boundaries[start_id]
            end = boundaries[end_id]
            options.append(
                RefinedCandidateOption(
                    option_id=f"{candidate.candidate_id}_{option_type}",
                    option_type=option_type,
                    start_boundary_id=start_id,
                    end_boundary_id=end_id,
                    source_unit_ids=unit_ids,
                    source_start=start,
                    source_end=end,
                    duration=round(end - start, 2),
                )
            )
        if not options:
            refined.append(_local_refined_candidate(candidate, units_by_id, risk="invalid_llm_refinement_options"))
            continue
        refined.append(
            RefinedHighlightCandidate(
                candidate_id=candidate.candidate_id,
                options=options,
                scores=_score_breakdown(raw.get("scores"), candidate.scores.overall),
                semantic_complete=bool(raw.get("semantic_complete", False)),
                visual_complete=bool(raw.get("visual_complete", False)),
                requires_previous_context=bool(raw.get("requires_previous_context", False)),
                requires_next_context=bool(raw.get("requires_next_context", False)),
                dependencies=candidate.dependencies,
                risk_flags=raw.get("risk_flags") or [],
            )
        )
    returned_ids = {candidate.candidate_id for candidate in refined}
    for candidate in group_candidates:
        if candidate.candidate_id not in returned_ids:
            refined.append(_local_refined_candidate(candidate, units_by_id, risk="candidate_missing_from_refinement"))
    return refined


def refine_candidate_group(
    group: CandidateRefinementGroup,
    group_candidates: list[CoarseHighlightCandidate],
    source_video_path: Path,
    output_dir: Path,
    fine_grained_units: list[FineGrainedUnit],
    settings: AppSettings,
) -> tuple[list[RefinedHighlightCandidate], CandidateRefinementGroup, dict[str, Any], dict[str, Any]]:
    """Render and refine one candidate group with one dense-video call and local fallback."""

    units = [
        unit
        for unit in fine_grained_units
        if unit.end > group.source_start + 0.01 and unit.start < group.source_end - 0.01
    ]
    units_by_id = {unit.unit_id: unit for unit in units}
    output_path = output_dir / f"{group.group_id}.mp4"
    rendered_group = group.model_copy(update={"proxy_video_path": str(output_path)})
    request_payload: dict[str, Any] = {}
    response: dict[str, Any] = {}
    try:
        _render_proxy_window(
            source_video_path,
            output_path,
            group.source_start,
            group.source_end,
            proxy_height=settings.video_understanding_proxy_height,
        )
        request_payload = build_qwen_candidate_refinement_request(rendered_group, group_candidates, units, settings)
        response = _call_window_provider(request_payload, rendered_group, settings)  # same media contract
        payload = _extract_json_payload(_extract_response_text(response))
        refined = _coerce_refinement_response(payload, group_candidates, units)
        return refined, rendered_group, request_payload, {"response": response, "used_fallback": False}
    except Exception as exc:
        refined = [
            _local_refined_candidate(candidate, units_by_id, risk=f"refinement_fallback: {exc}")
            for candidate in group_candidates
        ]
        return refined, rendered_group, request_payload, {
            "response": response,
            "used_fallback": True,
            "error": str(exc),
        }


def merge_windowed_video_understanding_outputs(
    global_result: GlobalUnderstandingResult,
    coarse_candidates: list[CoarseHighlightCandidate],
    refined_candidates: list[RefinedHighlightCandidate],
    user_request: UserRequest,
    settings: AppSettings,
) -> tuple[ContentAnalysis, VideoTimeline, list[LLMHighlightCandidate]]:
    """Adapt windowed results back into the stable downstream project-state contract."""

    coarse_by_id = {candidate.candidate_id: candidate for candidate in coarse_candidates}
    llm_candidates: list[LLMHighlightCandidate] = []
    for refined in refined_candidates:
        coarse = coarse_by_id.get(refined.candidate_id)
        if coarse is None or not refined.options:
            continue
        options = sorted(refined.options, key=lambda option: option.duration)
        ideal = next((option for option in options if option.option_type == "ideal"), options[len(options) // 2])
        source_start = min(option.source_start for option in options)
        source_end = max(option.source_end for option in options)
        unit_ids = list(dict.fromkeys(unit_id for option in options for unit_id in option.source_unit_ids))
        trim_policy = _build_trim_policy(
            source_start=source_start,
            source_end=source_end,
            preferred_start=ideal.source_start,
            preferred_end=ideal.source_end,
            safe_cut_points=sorted(
                {point for option in options for point in (option.source_start, option.source_end)}
            ),
            requested_min_duration=min(option.duration for option in options),
            requested_ideal_duration=ideal.duration,
            requested_max_duration=max(option.duration for option in options),
            trimmable=len(options) > 1,
        )
        llm_candidates.append(
            LLMHighlightCandidate(
                candidate_id=coarse.candidate_id,
                source_segment_ids=[
                    segment.segment_id
                    for segment in global_result.chapters
                    if set(segment.source_unit_ids).intersection(unit_ids)
                ],
                transcript_unit_ids=unit_ids,
                source_start=source_start,
                source_end=source_end,
                duration=round(source_end - source_start, 2),
                title=coarse.title,
                summary=coarse.summary,
                transcript_excerpt=coarse.transcript_excerpt,
                highlight_type=coarse.candidate_type,
                semantic_role=coarse.semantic_role,
                reason="Candidate discovered by a window pass and validated by focused refinement.",
                scores=refined.scores,
                trim_policy=trim_policy,
                dependencies=refined.dependencies,
                redundancy_group=coarse.redundancy_group,
                risk_flags=list(dict.fromkeys([*coarse.risk_flags, *refined.risk_flags])),
            )
        )

    content_analysis = ContentAnalysis(
        summary=global_result.summary,
        narrative_flow=global_result.narrative_flow,
        main_topics=global_result.main_topics,
        hook_candidates=[
            candidate.transcript_excerpt
            for candidate in sorted(llm_candidates, key=lambda item: item.scores.hook, reverse=True)[:3]
        ],
        tone=user_request.edit_style,
        recommended_structure=["opening_hook", "core_point", "ending"],
        pacing="medium",
    )
    timeline = VideoTimeline(
        provider="qwen",
        model=settings.video_understanding_model,
        overview=global_result.summary,
        segments=global_result.chapters,
        generation_mode="llm_windowed_refinement_v1",
        warnings=global_result.coverage_warnings,
    )
    return content_analysis, timeline, llm_candidates
