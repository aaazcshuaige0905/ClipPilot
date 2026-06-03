from typing import Any

from clippilot.core.exceptions import ClipPilotValidationError
from clippilot.schemas.editing_plan import HighlightCandidate, HighlightCandidatesResult
from clippilot.schemas.transcript import TranscriptResult
from clippilot.schemas.user_request import UserRequest
from clippilot.storage.path_manager import AppSettings, load_settings

HIGHLIGHT_KEYWORDS = {
    "关键": 0.16,
    "核心": 0.16,
    "最重要": 0.20,
    "但是": 0.12,
    "真正": 0.14,
    "为什么": 0.14,
    "如何": 0.14,
    "失败": 0.18,
    "成功": 0.18,
}

HOOK_PHRASES = {
    "我认为": 0.18,
    "先说结论": 0.22,
    "结论是": 0.22,
    "最重要的是": 0.24,
    "你一定要": 0.18,
    "真正的问题": 0.20,
}

INCOMPLETE_STARTS = ("但是", "然后", "所以", "而且", "并且")
INCOMPLETE_ENDINGS = ("但是", "所以", "然后", "如果", "因为", "就是", "那个", "这个")


def _normalize_user_request(user_request: UserRequest | dict[str, Any]) -> dict[str, Any]:
    """Convert a request model or mapping into a plain dictionary for heuristic scoring."""

    if isinstance(user_request, UserRequest):
        return user_request.model_dump()
    return dict(user_request)


def _merge_segments(transcript: TranscriptResult, settings: AppSettings) -> list[dict[str, Any]]:
    """Merge consecutive transcript segments into 8-20 second candidate windows."""

    segments = transcript.segments
    merged_candidates: list[dict[str, Any]] = []
    seen_ranges: set[tuple[float, float]] = set()

    for start_index in range(len(segments)):
        window_segments = []
        window_start = float(segments[start_index].start)

        for current_index in range(start_index, len(segments)):
            segment = segments[current_index]
            window_segments.append(segment)
            window_end = float(segment.end)
            duration = round(window_end - window_start, 2)

            if duration > settings.highlight_max_candidate_duration:
                break

            if settings.highlight_min_candidate_duration <= duration <= settings.highlight_max_candidate_duration:
                text = " ".join(item.text.strip() for item in window_segments if item.text.strip())
                range_key = (round(window_start, 2), round(window_end, 2))
                if text and range_key not in seen_ranges:
                    merged_candidates.append(
                        {
                            "start": range_key[0],
                            "end": range_key[1],
                            "text": text,
                        }
                    )
                    seen_ranges.add(range_key)

    if not merged_candidates and segments:
        fallback_end_index = min(len(segments) - 1, 1)
        merged_candidates.append(
            {
                "start": float(segments[0].start),
                "end": float(segments[fallback_end_index].end),
                "text": " ".join(segment.text.strip() for segment in segments[: fallback_end_index + 1]),
            }
        )

    return merged_candidates


def _score_candidate(candidate: dict[str, Any], request_data: dict[str, Any]) -> tuple[float, str]:
    """Score a candidate clip with deterministic local rules and return an explanation."""

    text = candidate["text"].strip()
    duration = candidate["end"] - candidate["start"]
    text_length = len(text)
    score = 0.35
    reasons: list[str] = []

    for keyword, bonus in HIGHLIGHT_KEYWORDS.items():
        if keyword in text:
            score += bonus
            reasons.append(f"Contains keyword '{keyword}'")

    for hook_phrase, bonus in HOOK_PHRASES.items():
        if hook_phrase in text:
            score += bonus
            reasons.append(f"Uses hook phrase '{hook_phrase}'")

    if candidate["start"] <= 20 and any(marker in text for marker in ("为什么", "最重要", "真正", "我认为", "先说结论")):
        score += 0.12
        reasons.append("Works well as an opening hook")

    if request_data.get("edit_style", "").lower() in {"powerful", "energetic"} and any(
        marker in text for marker in ("最重要", "真正", "成功", "失败", "为什么")
    ):
        score += 0.08
        reasons.append("Matches the requested editing style")

    if text_length < 18:
        score -= 0.20
        reasons.append("Text is too short to stand alone")
    elif text_length > 140:
        score -= 0.10
        reasons.append("Text is too long for a concise highlight")
    else:
        score += 0.06
        reasons.append("Text length suits a short highlight")

    if duration < 9.5:
        score -= 0.08
        reasons.append("Duration is slightly short")
    elif duration > 18:
        score -= 0.05
        reasons.append("Duration is slightly long")
    else:
        score += 0.08
        reasons.append("Duration fits short highlight pacing")

    if text.startswith(INCOMPLETE_STARTS) or text.endswith(INCOMPLETE_ENDINGS):
        score -= 0.18
        reasons.append("The segment feels semantically incomplete")

    if not any(punctuation in text for punctuation in ("。", "！", "？", "!", "?", ".")) and text_length < 24:
        score -= 0.08
        reasons.append("The sentence boundary is weak")

    if "[mock-" in text.lower():
        score -= 0.06
        reasons.append("Mock transcript text is less content-rich")

    final_score = round(max(0.0, min(score, 0.99)), 2)
    reason = ". ".join(reasons[:3]) if reasons else "Balanced timing and useful information density."
    if not reason.endswith("."):
        reason = f"{reason}."

    return final_score, reason


def generate_highlight_candidates(
    transcript: TranscriptResult,
    user_request: UserRequest | dict[str, Any],
    settings: AppSettings | None = None,
) -> HighlightCandidatesResult:
    """Generate and rank highlight candidates from a structured transcript."""

    if not transcript.video_id:
        raise ClipPilotValidationError(
            "Transcript is missing video_id and cannot generate highlight candidates."
        )
    if not transcript.segments:
        raise ClipPilotValidationError("Transcript does not contain any ASR segments.")

    active_settings = settings or load_settings()
    request_data = _normalize_user_request(user_request)
    merged_candidates = _merge_segments(transcript, active_settings)
    scored_candidates: list[HighlightCandidate] = []

    for candidate in merged_candidates:
        score, reason = _score_candidate(candidate, request_data)
        scored_candidates.append(
            HighlightCandidate(
                start=candidate["start"],
                end=candidate["end"],
                text=candidate["text"],
                score=score,
                reason=reason,
            )
        )

    scored_candidates.sort(key=lambda item: item.score, reverse=True)
    return HighlightCandidatesResult(video_id=transcript.video_id, candidates=scored_candidates[:8])
