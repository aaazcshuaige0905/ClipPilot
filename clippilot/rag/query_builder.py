from __future__ import annotations

from clippilot.rag.metadata_normalization import normalize_language, normalize_platform, normalize_style
from clippilot.rag.schemas import RetrievalQuery
from clippilot.schemas.project_state import FineGrainedUnit
from clippilot.schemas.user_request import UserRequest
from clippilot.storage.path_manager import AppSettings


def duration_band_for(seconds: int | float) -> str:
    """Bucket a target duration into a stable retrieval band."""

    if seconds <= 15:
        return "0_15"
    if seconds <= 30:
        return "15_30"
    if seconds <= 60:
        return "30_60"
    return "60_plus"


def build_retrieval_query(
    user_request: UserRequest,
    fine_grained_units: list[FineGrainedUnit] | None,
    settings: AppSettings,
) -> RetrievalQuery:
    """Build a planner-oriented retrieval query from the current task context."""

    normalized_platform = normalize_platform(user_request.target_platform)
    normalized_language = normalize_language(user_request.language)
    normalized_style = normalize_style(user_request.edit_style)
    keywords: list[str] = []
    for unit in fine_grained_units or []:
        for keyword in unit.keywords:
            if keyword not in keywords:
                keywords.append(keyword)
            if len(keywords) >= 8:
                break
        if len(keywords) >= 8:
            break

    query_text = " | ".join(
        part
        for part in [
            _metadata_query_phrase("platform", user_request.target_platform, normalized_platform),
            f"duration {user_request.target_duration} seconds",
            _metadata_query_phrase("style", user_request.edit_style, normalized_style),
            _metadata_query_phrase("language", user_request.language, normalized_language),
            f"keywords {' '.join(keywords)}" if keywords else "",
            "need strategy for short-video editing planner",
        ]
        if part
    )
    return RetrievalQuery(
        text=query_text,
        platform=normalized_platform,
        language=normalized_language,
        style=normalized_style,
        duration_band=duration_band_for(user_request.target_duration),
        stage="planning",
        top_k_dense=settings.rag_top_k_dense,
        top_k_bm25=settings.rag_top_k_bm25,
        top_k_final=settings.rag_top_k_final,
        tags=keywords,
    )


def _metadata_query_phrase(label: str, raw_value: str, normalized_value: str | None) -> str:
    """Build one retrieval phrase that keeps user wording while appending the canonical value."""

    raw = raw_value.strip()
    if not raw:
        return ""

    raw_lower = raw.lower()
    if normalized_value and normalized_value != raw_lower:
        return f"{label} {raw} {normalized_value}"
    return f"{label} {raw}"
