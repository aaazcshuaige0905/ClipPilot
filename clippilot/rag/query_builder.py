from __future__ import annotations

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
            f"platform {user_request.target_platform}",
            f"duration {user_request.target_duration} seconds",
            f"style {user_request.edit_style}",
            f"language {user_request.language}",
            f"keywords {' '.join(keywords)}" if keywords else "",
            "need strategy for short-video editing planner",
        ]
        if part
    )
    return RetrievalQuery(
        text=query_text,
        platform=user_request.target_platform.strip().lower(),
        language=user_request.language.strip().lower(),
        style=user_request.edit_style.strip().lower(),
        duration_band=duration_band_for(user_request.target_duration),
        stage="planning",
        top_k_dense=settings.rag_top_k_dense,
        top_k_bm25=settings.rag_top_k_bm25,
        top_k_final=settings.rag_top_k_final,
        tags=keywords,
    )
