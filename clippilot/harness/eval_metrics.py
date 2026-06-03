def summarize_candidate_scores(scores: list[float]) -> dict[str, float]:
    """Return lightweight score statistics for future evaluation hooks."""

    if not scores:
        return {"count": 0, "max": 0.0, "min": 0.0, "avg": 0.0}

    return {
        "count": float(len(scores)),
        "max": max(scores),
        "min": min(scores),
        "avg": round(sum(scores) / len(scores), 4),
    }
