from clippilot.schemas.transcript import TranscriptResult


def summarize_video_understanding(transcript: TranscriptResult) -> str:
    """Return a placeholder summary string for future agentic video understanding."""

    return f"Video understanding agent is not enabled yet for task {transcript.video_id}."
