from clippilot.schemas.transcript import TranscriptResult, TranscriptSegment


def test_transcript_result_can_be_instantiated() -> None:
    """Ensure transcript schema stays stable for the workflow."""

    transcript = TranscriptResult(
        video_id="task123",
        segments=[TranscriptSegment(start=0.0, end=3.0, text="hello")],
        full_text="hello",
        provider="mock",
    )
    assert transcript.video_id == "task123"
