from clippilot.schemas.transcript import TranscriptResult, TranscriptSegment
from clippilot.tools.highlight import generate_highlight_candidates


def test_generate_highlight_candidates_returns_candidates() -> None:
    """Ensure the rule-based highlight generator can rank a simple transcript."""

    transcript = TranscriptResult(
        video_id="task123",
        segments=[
            TranscriptSegment(start=0.0, end=5.0, text="先说结论，这里最重要的是理解核心问题。"),
            TranscriptSegment(start=5.0, end=11.0, text="为什么很多人会失败，但是真正的原因常常被忽略。"),
        ],
        full_text="先说结论，这里最重要的是理解核心问题。 为什么很多人会失败，但是真正的原因常常被忽略。",
        provider="mock",
    )
    result = generate_highlight_candidates(transcript, {"edit_style": "powerful"})
    assert result.candidates
