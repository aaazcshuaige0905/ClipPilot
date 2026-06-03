from pathlib import Path

from clippilot.tools.video_info import validate_supported_extension


def test_validate_supported_extension_accepts_mp4() -> None:
    """Ensure allowed extensions pass validation."""

    assert validate_supported_extension("demo.mp4", {".mp4", ".mov"}) == ".mp4"
