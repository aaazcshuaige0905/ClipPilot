from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from clippilot.core.exceptions import ClipPilotProcessingError
from clippilot.schemas.video_understanding import CandidateRefinementGroup, VideoWindow


def _resolve_video_path(project_root: Path, video_path: str) -> Path:
    candidate = Path(video_path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _video_to_data_url(video_path: Path, max_encoded_bytes: int) -> str:
    """Encode a compact MP4 proxy while enforcing a conservative request-size ceiling."""

    try:
        encoded = base64.b64encode(video_path.read_bytes())
    except OSError as exc:
        raise ClipPilotProcessingError(f"Failed to read proxy video: {video_path}") from exc
    if len(encoded) > max_encoded_bytes:
        raise ClipPilotProcessingError(
            f"Encoded proxy video exceeds the configured {max_encoded_bytes}-byte data-URL budget."
        )
    return f"data:video/mp4;base64,{encoded.decode('ascii')}"


def build_multimodal_video_messages(
    request_payload: dict[str, Any],
    video_window: VideoWindow | CandidateRefinementGroup,
    *,
    project_root: Path,
    max_encoded_bytes: int,
) -> list[dict[str, Any]]:
    """Convert a logical window request into OpenAI-compatible video and text content."""

    messages = request_payload.get("messages", [])
    if len(messages) < 2:
        raise ClipPilotProcessingError("Window-understanding request is missing system/user messages.")
    video_path = _resolve_video_path(project_root, video_window.proxy_video_path)
    user_content = messages[1].get("content", {})
    text_payload = json.dumps(user_content, ensure_ascii=False, indent=2)
    video_item = {
        "type": "video_url",
        "video_url": {"url": _video_to_data_url(video_path, max_encoded_bytes)},
        "fps": float(request_payload.get("video_fps", 2.0)),
        "min_pixels": int(request_payload.get("min_pixels", 65536)),
        "max_pixels": int(request_payload.get("max_pixels", 262144)),
    }
    return [
        {"role": messages[0]["role"], "content": messages[0]["content"]},
        {
            "role": "user",
            "content": [
                video_item,
                {
                    "type": "text",
                    "text": (
                        "Analyze the attached proxy video using this source-aligned structured context.\n"
                        f"{text_payload}"
                    ),
                },
            ],
        },
    ]
