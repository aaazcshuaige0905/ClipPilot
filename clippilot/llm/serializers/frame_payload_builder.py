from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from clippilot.core.exceptions import ClipPilotProcessingError
from clippilot.schemas.video_frames import SampledFramesResult


def _resolve_frame_path(project_root: Path, image_path: str) -> Path:
    """Resolve one saved frame path from task-relative form into a readable filesystem path."""

    candidate = Path(image_path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _image_to_data_url(frame_path: Path) -> str:
    """Encode one local frame file into a base64 data URL for multimodal provider requests."""

    suffix = frame_path.suffix.lower()
    mime_type = "image/png" if suffix == ".png" else "image/jpeg"
    try:
        encoded = base64.b64encode(frame_path.read_bytes()).decode("utf-8")
    except OSError as exc:
        raise ClipPilotProcessingError(f"Failed to read sampled frame: {frame_path}") from exc
    return f"data:{mime_type};base64,{encoded}"


def build_multimodal_frame_messages(
    request_payload: dict[str, Any],
    sampled_frames: SampledFramesResult,
    *,
    project_root: Path,
) -> list[dict[str, Any]]:
    """Convert a logical video-understanding request into multimodal chat messages with inline frames."""

    messages = request_payload.get("messages", [])
    if len(messages) < 2:
        raise ClipPilotProcessingError("Video-understanding request payload is missing required system/user messages.")

    system_message = messages[0]
    user_content = dict(messages[1].get("content", {}))
    frame_descriptors = user_content.pop("sampled_frames", [])
    text_payload = json.dumps(user_content, ensure_ascii=False, indent=2)

    multimodal_user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Use the following structured context together with the attached sampled frames to analyze the video.\n"
                f"{text_payload}"
            ),
        }
    ]

    for frame_descriptor in frame_descriptors[: sampled_frames.frame_count]:
        resolved_path = _resolve_frame_path(project_root, frame_descriptor["image_path"])
        multimodal_user_content.append(
            {
                "type": "text",
                "text": (
                    f"Frame {frame_descriptor['frame_id']} captured at {frame_descriptor['timestamp']} seconds. "
                    f"Note: {frame_descriptor.get('note') or 'No extra note.'}"
                ),
            }
        )
        multimodal_user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _image_to_data_url(resolved_path),
                    "detail": "low",
                },
            }
        )

    return [
        {
            "role": system_message["role"],
            "content": system_message["content"],
        },
        {
            "role": "user",
            "content": multimodal_user_content,
        },
    ]
