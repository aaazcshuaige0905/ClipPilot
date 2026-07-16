from __future__ import annotations

from math import ceil
from pathlib import Path
import shutil
import subprocess

from clippilot.core.exceptions import ClipPilotProcessingError
from clippilot.schemas.project_state import FineGrainedUnit
from clippilot.schemas.video_info import VideoInfo
from clippilot.schemas.video_understanding import VideoWindow


def _build_window_ranges(
    duration_seconds: float,
    window_duration_seconds: float = 48.0,
    overlap_seconds: float = 3.0,
) -> list[tuple[float, float]]:
    """Build overlapping ranges that cover the source duration without time gaps."""

    if duration_seconds <= 0:
        return []
    if window_duration_seconds <= 0:
        raise ClipPilotProcessingError("window_duration_seconds must be positive.")
    if overlap_seconds < 0 or overlap_seconds >= window_duration_seconds:
        raise ClipPilotProcessingError("overlap_seconds must be non-negative and smaller than the window duration.")

    stride = window_duration_seconds - overlap_seconds
    count = max(1, ceil(max(0.0, duration_seconds - window_duration_seconds) / stride) + 1)
    ranges: list[tuple[float, float]] = []
    for index in range(count):
        start = round(index * stride, 2)
        end = round(min(duration_seconds, start + window_duration_seconds), 2)
        if start >= duration_seconds:
            break
        ranges.append((start, end))
    if ranges and ranges[-1][1] < duration_seconds - 0.01:
        last_start = round(max(0.0, duration_seconds - window_duration_seconds), 2)
        ranges.append((last_start, round(duration_seconds, 2)))
    return ranges


def _render_proxy_window(
    source_video_path: Path,
    output_path: Path,
    start: float,
    end: float,
    *,
    proxy_height: int = 360,
    crf: int = 30,
) -> Path:
    """Render one compact, source-aligned, audio-free proxy window with ffmpeg."""

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise ClipPilotProcessingError("ffmpeg is required to create video-understanding windows.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source_video_path),
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-vf",
        f"scale=-2:{proxy_height}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0 or not output_path.exists():
        error = completed.stderr.strip() or "Unknown ffmpeg error."
        raise ClipPilotProcessingError(f"Failed to render proxy window: {error}")
    return output_path


def create_video_windows(
    source_video_path: Path,
    output_dir: Path,
    video_info: VideoInfo,
    fine_grained_units: list[FineGrainedUnit],
    *,
    window_duration_seconds: float = 48.0,
    overlap_seconds: float = 3.0,
    proxy_height: int = 360,
    max_encoded_bytes: int = 9_000_000,
) -> list[VideoWindow]:
    """Create every proxy video and retain its mapping to source transcript units."""

    output_dir.mkdir(parents=True, exist_ok=True)
    windows: list[VideoWindow] = []
    for order, (start, end) in enumerate(
        _build_window_ranges(video_info.duration_seconds, window_duration_seconds, overlap_seconds),
        start=1,
    ):
        window_id = f"window_{order:03d}"
        proxy_path = output_dir / f"{window_id}_{start:07.2f}_{end:07.2f}.mp4"
        _render_proxy_window(
            source_video_path,
            proxy_path,
            start,
            end,
            proxy_height=proxy_height,
        )
        warnings: list[str] = []
        estimated_encoded_bytes = ceil(proxy_path.stat().st_size / 3) * 4
        if estimated_encoded_bytes > max_encoded_bytes:
            _render_proxy_window(
                source_video_path,
                proxy_path,
                start,
                end,
                proxy_height=min(proxy_height, 240),
                crf=35,
            )
            warnings.append("Proxy was re-encoded at lower resolution to satisfy the data-URL budget.")
        estimated_encoded_bytes = ceil(proxy_path.stat().st_size / 3) * 4
        if estimated_encoded_bytes > max_encoded_bytes:
            warnings.append("Proxy still exceeds the data-URL budget; this window will use local fallback.")
        unit_ids = [
            unit.unit_id
            for unit in fine_grained_units
            if unit.end > start + 0.01 and unit.start < end - 0.01
        ]
        windows.append(
            VideoWindow(
                window_id=window_id,
                order=order,
                source_start=start,
                source_end=end,
                duration=round(end - start, 2),
                proxy_video_path=str(proxy_path),
                source_unit_ids=unit_ids,
                warnings=warnings,
            )
        )
    return windows
