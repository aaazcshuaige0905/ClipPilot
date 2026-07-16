from __future__ import annotations

from math import ceil
from pathlib import Path

from clippilot.core.exceptions import ClipPilotProcessingError
from clippilot.schemas.video_frames import SampledFrame, SampledFramesResult
from clippilot.schemas.video_info import VideoInfo

MAX_ADAPTIVE_FRAMES = 48
TARGET_SECONDS_PER_FRAME = 8.0


def _load_video_file_clip():
    """Load moviepy's VideoFileClip lazily so frame sampling shares the same dependency profile."""

    try:
        from moviepy import VideoFileClip as clip_class
    except ImportError:  # pragma: no cover
        try:
            from moviepy.video.io.VideoFileClip import VideoFileClip as clip_class
        except ImportError as exc:  # pragma: no cover
            raise ClipPilotProcessingError(
                "moviepy is not installed. Install dependencies before sampling video frames."
            ) from exc
    return clip_class


def _resolve_sample_frame_count(
    duration_seconds: float,
    requested_max_frames: int,
    *,
    hard_cap: int = MAX_ADAPTIVE_FRAMES,
    target_seconds_per_frame: float = TARGET_SECONDS_PER_FRAME,
) -> int:
    """Resolve an adaptive frame budget that still covers longer videos end-to-end."""

    if duration_seconds <= 0:
        return 1

    duration_driven_frames = max(1, ceil(duration_seconds / target_seconds_per_frame))
    return max(1, min(hard_cap, max(requested_max_frames, duration_driven_frames)))


def _build_sample_timestamps(duration_seconds: float, max_frames: int) -> list[float]:
    """Build evenly distributed sample timestamps that span the full source-video duration."""

    if duration_seconds <= 0:
        return [0.0]

    safe_duration = max(0.0, round(duration_seconds - 0.1, 2))
    if max_frames <= 1 or safe_duration <= 0:
        return [0.0]

    step = safe_duration / float(max_frames - 1)
    timestamps = [round(min(safe_duration, index * step), 2) for index in range(max_frames)]
    timestamps[0] = 0.0
    timestamps[-1] = safe_duration
    return timestamps


def sample_video_frames(
    video_path: Path,
    output_dir: Path,
    video_info: VideoInfo,
    *,
    video_id: str | None = None,
    max_frames: int = 12,
    interval_seconds: float = 3.0,
) -> SampledFramesResult:
    """Sample a lightweight set of frames from the local source video for visual understanding."""

    if max_frames <= 0:
        raise ClipPilotProcessingError("max_frames must be greater than zero for frame sampling.")
    if interval_seconds <= 0:
        raise ClipPilotProcessingError("interval_seconds must be greater than zero for frame sampling.")

    output_dir.mkdir(parents=True, exist_ok=True)
    effective_frame_count = _resolve_sample_frame_count(video_info.duration_seconds, requested_max_frames=max_frames)
    timestamps = _build_sample_timestamps(video_info.duration_seconds, max_frames=effective_frame_count)

    try:
        video_file_clip = _load_video_file_clip()
        with video_file_clip(str(video_path)) as video_clip:
            sampled_frames: list[SampledFrame] = []
            for index, timestamp in enumerate(timestamps):
                frame_id = f"frame_{index:03d}"
                file_name = f"{frame_id}_{timestamp:07.2f}.jpg"
                frame_path = output_dir / file_name
                video_clip.save_frame(str(frame_path), t=timestamp)
                sampled_frames.append(
                    SampledFrame(
                        frame_id=frame_id,
                        timestamp=timestamp,
                        image_path=str(frame_path),
                        width=video_info.width,
                        height=video_info.height,
                        note=f"Sampled frame at {timestamp:.2f} seconds.",
                    )
                )
    except Exception as exc:  # pragma: no cover - depends on video decoding backend
        raise ClipPilotProcessingError("Failed to sample frames from the uploaded video.") from exc

    if not sampled_frames:
        raise ClipPilotProcessingError("Frame sampling produced no usable frames.")

    return SampledFramesResult(
        video_id=video_id or video_path.stem,
        strategy=(
            f"adaptive_uniform_full_coverage_{effective_frame_count}_frames_"
            f"target_{TARGET_SECONDS_PER_FRAME:.2f}s"
        ),
        frame_count=len(sampled_frames),
        frames=sampled_frames,
        warnings=[],
    )
