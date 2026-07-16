from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

from clippilot.schemas.editing_plan import EditingPlan


@dataclass(frozen=True)
class SubtitleGenerationResult:
    """Represent the outcome of generating an SRT file from an editing plan."""

    subtitle_path: str | None
    subtitle_count: int
    success: bool


@dataclass(frozen=True)
class SubtitleBurnResult:
    """Represent the outcome of burning subtitles into a rendered video."""

    success: bool
    burned_video_path: str
    error: str | None = None


def build_subtitle_file(task_id: str, output_dir: Path) -> Path:
    """Return the future subtitle output path for a task."""

    return output_dir / f"{task_id}.srt"


def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds into standard SRT timestamp format."""

    total_milliseconds = max(0, int(round(seconds * 1000)))
    hours = total_milliseconds // 3_600_000
    remainder = total_milliseconds % 3_600_000
    minutes = remainder // 60_000
    remainder %= 60_000
    secs = remainder // 1000
    milliseconds = remainder % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def _normalize_editing_plan(editing_plan: EditingPlan | dict) -> EditingPlan:
    """Normalize an editing plan payload into a validated Pydantic model."""

    if isinstance(editing_plan, EditingPlan):
        return editing_plan
    return EditingPlan.model_validate(editing_plan)


def generate_srt_from_editing_plan(
    editing_plan: EditingPlan | dict,
    output_srt_path: str,
) -> SubtitleGenerationResult:
    """Generate an SRT file based on the rendered timeline items defined by the editing plan."""

    plan = _normalize_editing_plan(editing_plan)
    output_path = Path(output_srt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    timeline_cursor = 0.0
    subtitle_count = 0

    for index, item in enumerate(plan.timeline_items, start=1):
        start_time = timeline_cursor
        end_time = round(start_time + item.duration, 3)
        subtitle_text = str(item.subtitle or "").strip()
        if not subtitle_text:
            timeline_cursor = end_time
            continue

        lines.extend(
            [
                str(subtitle_count + 1),
                f"{seconds_to_srt_time(start_time)} --> {seconds_to_srt_time(end_time)}",
                subtitle_text,
                "",
            ]
        )
        timeline_cursor = end_time
        subtitle_count += 1

    if not lines:
        return SubtitleGenerationResult(
            subtitle_path=None,
            subtitle_count=0,
            success=True,
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return SubtitleGenerationResult(
        subtitle_path=str(output_path),
        subtitle_count=subtitle_count,
        success=True,
    )


def _to_ffmpeg_subtitle_filter_path(path: Path) -> str:
    """Convert a subtitle file path into a Windows-safe ffmpeg subtitles filter value."""

    normalized = path.resolve().as_posix()
    normalized = normalized.replace(":", "\\:")
    normalized = normalized.replace("'", r"\'")
    return normalized


def burn_subtitles_to_video(
    input_video_path: str,
    subtitle_path: str,
    output_video_path: str,
) -> SubtitleBurnResult:
    """Burn an SRT subtitle file into a video by calling ffmpeg."""

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return SubtitleBurnResult(
            success=False,
            burned_video_path=output_video_path,
            error="ffmpeg is not installed or not available in PATH.",
        )

    input_path = Path(input_video_path)
    subtitle_file = Path(subtitle_path)
    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subtitle_filter = f"subtitles='{_to_ffmpeg_subtitle_filter_path(subtitle_file)}'"
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        subtitle_filter,
        "-c:a",
        "copy",
        str(output_path),
    ]

    try:
        completed_process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return SubtitleBurnResult(
            success=False,
            burned_video_path=str(output_path),
            error=f"Failed to execute ffmpeg: {exc}",
        )

    if completed_process.returncode != 0:
        stderr = completed_process.stderr.strip() or "Unknown ffmpeg error."
        return SubtitleBurnResult(
            success=False,
            burned_video_path=str(output_path),
            error=stderr,
        )

    return SubtitleBurnResult(
        success=True,
        burned_video_path=str(output_path),
        error=None,
    )
