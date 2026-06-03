from dataclasses import dataclass
from pathlib import Path
import os

import yaml


@dataclass(frozen=True)
class AppSettings:
    """Store normalized application settings loaded from config.yaml and environment variables."""

    project_root: Path
    config_path: Path
    app_name: str
    app_version: str
    app_description: str
    tasks_root_dir: Path
    raw_videos_root: Path
    allowed_extensions: set[str]
    min_video_duration_seconds: int
    max_video_duration_seconds: int
    asr_provider: str
    whisper_model: str
    highlight_min_candidate_duration: float
    highlight_max_candidate_duration: float


@dataclass(frozen=True)
class TaskPaths:
    """Store structured artifact paths for a single task."""

    task_root: Path
    input_dir: Path
    metadata_dir: Path
    transcript_dir: Path
    highlights_dir: Path
    clips_dir: Path
    final_dir: Path
    plan_dir: Path
    review_dir: Path
    trace_dir: Path
    source_video_path: Path
    video_info_path: Path
    transcript_json_path: Path
    highlight_candidates_path: Path
    editing_plan_path: Path
    execution_report_path: Path
    review_report_path: Path
    final_video_path: Path
    subtitle_path: Path
    burned_video_path: Path
    task_result_path: Path
    artifact_manifest_path: Path
    trace_log_path: Path


def _default_config() -> dict:
    """Return default configuration values used when config.yaml is missing fields."""

    return {
        "app": {
            "name": "clip-pilot",
            "version": "0.1.0",
            "description": "MVP API for AI short video highlight upload tasks.",
        },
        "storage": {
            "tasks_root": "outputs/tasks",
            "raw_videos_root": "data/raw_videos",
        },
        "video": {
            "allowed_extensions": [".mp4", ".mov", ".mkv"],
            "min_duration_seconds": 180,
            "max_duration_seconds": 600,
        },
        "asr": {
            "provider": "mock",
            "whisper_model": "base",
        },
        "highlight": {
            "min_candidate_duration": 8.0,
            "max_candidate_duration": 20.0,
        },
    }


def _merge_defaults(defaults: dict, overrides: dict) -> dict:
    """Recursively merge a partially provided config dictionary into defaults."""

    result = dict(defaults)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_defaults(result[key], value)
        else:
            result[key] = value
    return result


def load_settings(project_root: Path | None = None) -> AppSettings:
    """Load application settings from config.yaml and environment variables."""

    resolved_root = project_root or Path(__file__).resolve().parents[2]
    config_path = resolved_root / "config.yaml"
    defaults = _default_config()
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    config = _merge_defaults(defaults, loaded or {})

    storage_config = config["storage"]
    video_config = config["video"]
    asr_config = config["asr"]
    highlight_config = config["highlight"]
    return AppSettings(
        project_root=resolved_root,
        config_path=config_path,
        app_name=config["app"]["name"],
        app_version=config["app"]["version"],
        app_description=config["app"]["description"],
        tasks_root_dir=resolved_root / storage_config["tasks_root"],
        raw_videos_root=resolved_root / storage_config["raw_videos_root"],
        allowed_extensions={item.lower() for item in video_config["allowed_extensions"]},
        min_video_duration_seconds=int(video_config["min_duration_seconds"]),
        max_video_duration_seconds=int(video_config["max_duration_seconds"]),
        asr_provider=os.getenv("CLIP_PILOT_ASR_PROVIDER", asr_config["provider"]).strip().lower() or "mock",
        whisper_model=os.getenv("CLIP_PILOT_WHISPER_MODEL", asr_config["whisper_model"]).strip() or "base",
        highlight_min_candidate_duration=float(highlight_config["min_candidate_duration"]),
        highlight_max_candidate_duration=float(highlight_config["max_candidate_duration"]),
    )


def ensure_base_directories(settings: AppSettings) -> None:
    """Create application-wide directories needed before processing tasks."""

    settings.tasks_root_dir.mkdir(parents=True, exist_ok=True)
    settings.raw_videos_root.mkdir(parents=True, exist_ok=True)


def build_task_paths(settings: AppSettings, task_id: str, original_file_name: str) -> TaskPaths:
    """Build all structured artifact paths for a single processing task."""

    source_name = original_file_name or "uploaded_video.mp4"
    suffix = Path(source_name).suffix.lower() or ".mp4"
    task_root = settings.tasks_root_dir / task_id
    input_dir = task_root / "input"
    metadata_dir = task_root / "metadata"
    transcript_dir = task_root / "transcript"
    highlights_dir = task_root / "highlights"
    clips_dir = task_root / "clips"
    final_dir = task_root / "final"
    plan_dir = task_root / "plan"
    review_dir = task_root / "review"
    trace_dir = task_root / "trace"

    return TaskPaths(
        task_root=task_root,
        input_dir=input_dir,
        metadata_dir=metadata_dir,
        transcript_dir=transcript_dir,
        highlights_dir=highlights_dir,
        clips_dir=clips_dir,
        final_dir=final_dir,
        plan_dir=plan_dir,
        review_dir=review_dir,
        trace_dir=trace_dir,
        source_video_path=input_dir / f"source{suffix}",
        video_info_path=metadata_dir / "video_info.json",
        transcript_json_path=transcript_dir / "transcript.json",
        highlight_candidates_path=highlights_dir / "candidates.json",
        editing_plan_path=plan_dir / "editing_plan.json",
        execution_report_path=plan_dir / "execution_report.json",
        review_report_path=review_dir / "review_report.json",
        final_video_path=final_dir / "final_video.mp4",
        subtitle_path=final_dir / "subtitles.srt",
        burned_video_path=final_dir / "final_video_burned.mp4",
        task_result_path=task_root / "task_result.json",
        artifact_manifest_path=task_root / "artifact_manifest.json",
        trace_log_path=trace_dir / "workflow_trace.jsonl",
    )
