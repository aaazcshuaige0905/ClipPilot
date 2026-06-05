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
    rag_enabled: bool
    rag_knowledge_dir: Path
    rag_data_dir: Path
    rag_chroma_dir: Path
    rag_bm25_dir: Path
    rag_manifest_dir: Path
    rag_collection_name: str
    rag_top_k_dense: int
    rag_top_k_bm25: int
    rag_top_k_final: int
    rag_dense_weight: float
    rag_bm25_weight: float
    rag_metadata_weight: float
    qwen_embedding_model: str
    qwen_embedding_dimensions: int
    qwen_embedding_base_url: str
    qwen_embedding_api_path: str
    qwen_api_key: str


@dataclass(frozen=True)
class TaskPaths:
    """Store structured artifact paths for a single task."""

    task_root: Path
    input_dir: Path
    audio_dir: Path
    metadata_dir: Path
    transcript_dir: Path
    highlights_dir: Path
    understanding_dir: Path
    clips_dir: Path
    final_dir: Path
    plan_dir: Path
    plan_versions_dir: Path
    review_dir: Path
    trace_dir: Path
    source_video_path: Path
    audio_path: Path
    video_info_path: Path
    transcript_json_path: Path
    timeline_path: Path
    retrieved_context_path: Path
    retrieval_trace_path: Path
    highlight_candidates_path: Path
    editing_plan_path: Path
    planner_memory_path: Path
    execution_report_path: Path
    review_report_path: Path
    final_video_path: Path
    subtitle_path: Path
    burned_video_path: Path
    project_state_path: Path
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
        "rag": {
            "enabled": True,
            "knowledge_dir": "clippilot/rag/knowledge",
            "data_dir": "data/rag",
            "collection_name": "clip_pilot_strategy",
            "top_k_dense": 8,
            "top_k_bm25": 8,
            "top_k_final": 3,
            "dense_weight": 0.55,
            "bm25_weight": 0.30,
            "metadata_weight": 0.15,
        },
        "embedding": {
            "qwen_model": "text-embedding-v4",
            "qwen_dimensions": 1024,
            "qwen_base_url": "https://dashscope.aliyuncs.com",
            "qwen_embedding_api_path": "/api/v1/services/embeddings/text-embedding/text-embedding",
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
    rag_config = config["rag"]
    embedding_config = config["embedding"]
    qwen_api_key = os.getenv("DASHSCOPE_API_KEY", os.getenv("CLIP_PILOT_QWEN_API_KEY", "")).strip()
    qwen_base_url = os.getenv("CLIP_PILOT_QWEN_EMBEDDING_BASE_URL", embedding_config["qwen_base_url"]).strip()
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
        rag_enabled=bool(rag_config["enabled"]),
        rag_knowledge_dir=resolved_root / rag_config["knowledge_dir"],
        rag_data_dir=resolved_root / rag_config["data_dir"],
        rag_chroma_dir=(resolved_root / rag_config["data_dir"]) / "chroma",
        rag_bm25_dir=(resolved_root / rag_config["data_dir"]) / "bm25",
        rag_manifest_dir=(resolved_root / rag_config["data_dir"]) / "manifests",
        rag_collection_name=str(rag_config["collection_name"]),
        rag_top_k_dense=int(rag_config["top_k_dense"]),
        rag_top_k_bm25=int(rag_config["top_k_bm25"]),
        rag_top_k_final=int(rag_config["top_k_final"]),
        rag_dense_weight=float(rag_config["dense_weight"]),
        rag_bm25_weight=float(rag_config["bm25_weight"]),
        rag_metadata_weight=float(rag_config["metadata_weight"]),
        qwen_embedding_model=os.getenv("CLIP_PILOT_QWEN_EMBEDDING_MODEL", embedding_config["qwen_model"]).strip() or "text-embedding-v4",
        qwen_embedding_dimensions=int(os.getenv("CLIP_PILOT_QWEN_EMBEDDING_DIMENSIONS", str(embedding_config["qwen_dimensions"]))),
        qwen_embedding_base_url=qwen_base_url or "https://dashscope.aliyuncs.com",
        qwen_embedding_api_path=os.getenv("CLIP_PILOT_QWEN_EMBEDDING_API_PATH", embedding_config["qwen_embedding_api_path"]).strip()
        or "/api/v1/services/embeddings/text-embedding/text-embedding",
        qwen_api_key=qwen_api_key,
    )


def ensure_base_directories(settings: AppSettings) -> None:
    """Create application-wide directories needed before processing tasks."""

    settings.tasks_root_dir.mkdir(parents=True, exist_ok=True)
    settings.raw_videos_root.mkdir(parents=True, exist_ok=True)
    if settings.rag_enabled:
        settings.rag_data_dir.mkdir(parents=True, exist_ok=True)
        settings.rag_chroma_dir.mkdir(parents=True, exist_ok=True)
        settings.rag_bm25_dir.mkdir(parents=True, exist_ok=True)
        settings.rag_manifest_dir.mkdir(parents=True, exist_ok=True)


def build_task_paths(settings: AppSettings, task_id: str, original_file_name: str) -> TaskPaths:
    """Build all structured artifact paths for a single processing task."""

    source_name = original_file_name or "uploaded_video.mp4"
    suffix = Path(source_name).suffix.lower() or ".mp4"
    task_root = settings.tasks_root_dir / task_id
    input_dir = task_root / "input"
    audio_dir = task_root / "audio"
    metadata_dir = task_root / "metadata"
    transcript_dir = task_root / "transcript"
    highlights_dir = task_root / "highlights"
    understanding_dir = task_root / "understanding"
    clips_dir = task_root / "clips"
    final_dir = task_root / "final"
    plan_dir = task_root / "plan"
    plan_versions_dir = plan_dir / "versions"
    review_dir = task_root / "review"
    trace_dir = task_root / "trace"

    return TaskPaths(
        task_root=task_root,
        input_dir=input_dir,
        audio_dir=audio_dir,
        metadata_dir=metadata_dir,
        transcript_dir=transcript_dir,
        highlights_dir=highlights_dir,
        understanding_dir=understanding_dir,
        clips_dir=clips_dir,
        final_dir=final_dir,
        plan_dir=plan_dir,
        plan_versions_dir=plan_versions_dir,
        review_dir=review_dir,
        trace_dir=trace_dir,
        source_video_path=input_dir / f"source{suffix}",
        audio_path=audio_dir / "source.wav",
        video_info_path=metadata_dir / "video_info.json",
        transcript_json_path=transcript_dir / "transcript.json",
        timeline_path=understanding_dir / "timeline.json",
        retrieved_context_path=understanding_dir / "retrieved_context.json",
        retrieval_trace_path=understanding_dir / "retrieval_trace.json",
        highlight_candidates_path=highlights_dir / "candidates.json",
        editing_plan_path=plan_dir / "editing_plan.json",
        planner_memory_path=plan_dir / "planner_memory.json",
        execution_report_path=plan_dir / "execution_report.json",
        review_report_path=review_dir / "review_report.json",
        final_video_path=final_dir / "final_video.mp4",
        subtitle_path=final_dir / "subtitles.srt",
        burned_video_path=final_dir / "final_video_burned.mp4",
        project_state_path=task_root / "project_state.json",
        task_result_path=task_root / "task_result.json",
        artifact_manifest_path=task_root / "artifact_manifest.json",
        trace_log_path=trace_dir / "workflow_trace.jsonl",
    )
