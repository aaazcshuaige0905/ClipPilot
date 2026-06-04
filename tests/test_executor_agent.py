from pathlib import Path
import shutil
import subprocess
import json

import pytest

from clippilot.agents import executor_agent
from clippilot.agents.executor_agent import execute_editing_plan
from clippilot.schemas.editing_plan import EditingClip, EditingPlan
from clippilot.storage.path_manager import AppSettings
from clippilot.storage.task_storage import TaskStorage


def _workspace_root(test_name: str) -> Path:
    """Return a writable workspace-local directory for filesystem tests."""

    root = Path("tests_runtime") / test_name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ffmpeg_available() -> bool:
    """Return whether ffmpeg is available in the current test environment."""

    return shutil.which("ffmpeg") is not None


def _build_test_settings(root: Path) -> AppSettings:
    """Create isolated settings for executor tests."""

    return AppSettings(
        project_root=root,
        config_path=root / "config.yaml",
        app_name="clip-pilot",
        app_version="0.1.0",
        app_description="test",
        tasks_root_dir=root / "outputs" / "tasks",
        raw_videos_root=root / "data" / "raw_videos",
        allowed_extensions={".mp4", ".mov", ".mkv"},
        min_video_duration_seconds=180,
        max_video_duration_seconds=600,
        asr_provider="mock",
        whisper_model="base",
        highlight_min_candidate_duration=8.0,
        highlight_max_candidate_duration=20.0,
        rag_enabled=True,
        rag_knowledge_dir=root / "clippilot" / "rag" / "knowledge",
        rag_data_dir=root / "data" / "rag",
        rag_chroma_dir=root / "data" / "rag" / "chroma",
        rag_bm25_dir=root / "data" / "rag" / "bm25",
        rag_manifest_dir=root / "data" / "rag" / "manifests",
        rag_collection_name="clip_pilot_strategy",
        rag_top_k_dense=8,
        rag_top_k_bm25=8,
        rag_top_k_final=3,
        rag_dense_weight=0.55,
        rag_bm25_weight=0.30,
        rag_metadata_weight=0.15,
        qwen_embedding_model="text-embedding-v4",
        qwen_embedding_dimensions=1024,
        qwen_embedding_base_url="https://dashscope.aliyuncs.com",
        qwen_embedding_api_path="/api/v1/services/embeddings/text-embedding/text-embedding",
        qwen_api_key="",
    )


def _create_sample_video(output_path: Path) -> None:
    """Create a small integration-test video with ffmpeg test sources."""

    command = [
        shutil.which("ffmpeg") or "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=320x240:rate=24",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=44100",
        "-t",
        "7",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def _build_editing_plan() -> EditingPlan:
    """Create a minimal executable editing plan fixture."""

    return EditingPlan(
        task_id="task123",
        target_duration=10,
        total_duration=6.0,
        clips=[
            EditingClip(
                clip_id="clip_01",
                source_start=0.0,
                source_end=3.0,
                duration=3.0,
                purpose="hook",
                text="Opening hook.",
                subtitle="Opening hook.",
                score=0.9,
                reason="Strong opener.",
            ),
            EditingClip(
                clip_id="clip_02",
                source_start=3.0,
                source_end=6.0,
                duration=3.0,
                purpose="core_point",
                text="Main point.",
                subtitle="Main point.",
                score=0.82,
                reason="Key supporting point.",
            ),
        ],
        editing_notes=["Executor test plan."],
    )


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg is not installed")
def test_execute_editing_plan_generates_final_outputs() -> None:
    """Ensure the executor can cut, merge, subtitle, and report final outputs."""

    settings = _build_test_settings(_workspace_root("executor_success"))
    storage = TaskStorage(settings)
    task_paths = storage.create_task_paths("task123", "source.mp4")
    _create_sample_video(task_paths.source_video_path)

    report = execute_editing_plan(
        task_id="task123",
        source_video_path=str(task_paths.source_video_path),
        editing_plan=_build_editing_plan(),
        task_paths=task_paths,
        need_burn_subtitle=False,
    )

    assert report.status == "completed"
    assert report.final_video_path is not None
    assert Path(report.final_video_path).exists()
    assert report.subtitle_path is not None
    assert Path(report.subtitle_path).exists()
    assert report.clip_results

    report_path = storage.save_execution_report(report, task_paths)
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report["final_video_path"] == report.final_video_path
    assert saved_report["subtitle_path"] == report.subtitle_path


def test_execute_editing_plan_collects_errors_without_crashing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure execution failures are recorded into the report instead of crashing the workflow."""

    settings = _build_test_settings(_workspace_root("executor_failure"))
    storage = TaskStorage(settings)
    task_paths = storage.create_task_paths("task123", "source.mp4")

    def _fake_cut_video_clip(input_video_path: str, output_clip_path: str, start_time: float, end_time: float) -> object:
        raise RuntimeError("Simulated cut failure.")

    monkeypatch.setattr(executor_agent, "cut_video_clip", _fake_cut_video_clip)

    report = execute_editing_plan(
        task_id="task123",
        source_video_path=str(task_paths.source_video_path),
        editing_plan=_build_editing_plan(),
        task_paths=task_paths,
        need_burn_subtitle=False,
    )

    assert report.status == "failed"
    assert report.errors
    assert "Simulated cut failure." in report.errors[0]
    assert report.final_video_path is not None
    assert report.subtitle_path is not None
