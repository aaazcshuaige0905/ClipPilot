from pathlib import Path
import sys
import types

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

_fake_moviepy = types.ModuleType("moviepy")


class _FakeVideoFileClip:
    """Provide a lightweight moviepy stand-in for API unit tests."""


_fake_moviepy.VideoFileClip = _FakeVideoFileClip
sys.modules.setdefault("moviepy", _fake_moviepy)
sys.modules.setdefault("moviepy.video", types.ModuleType("moviepy.video"))
sys.modules.setdefault("moviepy.video.io", types.ModuleType("moviepy.video.io"))
_fake_moviepy_video_file_clip = types.ModuleType("moviepy.video.io.VideoFileClip")
_fake_moviepy_video_file_clip.VideoFileClip = _FakeVideoFileClip
sys.modules.setdefault("moviepy.video.io.VideoFileClip", _fake_moviepy_video_file_clip)

from clippilot.agents.planner_agent import build_editing_plan
from clippilot.api import app as app_module
from clippilot.core.memory_manager import build_initial_planner_memory, record_plan_generation
from clippilot.rag.schemas import RetrievedChunk, RetrievedContext, RetrievalQuery
from clippilot.schemas.project_state import (
    FineGrainedUnit,
    HighlightScoreBreakdown,
    LLMHighlightCandidate,
    ProjectPaths,
    ProjectState,
    TimelineSegment,
    TrimPolicy,
    VideoTimeline,
)
from clippilot.schemas.task_result import TaskArtifactManifest, TaskResult
from clippilot.schemas.transcript import TranscriptResult, TranscriptSegment
from clippilot.schemas.editing_plan import HighlightCandidate, HighlightCandidatesResult
from clippilot.schemas.user_request import UserRequest
from clippilot.schemas.video_info import VideoInfo
from clippilot.storage.path_manager import AppSettings
from clippilot.storage.task_storage import TaskStorage


def _workspace_root(test_name: str) -> Path:
    """Return a writable workspace-local directory for filesystem tests."""

    root = Path("tests_runtime") / test_name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build_test_settings(root: Path) -> AppSettings:
    """Create isolated settings for revision API tests."""

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


def _build_user_request() -> UserRequest:
    """Create a reusable request fixture for revision API tests."""

    return UserRequest(
        target_platform="bilibili",
        target_duration=30,
        edit_style="powerful",
        language="zh",
        need_burn_subtitle=True,
    )


def _build_video_info() -> VideoInfo:
    """Create reusable source video metadata for revision API tests."""

    return VideoInfo(
        duration_seconds=320.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
        file_size_mb=128.0,
        source_path="outputs/tasks/task123/input/source.mp4",
    )


def _build_transcript() -> TranscriptResult:
    """Create a transcript fixture for planner and revision flows."""

    return TranscriptResult(
        video_id="task123",
        provider="mock",
        full_text="Opening claim. Supporting proof. Final takeaway.",
        segments=[
            TranscriptSegment(start=0.0, end=4.0, text="Opening claim."),
            TranscriptSegment(start=4.0, end=8.0, text="Supporting proof."),
            TranscriptSegment(start=8.0, end=12.0, text="Final takeaway."),
        ],
    )


def _build_fine_grained_units() -> list[FineGrainedUnit]:
    """Create fine-grained transcript units for planner inputs."""

    return [
        FineGrainedUnit(
            unit_id="unit_001",
            start=0.0,
            end=3.0,
            duration=3.0,
            text="Opening claim with urgency.",
            has_pause_after=True,
            keywords=["opening", "claim"],
        ),
        FineGrainedUnit(
            unit_id="unit_002",
            start=3.3,
            end=7.0,
            duration=3.7,
            text="Supporting proof with numbers.",
            has_pause_before=True,
            keywords=["proof", "numbers"],
        ),
        FineGrainedUnit(
            unit_id="unit_003",
            start=8.0,
            end=11.5,
            duration=3.5,
            text="Final takeaway and CTA.",
            has_pause_before=True,
            keywords=["takeaway", "cta"],
        ),
    ]


def _build_llm_candidates() -> list[LLMHighlightCandidate]:
    """Create transcript-backed highlight candidates for revision API tests."""

    return [
        LLMHighlightCandidate(
            candidate_id="cand_hook",
            source_segment_ids=["seg_01"],
            transcript_unit_ids=["unit_001", "unit_002"],
            source_start=0.0,
            source_end=7.0,
            duration=7.0,
            title="Fast opener",
            summary="Start with the strongest payoff.",
            transcript_excerpt="Opening claim with urgency. Supporting proof with numbers.",
            highlight_type="insight",
            semantic_role="opening_hook",
            reason="High hook potential.",
            scores=HighlightScoreBreakdown(
                overall=0.92,
                hook=0.95,
                emotion=0.62,
                clarity=0.89,
                platform_fit=0.91,
                editability=0.88,
            ),
            trim_policy=TrimPolicy(
                min_duration=3.0,
                ideal_duration=6.0,
                max_duration=7.0,
                preferred_start=0.0,
                preferred_end=6.0,
            ),
        ),
        LLMHighlightCandidate(
            candidate_id="cand_ending",
            source_segment_ids=["seg_03"],
            transcript_unit_ids=["unit_003"],
            source_start=8.0,
            source_end=11.5,
            duration=3.5,
            title="CTA close",
            summary="Close with a direct takeaway.",
            transcript_excerpt="Final takeaway and CTA.",
            highlight_type="takeaway",
            semantic_role="ending",
            reason="Strong closer aligned with platform pacing.",
            scores=HighlightScoreBreakdown(
                overall=0.82,
                hook=0.35,
                emotion=0.52,
                clarity=0.84,
                platform_fit=0.82,
                editability=0.8,
            ),
            trim_policy=TrimPolicy(
                min_duration=2.5,
                ideal_duration=3.5,
                max_duration=3.5,
                preferred_start=8.0,
                preferred_end=11.5,
            ),
        ),
    ]


def _build_timeline() -> VideoTimeline:
    """Create a semantic timeline for revision API tests."""

    return VideoTimeline(
        provider="qwen",
        model="stub",
        overview="A two-part narrative with a hook and CTA.",
        segments=[
            TimelineSegment(
                segment_id="seg_01",
                start=0.0,
                end=7.0,
                duration=7.0,
                title="Opening setup",
                summary="Strong claim and immediate payoff.",
                transcript_excerpt="Opening claim with urgency.",
                source_unit_ids=["unit_001", "unit_002"],
                importance=0.94,
                hook_score=0.93,
            ),
            TimelineSegment(
                segment_id="seg_03",
                start=8.0,
                end=11.5,
                duration=3.5,
                title="Ending CTA",
                summary="Direct actionable close.",
                transcript_excerpt="Final takeaway and CTA.",
                source_unit_ids=["unit_003"],
                importance=0.8,
                hook_score=0.44,
            ),
        ],
    )


def _build_retrieved_context() -> RetrievedContext:
    """Create retrieved rules for revision API tests."""

    return RetrievedContext(
        query=RetrievalQuery(text="platform bilibili | duration 30 seconds"),
        chunks=[
            RetrievedChunk(
                chunk_id="rule_hook",
                text="Lead with a payoff in the first three seconds.",
                title="Hook Rule",
                source_file="platform_rules.md",
                knowledge_type="rule",
                strategy_type="hook_rule",
                platform="bilibili",
                language="zh",
                style="powerful",
                duration_band="15_30",
                stage="planning",
                priority="must",
                final_score=0.9,
                retrieval_sources=["dense"],
            )
        ],
    )


def _build_candidates_result() -> HighlightCandidatesResult:
    """Create stored rule-based candidates for the planner contract."""

    return HighlightCandidatesResult(
        video_id="task123",
        candidates=[
            HighlightCandidate(
                start=0.0,
                end=7.0,
                text="Opening claim with urgency. Supporting proof with numbers.",
                score=0.92,
                reason="Strong opening clip.",
            ),
            HighlightCandidate(
                start=8.0,
                end=11.5,
                text="Final takeaway and CTA.",
                score=0.82,
                reason="Concise closer.",
            ),
        ],
    )


def _seed_completed_task(settings: AppSettings) -> TaskStorage:
    """Seed one completed task so the revision API can operate on realistic artifacts."""

    storage = TaskStorage(settings)
    task_paths = storage.create_task_paths("task123", "source.mp4")
    user_request = _build_user_request()
    planner_memory = build_initial_planner_memory("task123", user_request)
    project_state = ProjectState(
        project_id="task123",
        user_request=user_request,
        paths=ProjectPaths(raw_video="outputs/tasks/task123/input/source.mp4"),
        fine_grained_units=_build_fine_grained_units(),
        retrieved_context=_build_retrieved_context(),
        timeline=_build_timeline(),
        highlight_candidates_llm=_build_llm_candidates(),
        status="COMPLETED",
        current_step="completed",
    )
    initial_plan = build_editing_plan(
        task_id="task123",
        user_request=user_request,
        video_info=_build_video_info(),
        transcript=_build_transcript(),
        candidates=_build_candidates_result(),
        fine_grained_units=project_state.fine_grained_units,
        timeline=project_state.timeline,
        llm_candidates=project_state.highlight_candidates_llm,
        retrieved_context=project_state.retrieved_context,
        planner_memory=planner_memory,
    )
    record_plan_generation(planner_memory, initial_plan)

    storage.save_project_state(project_state, task_paths)
    storage.save_planner_memory(planner_memory, task_paths)
    storage.save_video_info(_build_video_info(), task_paths)
    storage.save_transcript(_build_transcript(), task_paths)
    storage.save_candidates(_build_candidates_result(), task_paths)
    storage.save_editing_plan(initial_plan, task_paths)
    storage.save_plan_version(initial_plan, task_paths, initial_plan.plan_version)
    storage.save_task_result(
        TaskResult(
            task_id="task123",
            status="completed",
            stage="completed",
            upload_file_path="outputs/tasks/task123/input/source.mp4",
            user_params=user_request,
            metadata=_build_video_info(),
            transcript_path="outputs/tasks/task123/transcript/transcript.json",
            candidates_path="outputs/tasks/task123/highlights/candidates.json",
            editing_plan_path="outputs/tasks/task123/plan/editing_plan.json",
            project_state_path="outputs/tasks/task123/project_state.json",
            timeline_path="outputs/tasks/task123/understanding/timeline.json",
            retrieved_context_path="outputs/tasks/task123/understanding/retrieved_context.json",
            planner_memory_path="outputs/tasks/task123/plan/planner_memory.json",
            execution_report_path="outputs/tasks/task123/plan/execution_report.json",
            final_video_path="outputs/tasks/task123/final/final_video.mp4",
            subtitle_path="outputs/tasks/task123/final/subtitles.srt",
            burned_video_path=None,
        ),
        task_paths,
    )
    storage.save_model(
        TaskArtifactManifest(task_id="task123", stage="completed", artifacts=[]),
        task_paths.artifact_manifest_path,
    )
    return storage


def test_revision_api_regenerates_plan_and_returns_task_snapshot(monkeypatch) -> None:
    """Ensure the revision API updates task status and returns the refreshed snapshot."""

    settings = _build_test_settings(_workspace_root("api_revision"))
    storage = _seed_completed_task(settings)
    monkeypatch.setattr(app_module, "settings", settings)

    with TestClient(app_module.app) as client:
        response = client.post(
            "/api/v1/tasks/task123/revise",
            json={"feedback_text": '改成20秒，保留 cand_ending，不要 cand_hook'},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "processing"
    assert payload["stage"] == "editing_plan_generated"
    assert payload["task_result"]["user_params"]["target_duration"] == 20
    assert payload["task_result"]["final_video_path"] is None
    assert payload["artifact_manifest"]["stage"] == "editing_plan_generated"
    assert storage.load_editing_plan("task123").plan_version == 2


def test_revision_api_returns_404_for_missing_task(monkeypatch) -> None:
    """Ensure the revision API reports missing tasks as 404 errors."""

    settings = _build_test_settings(_workspace_root("api_revision_missing"))
    monkeypatch.setattr(app_module, "settings", settings)

    with TestClient(app_module.app) as client:
        response = client.post(
            "/api/v1/tasks/unknown-task/revise",
            json={"feedback_text": "改成20秒"},
        )

    assert response.status_code == 404
