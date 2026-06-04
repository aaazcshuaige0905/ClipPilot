from pathlib import Path

from clippilot.agents.planner_agent import build_editing_plan
from clippilot.schemas.editing_plan import HighlightCandidate, HighlightCandidatesResult, EditingPlan
from clippilot.schemas.transcript import TranscriptResult, TranscriptSegment
from clippilot.schemas.user_request import UserRequest
from clippilot.schemas.video_info import VideoInfo
from clippilot.storage.path_manager import AppSettings
from clippilot.storage.task_storage import TaskStorage
from clippilot.storage.json_io import read_json_file


def _workspace_root(test_name: str) -> Path:
    """Return a writable workspace-local directory for filesystem tests."""

    root = Path("tests_runtime") / test_name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build_test_settings(root: Path) -> AppSettings:
    """Create isolated settings for planner tests."""

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
    """Create a reusable user request fixture for planner tests."""

    return UserRequest(
        target_platform="bilibili",
        target_duration=30,
        edit_style="powerful",
        language="zh",
        need_burn_subtitle=True,
    )


def _build_video_info() -> VideoInfo:
    """Create reusable source video metadata for planner tests."""

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
    """Create a transcript fixture long enough for planner fallback logic."""

    return TranscriptResult(
        video_id="task123",
        provider="mock",
        full_text="先说结论 这个方法最重要 为什么很多人会失败 真正关键在执行细节 最后给你一个明确建议",
        segments=[
            TranscriptSegment(start=0.0, end=4.0, text="先说结论，这个方法最重要。"),
            TranscriptSegment(start=4.0, end=8.0, text="为什么很多人会失败？"),
            TranscriptSegment(start=8.0, end=13.0, text="真正关键在执行细节。"),
            TranscriptSegment(start=13.0, end=18.0, text="最后给你一个明确建议。"),
            TranscriptSegment(start=18.0, end=24.0, text="这是一个补充说明片段。"),
        ],
    )


def test_build_editing_plan_generates_executable_clips() -> None:
    """Ensure the planner returns a non-empty executable timeline under duration constraints."""

    candidates = HighlightCandidatesResult(
        video_id="task123",
        candidates=[
            HighlightCandidate(start=0.0, end=9.0, text="先说结论，这个方法最重要。", score=0.93, reason="Strong opening hook."),
            HighlightCandidate(start=10.0, end=19.0, text="为什么很多人会失败？真正关键在执行细节。", score=0.88, reason="Clear opinion and contrast."),
            HighlightCandidate(start=21.0, end=30.0, text="最后给你一个明确建议。", score=0.81, reason="Actionable takeaway."),
            HighlightCandidate(start=32.0, end=41.0, text="最后给你一个明确建议。", score=0.72, reason="Duplicate text should be ignored."),
        ],
    )

    plan = build_editing_plan(
        task_id="task123",
        user_request=_build_user_request(),
        video_info=_build_video_info(),
        transcript=_build_transcript(),
        candidates=candidates,
    )

    assert plan.task_id == "task123"
    assert plan.clips
    assert plan.total_duration <= plan.target_duration + 5
    assert plan.clips[0].purpose == "hook"
    assert len({clip.text for clip in plan.clips}) == len(plan.clips)
    for clip in plan.clips:
        assert clip.source_start < clip.source_end
        assert 3 <= clip.duration <= 20


def test_editing_plan_can_be_saved_and_loaded() -> None:
    """Ensure editing_plan.json can be persisted and reloaded through the task storage layer."""

    settings = _build_test_settings(_workspace_root("planner_agent"))
    storage = TaskStorage(settings)
    task_paths = storage.create_task_paths("task123", "demo.mp4")

    plan = build_editing_plan(
        task_id="task123",
        user_request=_build_user_request(),
        video_info=_build_video_info(),
        transcript=_build_transcript(),
        candidates=HighlightCandidatesResult(
            video_id="task123",
            candidates=[
                HighlightCandidate(start=0.0, end=9.0, text="先说结论，这个方法最重要。", score=0.93, reason="Strong opening hook."),
                HighlightCandidate(start=10.0, end=19.0, text="为什么很多人会失败？真正关键在执行细节。", score=0.88, reason="Clear opinion and contrast."),
            ],
        ),
    )

    saved_path = storage.save_editing_plan(plan, task_paths)
    loaded_plan = EditingPlan.model_validate(read_json_file(saved_path))

    assert saved_path.exists()
    assert loaded_plan.task_id == "task123"
    assert loaded_plan.clips
    assert loaded_plan.total_duration <= loaded_plan.target_duration + 5
