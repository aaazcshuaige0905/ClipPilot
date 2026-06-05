from pathlib import Path
import sys
import types

_fake_moviepy = types.ModuleType("moviepy")


class _FakeVideoFileClip:
    """Provide a lightweight moviepy stand-in for workflow unit tests."""


_fake_moviepy.VideoFileClip = _FakeVideoFileClip
sys.modules.setdefault("moviepy", _fake_moviepy)
sys.modules.setdefault("moviepy.video", types.ModuleType("moviepy.video"))
sys.modules.setdefault("moviepy.video.io", types.ModuleType("moviepy.video.io"))
_fake_moviepy_video_file_clip = types.ModuleType("moviepy.video.io.VideoFileClip")
_fake_moviepy_video_file_clip.VideoFileClip = _FakeVideoFileClip
sys.modules.setdefault("moviepy.video.io.VideoFileClip", _fake_moviepy_video_file_clip)

from clippilot.core.memory_manager import (
    build_initial_planner_memory,
    record_plan_generation,
    record_retrieved_context,
    record_review_outcome,
)
from clippilot.core.task_context import TaskContext
from clippilot.core.workflow import _process_review_report
from clippilot.rag.schemas import RetrievedChunk, RetrievedContext, RetrievalQuery, RetrievalTrace
from clippilot.schemas.editing_plan import EditingPlan, PlanningBeat, TimelineItem, TimelineSourceRef
from clippilot.schemas.execution_report import ExecutionReport
from clippilot.schemas.project_state import ProjectPaths, ProjectState
from clippilot.schemas.review_report import ReviewCheck, ReviewReport
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
    """Create isolated settings for memory-manager and workflow tests."""

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


def _build_request() -> UserRequest:
    """Create a stable user request fixture for planner-memory tests."""

    return UserRequest(
        target_platform="bilibili",
        target_duration=30,
        edit_style="powerful",
        language="zh",
        need_burn_subtitle=True,
    )


def _build_retrieved_context() -> RetrievedContext:
    """Create a compact retrieved-context fixture for memory tests."""

    return RetrievedContext(
        query=RetrievalQuery(text="platform bilibili | duration 30 seconds"),
        chunks=[
            RetrievedChunk(
                chunk_id="chunk_hook",
                text="Lead with a fast payoff.",
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
                retrieval_sources=["dense"],
            ),
            RetrievedChunk(
                chunk_id="chunk_pacing",
                text="Keep supporting clips tight.",
                title="Pacing Rule",
                source_file="editing_templates.md",
                knowledge_type="template",
                strategy_type="pacing_rule",
                platform="bilibili",
                language="zh",
                style="powerful",
                duration_band="15_30",
                stage="planning",
                priority="should",
                retrieval_sources=["bm25"],
            ),
        ],
        trace=RetrievalTrace(dense_candidate_count=2, bm25_candidate_count=2, merged_candidate_count=2),
    )


def _build_editing_plan(task_id: str) -> EditingPlan:
    """Create a minimal structured editing plan for planner-memory tests."""

    return EditingPlan(
        task_id=task_id,
        plan_version=2,
        target_duration=30,
        total_duration=5.0,
        beats=[
            PlanningBeat(
                beat_id="beat_01",
                order=1,
                role="hook",
                goal="Open strong.",
                target_duration=5.0,
                summary="Fast opening payoff.",
                source_candidate_ids=["cand_01"],
                notes=["Open with the strongest claim."],
            )
        ],
        timeline_items=[
            TimelineItem(
                item_id="item_01",
                beat_id="beat_01",
                purpose="hook",
                assembly_mode="continuous_trim",
                source_start=0.0,
                source_end=5.0,
                duration=5.0,
                candidate_ids=["cand_01"],
                source_unit_ids=["unit_001"],
                source_refs=[
                    TimelineSourceRef(
                        ref_id="unit_001",
                        ref_type="unit",
                        start=0.0,
                        end=5.0,
                        text="A compact opening line.",
                    )
                ],
                text="A compact opening line.",
                subtitle="A compact opening line.",
                score=0.91,
                transition="straight_cut",
                reason="High hook score.",
            )
        ],
        editing_notes=["Use a strong opening beat."],
        warnings=["Monitor duration if extra context is added."],
        strategy="llm_montage_planner",
        generation_mode="stub",
        source_candidate_count=4,
    )


def test_build_initial_planner_memory_seeds_instruction_history() -> None:
    """Ensure initial planner memory is created with a first instruction record."""

    memory = build_initial_planner_memory("task123", _build_request())

    assert memory.project_id == "task123"
    assert memory.active_constraints["target_platform"] == "bilibili"
    assert memory.instruction_history[0].turn_id == "initial_request"
    assert any("30 seconds" in focus for focus in memory.planning_focus)
    assert [event.event_type for event in memory.memory_events[:2]] == [
        "user_instruction",
        "request_initialized",
    ]


def test_record_retrieved_context_updates_memory_focus() -> None:
    """Ensure retrieved strategy context is written back into planner memory."""

    memory = build_initial_planner_memory("task123", _build_request())

    record_retrieved_context(memory, _build_retrieved_context())

    assert memory.retrieved_chunk_ids == ["chunk_hook", "chunk_pacing"]
    assert memory.retrieved_rule_titles == ["Hook Rule", "Pacing Rule"]
    assert any("hook rule" in focus.lower() for focus in memory.planning_focus)
    assert memory.memory_events[-1].event_type == "retrieved_context"


def test_record_plan_generation_tracks_revision_history() -> None:
    """Ensure generated plans append revision memory and preserve warnings."""

    memory = build_initial_planner_memory("task123", _build_request())
    editing_plan = _build_editing_plan("task123")

    record_plan_generation(memory, editing_plan)

    assert memory.current_plan_version == 2
    assert memory.revision_history[-1].plan_version == 2
    assert memory.open_issues == ["Monitor duration if extra context is added."]
    assert memory.memory_events[-1].event_type == "plan_generated"
    assert "cand_01" in memory.planner_working_summary


def test_record_review_outcome_tracks_open_issues() -> None:
    """Ensure review failures and suggestions become future revision issues."""

    memory = build_initial_planner_memory("task123", _build_request())
    review_report = ReviewReport(
        task_id="task123",
        passed=False,
        score=0.45,
        checks=[
            ReviewCheck(
                name="subtitle_sync",
                passed=False,
                severity="high",
                message="Subtitle timing drifts after the opening cut.",
            )
        ],
        suggestions=["Tighten the ending beat by 2 seconds."],
    )

    record_review_outcome(memory, review_report)

    assert memory.open_issues == [
        "Subtitle timing drifts after the opening cut.",
        "Tighten the ending beat by 2 seconds.",
    ]
    assert memory.memory_events[-1].event_type == "review_completed"
    assert "failed" in memory.planner_working_summary


def test_process_review_report_persists_memory_updates(monkeypatch) -> None:
    """Ensure workflow review processing persists planner-memory review writes."""

    settings = _build_test_settings(_workspace_root("memory_manager_review_workflow"))
    storage = TaskStorage(settings)
    task_paths = storage.create_task_paths("task123", "source.mp4")
    memory = build_initial_planner_memory("task123", _build_request())
    project_state = ProjectState(
        project_id="task123",
        user_request=_build_request(),
        paths=ProjectPaths(raw_video="outputs/tasks/task123/input/source.mp4"),
    )
    context = TaskContext(
        task_id="task123",
        request=_build_request(),
        paths=task_paths,
        project_state=project_state,
        planner_memory=memory,
    )
    video_info = VideoInfo(
        duration_seconds=300.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
        file_size_mb=120.0,
        source_path=str(task_paths.source_video_path),
    )
    execution_report = ExecutionReport(
        task_id="task123",
        status="completed",
        item_results=[],
        final_video_path=str(task_paths.final_video_path),
        subtitle_path=str(task_paths.subtitle_path),
        burned_video_path=None,
        warnings=[],
        errors=[],
    )
    editing_plan = _build_editing_plan("task123")
    review_report = ReviewReport(
        task_id="task123",
        passed=False,
        score=0.5,
        checks=[
            ReviewCheck(
                name="ending_cta",
                passed=False,
                severity="medium",
                message="Ending CTA is too weak for the target platform.",
            )
        ],
        suggestions=["Try a sharper last line."],
    )

    monkeypatch.setattr("clippilot.core.workflow.review_task_output", lambda **kwargs: review_report)

    _process_review_report(
        storage=storage,
        context=context,
        video_info=video_info,
        editing_plan=editing_plan,
        execution_report=execution_report,
    )
    loaded_memory = storage.load_planner_memory("task123")

    assert loaded_memory.memory_events[-1].event_type == "review_completed"
    assert loaded_memory.open_issues[0] == "Ending CTA is too weak for the target platform."
