from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import UploadFile
else:
    UploadFile = Any

from clippilot.agents.executor_agent import execute_editing_plan
from clippilot.agents.planner_agent import build_editing_plan
from clippilot.agents.review_agent import review_task_output
from clippilot.agents.video_understanding_agent import analyze_video_understanding
from clippilot.core.states import TaskStatus, WorkflowStage
from clippilot.core.task_context import TaskContext
from clippilot.harness.trace_logger import log_stage_transition
from clippilot.harness.validators import (
    validate_candidates_not_empty,
    validate_output_file_exists,
    validate_transcript_not_empty,
)
from clippilot.schemas.editing_plan import EditingPlan, HighlightCandidatesResult
from clippilot.schemas.execution_report import ExecutionReport
from clippilot.schemas.project_state import (
    FineGrainedUnit,
    LLMHighlightCandidate,
    ProjectPaths,
    ProjectState,
    VideoTimeline,
)
from clippilot.schemas.review_report import ReviewReport
from clippilot.schemas.task_result import TaskResult
from clippilot.schemas.transcript import TranscriptResult
from clippilot.schemas.user_request import UserRequest
from clippilot.schemas.video_info import VideoInfo
from clippilot.storage.path_manager import AppSettings, load_settings
from clippilot.storage.task_storage import TaskStorage
from clippilot.tools.audio_extract import extract_audio_track
from clippilot.tools.highlight import generate_highlight_candidates
from clippilot.tools.transcribe import transcribe_video
from clippilot.tools.video_info import extract_video_info, validate_supported_extension, validate_video_duration


def _as_relative_output_path(storage: TaskStorage, output_path: str | None) -> str | None:
    """Convert an absolute artifact path string into the public project-relative form."""

    if not output_path:
        return None
    return storage.as_relative(Path(output_path))


def _register_artifact_once(
    context: TaskContext,
    name: str,
    path: str,
    stage: str | None = None,
) -> None:
    """Register an artifact only once so repeated state saves do not duplicate manifest entries."""

    if any(artifact.name == name and artifact.relative_path == path for artifact in context.artifacts):
        return
    context.add_artifact(name=name, path=path, stage=stage)


def _save_project_state(storage: TaskStorage, context: TaskContext) -> None:
    """Persist the current shared project state to disk."""

    if context.project_state is None:
        return
    storage.save_project_state(context.project_state, context.paths)
    _register_artifact_once(
        context,
        "project_state",
        storage.as_relative(context.paths.project_state_path),
        stage=context.stage,
    )


def _extract_keywords_from_text(text: str) -> list[str]:
    """Extract simple stable keywords from transcript text for downstream prompting."""

    keywords: list[str] = []
    for raw_token in text.replace(",", " ").replace(".", " ").split():
        token = raw_token.strip().lower()
        if len(token) < 4 or token in keywords:
            continue
        keywords.append(token)
        if len(keywords) >= 5:
            break
    return keywords


def _build_fine_grained_units(transcript: TranscriptResult) -> list[FineGrainedUnit]:
    """Convert transcript segments into fine-grained editable units with boundary hints."""

    source_segments = transcript.segments or []
    fine_grained_units: list[FineGrainedUnit] = []
    pause_threshold_seconds = 0.35

    for index, segment in enumerate(source_segments, start=1):
        previous_segment = source_segments[index - 2] if index > 1 else None
        next_segment = source_segments[index] if index < len(source_segments) else None
        previous_gap = round(max(0.0, segment.start - previous_segment.end), 2) if previous_segment else 0.0
        next_gap = round(max(0.0, next_segment.start - segment.end), 2) if next_segment else 0.0
        emphasis_score = min(1.0, round((len(segment.text.strip()) / 80.0), 2)) if segment.text.strip() else 0.0

        fine_grained_units.append(
            FineGrainedUnit(
                unit_id=f"unit_{index:03d}",
                start=round(segment.start, 2),
                end=round(segment.end, 2),
                duration=round(segment.end - segment.start, 2),
                text=segment.text.strip(),
                previous_gap_seconds=previous_gap,
                next_gap_seconds=next_gap,
                has_pause_before=previous_gap >= pause_threshold_seconds,
                has_pause_after=next_gap >= pause_threshold_seconds,
                emphasis_score=emphasis_score,
                keywords=_extract_keywords_from_text(segment.text),
            )
        )

    return fine_grained_units


def _build_initial_project_state(storage: TaskStorage, task_id: str, user_request: UserRequest, raw_video_path: Path) -> ProjectState:
    """Create the task-wide mutable state using the current request schema and path layout."""

    return ProjectState(
        project_id=task_id,
        user_request=user_request,
        paths=ProjectPaths(raw_video=storage.as_relative(raw_video_path)),
    )


def _create_task_context(
    upload_file: UploadFile,
    user_request: UserRequest,
    storage: TaskStorage,
    settings: AppSettings,
) -> TaskContext:
    """Create task identifiers, paths, global state, and the initial task context."""

    original_file_name = upload_file.filename or "uploaded_video.mp4"
    validate_supported_extension(original_file_name, settings.allowed_extensions)
    task_id = storage.create_task_id()
    task_paths = storage.create_task_paths(task_id, original_file_name)
    project_state = _build_initial_project_state(
        storage=storage,
        task_id=task_id,
        user_request=user_request,
        raw_video_path=task_paths.source_video_path,
    )
    context = TaskContext(task_id=task_id, request=user_request, paths=task_paths, project_state=project_state)
    context.mark_status(TaskStatus.PROCESSING.value)
    context.add_trace(stage=WorkflowStage.CREATED.value, message="Global project state initialized.")
    _save_project_state(storage, context)
    return context


def _ingest_source_video(storage: TaskStorage, context: TaskContext, upload_file: UploadFile) -> None:
    """Save the uploaded source video and register its artifacts."""

    storage.save_upload_file(upload_file, context.paths)
    _register_artifact_once(
        context,
        "source_video",
        storage.as_relative(context.paths.source_video_path),
        stage=WorkflowStage.UPLOADED.value,
    )
    log_stage_transition(context, WorkflowStage.UPLOADED.value, "Uploaded source video saved.")
    _save_project_state(storage, context)


def _process_video_info(storage: TaskStorage, context: TaskContext, settings: AppSettings) -> VideoInfo:
    """Extract, validate, and persist source video metadata."""

    video_info = extract_video_info(context.paths.source_video_path)
    validate_video_duration(video_info, settings)
    video_info_path = storage.save_video_info(video_info, context.paths)
    if context.project_state is not None:
        context.project_state.video_metadata = video_info
        context.project_state.validation_result = {
            "supported_extension": True,
            "duration_valid": True,
        }
    _register_artifact_once(
        context,
        "video_info",
        storage.as_relative(video_info_path),
        stage=WorkflowStage.VIDEO_INFO_EXTRACTED.value,
    )
    log_stage_transition(context, WorkflowStage.VIDEO_INFO_EXTRACTED.value, "Video metadata extracted and saved.")
    _save_project_state(storage, context)
    return video_info


def _process_audio(storage: TaskStorage, context: TaskContext, video_info: VideoInfo) -> str | None:
    """Extract the audio track when the source video contains audio."""

    if not video_info.has_audio:
        if context.project_state is not None:
            context.project_state.logs.append("[audio_extracted] Audio extraction skipped because the video has no audio track.")
        _save_project_state(storage, context)
        return None

    audio_path = extract_audio_track(context.paths.source_video_path, context.paths.audio_path)
    relative_audio_path = storage.as_relative(audio_path)
    if context.project_state is not None:
        context.project_state.paths.audio = relative_audio_path
    _register_artifact_once(
        context,
        "audio_track",
        relative_audio_path,
        stage=WorkflowStage.AUDIO_EXTRACTED.value,
    )
    log_stage_transition(context, WorkflowStage.AUDIO_EXTRACTED.value, "Audio track extracted and saved.")
    _save_project_state(storage, context)
    return relative_audio_path


def _process_transcript(storage: TaskStorage, context: TaskContext, settings: AppSettings) -> TranscriptResult:
    """Generate, validate, and persist transcript artifacts."""

    transcript = transcribe_video(
        video_path=context.paths.source_video_path,
        task_id=context.task_id,
        language=context.request.language,
        provider=settings.asr_provider,
        settings=settings,
    )
    validate_transcript_not_empty(transcript)
    transcript_path = storage.save_transcript(transcript, context.paths)
    validate_output_file_exists(transcript_path)
    if context.project_state is not None:
        fine_grained_units = _build_fine_grained_units(transcript)
        context.project_state.transcript = transcript
        context.project_state.fine_grained_units = fine_grained_units
        context.project_state.paths.transcript = storage.as_relative(transcript_path)
        validation_result = context.project_state.validation_result or {}
        validation_result["transcript_non_empty"] = True
        validation_result["fine_grained_unit_count"] = len(fine_grained_units)
        context.project_state.validation_result = validation_result
    _register_artifact_once(
        context,
        "transcript",
        storage.as_relative(transcript_path),
        stage=WorkflowStage.TRANSCRIBED.value,
    )
    log_stage_transition(context, WorkflowStage.TRANSCRIBED.value, "Transcript generated and saved.")
    log_stage_transition(context, WorkflowStage.PREPROCESSED.value, "Video preprocessing completed.")
    _save_project_state(storage, context)
    return transcript


def _process_video_understanding(
    storage: TaskStorage,
    context: TaskContext,
    video_info: VideoInfo,
    transcript: TranscriptResult,
) -> tuple[VideoTimeline, list[LLMHighlightCandidate]]:
    """Generate transcript-backed video understanding outputs and write them into the global state."""

    fine_grained_units = context.project_state.fine_grained_units if context.project_state is not None else None
    content_analysis, timeline, highlight_candidates_llm = analyze_video_understanding(
        video_path=context.paths.source_video_path,
        user_request=context.request,
        video_info=video_info,
        transcript=transcript,
        fine_grained_units=fine_grained_units or [],
    )
    timeline_path = storage.save_timeline(timeline, context.paths)
    if context.project_state is not None:
        context.project_state.content_analysis = content_analysis
        context.project_state.timeline = timeline
        context.project_state.highlight_candidates_llm = highlight_candidates_llm
        context.project_state.paths.timeline = storage.as_relative(timeline_path)
        context.project_state.retrieved_context = {
            "provider": timeline.provider,
            "model": timeline.model,
            "generation_mode": timeline.generation_mode,
            "highlight_candidate_count": len(highlight_candidates_llm),
        }
    _register_artifact_once(
        context,
        "video_timeline",
        storage.as_relative(timeline_path),
        stage=WorkflowStage.VIDEO_UNDERSTOOD.value,
    )
    log_stage_transition(context, WorkflowStage.VIDEO_UNDERSTOOD.value, "Video understanding completed and timeline saved.")
    _save_project_state(storage, context)
    return timeline, highlight_candidates_llm


def _process_highlight_candidates(
    storage: TaskStorage,
    context: TaskContext,
    settings: AppSettings,
    transcript: TranscriptResult,
) -> HighlightCandidatesResult:
    """Generate, validate, and persist highlight candidate artifacts."""

    candidates = generate_highlight_candidates(
        transcript=transcript,
        user_request=context.request,
        settings=settings,
    )
    validate_candidates_not_empty(candidates)
    candidates_path = storage.save_candidates(candidates, context.paths)
    validate_output_file_exists(candidates_path)
    _register_artifact_once(
        context,
        "highlight_candidates",
        storage.as_relative(candidates_path),
        stage=WorkflowStage.HIGHLIGHT_CANDIDATES_GENERATED.value,
    )
    log_stage_transition(
        context,
        WorkflowStage.HIGHLIGHT_CANDIDATES_GENERATED.value,
        "Highlight candidates generated and saved.",
    )
    _save_project_state(storage, context)
    return candidates


def _process_editing_plan(
    storage: TaskStorage,
    context: TaskContext,
    video_info: VideoInfo,
    transcript: TranscriptResult,
    candidates: HighlightCandidatesResult,
) -> EditingPlan:
    """Build and persist an editing plan artifact from ranked candidates."""

    editing_plan = build_editing_plan(
        task_id=context.task_id,
        user_request=context.request,
        video_info=video_info,
        transcript=transcript,
        candidates=candidates,
    )
    editing_plan_path = storage.save_editing_plan(editing_plan, context.paths)
    validate_output_file_exists(editing_plan_path)
    if context.project_state is not None:
        context.project_state.timeline = context.project_state.timeline or None
    _register_artifact_once(
        context,
        "editing_plan",
        storage.as_relative(editing_plan_path),
        stage=WorkflowStage.EDITING_PLAN_GENERATED.value,
    )
    log_stage_transition(context, WorkflowStage.EDITING_PLAN_GENERATED.value, "Editing plan generated and saved.")
    _save_project_state(storage, context)
    return editing_plan


def _process_execution_report(
    storage: TaskStorage,
    context: TaskContext,
    editing_plan: EditingPlan,
) -> ExecutionReport:
    """Execute the editing plan and persist the resulting execution artifacts."""

    execution_report = execute_editing_plan(
        task_id=context.task_id,
        source_video_path=str(context.paths.source_video_path),
        editing_plan=editing_plan,
        task_paths=context.paths,
        need_burn_subtitle=context.request.need_burn_subtitle,
    )
    execution_report_path = storage.save_execution_report(execution_report, context.paths)
    validate_output_file_exists(execution_report_path)
    _register_artifact_once(
        context,
        "execution_report",
        storage.as_relative(execution_report_path),
        stage=WorkflowStage.EXECUTION_REPORT_GENERATED.value,
    )
    for clip_result in execution_report.clip_results:
        if clip_result.success:
            _register_artifact_once(
                context,
                f"clip_{clip_result.clip_id}",
                storage.as_relative(Path(clip_result.clip_path)),
                stage=WorkflowStage.EXECUTION_REPORT_GENERATED.value,
            )
    if context.paths.final_video_path.exists():
        final_video_relative = storage.as_relative(context.paths.final_video_path)
        if context.project_state is not None:
            context.project_state.paths.final_video = final_video_relative
        _register_artifact_once(
            context,
            "final_video",
            final_video_relative,
            stage=WorkflowStage.EXECUTION_REPORT_GENERATED.value,
        )
    if context.paths.subtitle_path.exists():
        _register_artifact_once(
            context,
            "subtitle_file",
            storage.as_relative(context.paths.subtitle_path),
            stage=WorkflowStage.EXECUTION_REPORT_GENERATED.value,
        )
    if context.paths.burned_video_path.exists():
        _register_artifact_once(
            context,
            "burned_video",
            storage.as_relative(context.paths.burned_video_path),
            stage=WorkflowStage.EXECUTION_REPORT_GENERATED.value,
        )
    if context.project_state is not None:
        context.project_state.render_output = execution_report.model_dump()
    log_stage_transition(
        context,
        WorkflowStage.EXECUTION_REPORT_GENERATED.value,
        "Execution report generated and saved.",
    )
    _save_project_state(storage, context)
    return execution_report


def _process_review_report(
    storage: TaskStorage,
    context: TaskContext,
    video_info: VideoInfo,
    editing_plan: EditingPlan,
    execution_report: ExecutionReport,
) -> ReviewReport:
    """Build and persist the review report based on real task artifacts."""

    review_report = review_task_output(
        user_request=context.request,
        original_video_info=video_info,
        editing_plan=editing_plan,
        execution_report=execution_report,
    )
    review_report_path = storage.save_review_report(review_report, context.paths)
    validate_output_file_exists(review_report_path)
    if context.project_state is not None:
        context.project_state.qc_report = review_report.model_dump()
    _register_artifact_once(
        context,
        "review_report",
        storage.as_relative(review_report_path),
        stage=WorkflowStage.REVIEW_REPORT_GENERATED.value,
    )
    log_stage_transition(
        context,
        WorkflowStage.REVIEW_REPORT_GENERATED.value,
        "Review report generated and saved.",
    )
    _save_project_state(storage, context)
    return review_report


def _build_task_result(
    storage: TaskStorage,
    context: TaskContext,
    video_info: VideoInfo,
    execution_report: ExecutionReport,
) -> TaskResult:
    """Build the public task result response from the current task context."""

    return TaskResult(
        task_id=context.task_id,
        status=context.status,
        stage=context.stage,
        upload_file_path=storage.as_relative(context.paths.source_video_path),
        user_params=context.request,
        metadata=video_info,
        transcript_path=storage.as_relative(context.paths.transcript_json_path),
        candidates_path=storage.as_relative(context.paths.highlight_candidates_path),
        editing_plan_path=storage.as_relative(context.paths.editing_plan_path),
        project_state_path=storage.as_relative(context.paths.project_state_path),
        timeline_path=storage.as_relative(context.paths.timeline_path) if context.paths.timeline_path.exists() else None,
        execution_report_path=storage.as_relative(context.paths.execution_report_path),
        final_video_path=_as_relative_output_path(storage, execution_report.final_video_path),
        subtitle_path=_as_relative_output_path(storage, execution_report.subtitle_path),
        burned_video_path=_as_relative_output_path(storage, execution_report.burned_video_path),
    )


def _finalize_task(
    storage: TaskStorage,
    context: TaskContext,
    task_result: TaskResult,
    final_status: str,
) -> TaskResult:
    """Persist final task-level artifacts and mark the task with its final workflow status."""

    context.mark_status(final_status)
    completion_message = (
        "Stage-1 workflow completed successfully."
        if final_status == TaskStatus.COMPLETED.value
        else "Stage-1 workflow finished with execution or review errors."
    )
    log_stage_transition(context, WorkflowStage.COMPLETED.value, completion_message)
    completed_result = task_result.model_copy(update={"status": context.status, "stage": context.stage})
    storage.save_task_result(completed_result, context.paths)
    _register_artifact_once(
        context,
        "task_result",
        storage.as_relative(context.paths.task_result_path),
        stage=WorkflowStage.COMPLETED.value,
    )
    _save_project_state(storage, context)
    manifest_path = storage.save_artifact_manifest(context)
    trace_path = storage.append_trace_entries(context)
    _register_artifact_once(
        context,
        "artifact_manifest",
        storage.as_relative(manifest_path),
        stage=WorkflowStage.COMPLETED.value,
    )
    _register_artifact_once(
        context,
        "workflow_trace",
        storage.as_relative(trace_path),
        stage=WorkflowStage.COMPLETED.value,
    )
    storage.save_artifact_manifest(context)
    return completed_result


def _persist_failure_state(storage: TaskStorage, context: TaskContext, exc: Exception) -> None:
    """Write failure details into the shared state before re-raising the exception."""

    context.mark_status(TaskStatus.FAILED.value)
    context.add_error(str(exc))
    log_stage_transition(context, context.stage, f"Workflow failed: {exc}")
    _save_project_state(storage, context)
    storage.save_artifact_manifest(context)
    storage.append_trace_entries(context)


def run_stage1_workflow(
    upload_file: UploadFile,
    user_request: UserRequest,
    settings: AppSettings | None = None,
) -> TaskResult:
    """Execute the current stage-1 workflow while saving all important intermediate artifacts."""

    active_settings = settings or load_settings()
    storage = TaskStorage(active_settings)
    context = _create_task_context(upload_file=upload_file, user_request=user_request, storage=storage, settings=active_settings)

    try:
        _ingest_source_video(storage=storage, context=context, upload_file=upload_file)
        video_info = _process_video_info(storage=storage, context=context, settings=active_settings)
        _process_audio(storage=storage, context=context, video_info=video_info)
        transcript = _process_transcript(storage=storage, context=context, settings=active_settings)
        _process_video_understanding(
            storage=storage,
            context=context,
            video_info=video_info,
            transcript=transcript,
        )
        candidates = _process_highlight_candidates(
            storage=storage,
            context=context,
            settings=active_settings,
            transcript=transcript,
        )
        editing_plan = _process_editing_plan(
            storage=storage,
            context=context,
            video_info=video_info,
            transcript=transcript,
            candidates=candidates,
        )
        execution_report = _process_execution_report(storage=storage, context=context, editing_plan=editing_plan)
        review_report = _process_review_report(
            storage=storage,
            context=context,
            video_info=video_info,
            editing_plan=editing_plan,
            execution_report=execution_report,
        )

        task_result = _build_task_result(
            storage=storage,
            context=context,
            video_info=video_info,
            execution_report=execution_report,
        )
        final_status = (
            TaskStatus.COMPLETED.value
            if execution_report.status == "completed" and review_report.passed
            else TaskStatus.FAILED.value
        )
        return _finalize_task(
            storage=storage,
            context=context,
            task_result=task_result,
            final_status=final_status,
        )
    except Exception as exc:
        _persist_failure_state(storage=storage, context=context, exc=exc)
        raise
