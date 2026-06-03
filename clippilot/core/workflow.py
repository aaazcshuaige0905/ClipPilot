from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import UploadFile
else:
    UploadFile = Any

from clippilot.agents.executor_agent import execute_editing_plan
from clippilot.agents.planner_agent import build_editing_plan
from clippilot.agents.review_agent import review_task_output
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
from clippilot.schemas.review_report import ReviewReport
from clippilot.schemas.task_result import TaskResult
from clippilot.schemas.transcript import TranscriptResult
from clippilot.schemas.user_request import UserRequest
from clippilot.schemas.video_info import VideoInfo
from clippilot.storage.path_manager import AppSettings, load_settings
from clippilot.storage.task_storage import TaskStorage
from clippilot.tools.highlight import generate_highlight_candidates
from clippilot.tools.transcribe import transcribe_video
from clippilot.tools.video_info import extract_video_info, validate_supported_extension, validate_video_duration


def _as_relative_output_path(storage: TaskStorage, output_path: str | None) -> str | None:
    """Convert an absolute artifact path string into the public project-relative form."""

    if not output_path:
        return None
    return storage.as_relative(Path(output_path))


def _create_task_context(
    upload_file: UploadFile,
    user_request: UserRequest,
    storage: TaskStorage,
    settings: AppSettings,
) -> TaskContext:
    """Create task identifiers, paths, and the initial task context."""

    original_file_name = upload_file.filename or "uploaded_video.mp4"
    validate_supported_extension(original_file_name, settings.allowed_extensions)
    task_id = storage.create_task_id()
    task_paths = storage.create_task_paths(task_id, original_file_name)
    context = TaskContext(task_id=task_id, request=user_request, paths=task_paths)
    context.mark_status(TaskStatus.PROCESSING.value)
    return context


def _ingest_source_video(storage: TaskStorage, context: TaskContext, upload_file: UploadFile) -> None:
    """Save the uploaded source video and register its artifacts."""

    storage.save_upload_file(upload_file, context.paths)
    context.add_artifact("source_video", storage.as_relative(context.paths.source_video_path), stage=WorkflowStage.UPLOADED.value)
    log_stage_transition(context, WorkflowStage.UPLOADED.value, "Uploaded source video saved.")


def _process_video_info(storage: TaskStorage, context: TaskContext, settings: AppSettings) -> VideoInfo:
    """Extract, validate, and persist source video metadata."""

    video_info = extract_video_info(context.paths.source_video_path)
    validate_video_duration(video_info, settings)
    video_info_path = storage.save_video_info(video_info, context.paths)
    context.add_artifact("video_info", storage.as_relative(video_info_path), stage=WorkflowStage.VIDEO_INFO_EXTRACTED.value)
    log_stage_transition(context, WorkflowStage.VIDEO_INFO_EXTRACTED.value, "Video metadata extracted and saved.")
    return video_info


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
    context.add_artifact("transcript", storage.as_relative(transcript_path), stage=WorkflowStage.TRANSCRIBED.value)
    log_stage_transition(context, WorkflowStage.TRANSCRIBED.value, "Transcript generated and saved.")
    return transcript


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
    context.add_artifact(
        "highlight_candidates",
        storage.as_relative(candidates_path),
        stage=WorkflowStage.HIGHLIGHT_CANDIDATES_GENERATED.value,
    )
    log_stage_transition(
        context,
        WorkflowStage.HIGHLIGHT_CANDIDATES_GENERATED.value,
        "Highlight candidates generated and saved.",
    )
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
    context.add_artifact(
        "editing_plan",
        storage.as_relative(editing_plan_path),
        stage=WorkflowStage.EDITING_PLAN_GENERATED.value,
    )
    log_stage_transition(context, WorkflowStage.EDITING_PLAN_GENERATED.value, "Editing plan generated and saved.")
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
    context.add_artifact(
        "execution_report",
        storage.as_relative(execution_report_path),
        stage=WorkflowStage.EXECUTION_REPORT_GENERATED.value,
    )
    for clip_result in execution_report.clip_results:
        if clip_result.success:
            context.add_artifact(
                f"clip_{clip_result.clip_id}",
                storage.as_relative(Path(clip_result.clip_path)),
                stage=WorkflowStage.EXECUTION_REPORT_GENERATED.value,
            )
    if context.paths.final_video_path.exists():
        context.add_artifact(
            "final_video",
            storage.as_relative(context.paths.final_video_path),
            stage=WorkflowStage.EXECUTION_REPORT_GENERATED.value,
        )
    if context.paths.subtitle_path.exists():
        context.add_artifact(
            "subtitle_file",
            storage.as_relative(context.paths.subtitle_path),
            stage=WorkflowStage.EXECUTION_REPORT_GENERATED.value,
        )
    if context.paths.burned_video_path.exists():
        context.add_artifact(
            "burned_video",
            storage.as_relative(context.paths.burned_video_path),
            stage=WorkflowStage.EXECUTION_REPORT_GENERATED.value,
        )
    log_stage_transition(
        context,
        WorkflowStage.EXECUTION_REPORT_GENERATED.value,
        "Execution report generated and saved.",
    )
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
    context.add_artifact(
        "review_report",
        storage.as_relative(review_report_path),
        stage=WorkflowStage.REVIEW_REPORT_GENERATED.value,
    )
    log_stage_transition(
        context,
        WorkflowStage.REVIEW_REPORT_GENERATED.value,
        "Review report generated and saved.",
    )
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
    context.add_artifact("task_result", storage.as_relative(context.paths.task_result_path), stage=WorkflowStage.COMPLETED.value)
    manifest_path = storage.save_artifact_manifest(context)
    trace_path = storage.append_trace_entries(context)
    context.add_artifact("artifact_manifest", storage.as_relative(manifest_path), stage=WorkflowStage.COMPLETED.value)
    context.add_artifact("workflow_trace", storage.as_relative(trace_path), stage=WorkflowStage.COMPLETED.value)
    storage.save_artifact_manifest(context)
    return completed_result


def run_stage1_workflow(
    upload_file: UploadFile,
    user_request: UserRequest,
    settings: AppSettings | None = None,
) -> TaskResult:
    """Execute the current stage-1 workflow while saving all important intermediate artifacts."""

    active_settings = settings or load_settings()
    storage = TaskStorage(active_settings)
    context = _create_task_context(upload_file=upload_file, user_request=user_request, storage=storage, settings=active_settings)

    _ingest_source_video(storage=storage, context=context, upload_file=upload_file)
    video_info = _process_video_info(storage=storage, context=context, settings=active_settings)
    transcript = _process_transcript(storage=storage, context=context, settings=active_settings)
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
