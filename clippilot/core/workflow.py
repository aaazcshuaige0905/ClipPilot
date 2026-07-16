from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import UploadFile
else:
    UploadFile = Any

from clippilot.agents.executor_agent import execute_editing_plan
from clippilot.agents.planner_agent import build_editing_plan
from clippilot.agents.review_agent import review_task_output
from clippilot.agents.video_understanding_agent import (
    aggregate_window_understandings,
    analyze_video_understanding,
    build_candidate_refinement_groups,
    merge_windowed_video_understanding_outputs,
    refine_candidate_group,
    understand_video_window,
)
from clippilot.core.parallel_runner import run_bounded_jobs
from clippilot.core.memory_manager import (
    build_initial_planner_memory,
    record_plan_generation,
    record_retrieved_context,
    record_review_outcome,
)
from clippilot.core.states import TaskStatus, WorkflowStage
from clippilot.core.task_context import TaskContext
from clippilot.harness.trace_logger import log_stage_transition
from clippilot.rag.query_builder import build_retrieval_query
from clippilot.rag.retrieve import retrieve_context
from clippilot.harness.validators import (
    validate_candidates_not_empty,
    validate_output_file_exists,
    validate_transcript_not_empty,
)
from clippilot.schemas.editing_plan import EditingPlan, HighlightCandidatesResult
from clippilot.schemas.execution_report import ExecutionReport
from clippilot.schemas.project_state import (
    ContentAnalysis,
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
from clippilot.schemas.video_frames import SampledFramesResult
from clippilot.schemas.video_info import VideoInfo
from clippilot.schemas.video_understanding import CoarseHighlightCandidate, VideoWindow
from clippilot.storage.path_manager import AppSettings, load_settings
from clippilot.storage.task_storage import TaskStorage
from clippilot.tools.audio_extract import extract_audio_track
from clippilot.tools.frame_sampler import sample_video_frames
from clippilot.tools.highlight import generate_highlight_candidates
from clippilot.tools.transcribe import build_skipped_transcript, transcribe_video
from clippilot.tools.video_info import extract_video_info, validate_supported_extension, validate_video_duration
from clippilot.tools.video_segmenter import create_video_windows


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


def _save_planner_memory(storage: TaskStorage, context: TaskContext) -> None:
    """Persist standalone planner short-term memory to its dedicated artifact path."""

    if context.planner_memory is None:
        return
    context.planner_memory.touch()
    storage.save_planner_memory(context.planner_memory, context.paths)
    _register_artifact_once(
        context,
        "planner_memory",
        storage.as_relative(context.paths.planner_memory_path),
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
    planner_memory = build_initial_planner_memory(task_id=task_id, user_request=user_request)
    context = TaskContext(
        task_id=task_id,
        request=user_request,
        paths=task_paths,
        project_state=project_state,
        planner_memory=planner_memory,
    )
    context.mark_status(TaskStatus.PROCESSING.value)
    context.add_trace(stage=WorkflowStage.CREATED.value, message="Global project state initialized.")
    _save_project_state(storage, context)
    _save_planner_memory(storage, context)
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


def _process_transcript(
    storage: TaskStorage,
    context: TaskContext,
    settings: AppSettings,
    video_info: VideoInfo,
) -> TranscriptResult:
    """Generate, validate, and persist transcript artifacts."""

    if not video_info.has_audio:
        transcript = build_skipped_transcript(
            video_id=context.task_id,
            provider=settings.asr_provider,
            status="no_audio",
            warning="ASR skipped because the source video does not contain an audio track.",
        )
    else:
        transcript_input_path = context.paths.audio_path if context.paths.audio_path.exists() else context.paths.source_video_path
        transcript = transcribe_video(
            video_path=transcript_input_path,
            task_id=context.task_id,
            language=context.request.language,
            provider=settings.asr_provider,
            settings=settings,
        )
    validate_transcript_not_empty(transcript)
    transcript_path = storage.save_transcript(transcript, context.paths)
    validate_output_file_exists(transcript_path)
    if context.project_state is not None:
        fine_grained_units = _build_fine_grained_units(transcript) if transcript.has_content() else []
        context.project_state.transcript = transcript
        context.project_state.fine_grained_units = fine_grained_units
        context.project_state.paths.transcript = storage.as_relative(transcript_path)
        validation_result = context.project_state.validation_result or {}
        validation_result["transcript_status"] = transcript.status
        validation_result["transcript_non_empty"] = transcript.has_content()
        validation_result["fine_grained_unit_count"] = len(fine_grained_units)
        context.project_state.validation_result = validation_result
    _register_artifact_once(
        context,
        "transcript",
        storage.as_relative(transcript_path),
        stage=WorkflowStage.TRANSCRIBED.value,
    )
    transcript_message = (
        "Transcript generated and saved."
        if transcript.has_content()
        else f"Transcript saved without subtitle text because ASR status is {transcript.status}."
    )
    log_stage_transition(context, WorkflowStage.TRANSCRIBED.value, transcript_message)
    log_stage_transition(context, WorkflowStage.PREPROCESSED.value, "Video preprocessing completed.")
    _save_project_state(storage, context)
    return transcript


def _process_sampled_frames(
    storage: TaskStorage,
    context: TaskContext,
    video_info: VideoInfo,
    settings: AppSettings | None = None,
) -> SampledFramesResult:
    """Sample local video frames, persist their manifest, and attach them to shared task state."""

    sampled_frames = sample_video_frames(
        video_path=context.paths.source_video_path,
        output_dir=context.paths.understanding_frames_dir,
        video_info=video_info,
        video_id=context.task_id,
        max_frames=(settings.video_understanding_max_frames if settings is not None else 12),
        interval_seconds=(settings.video_understanding_frame_interval_seconds if settings is not None else 3.0),
    )
    normalized_frames = sampled_frames.model_copy(
        update={
            "frames": [
                frame.model_copy(update={"image_path": storage.as_relative(Path(frame.image_path))})
                for frame in sampled_frames.frames
            ]
        }
    )
    sampled_frames_path = storage.save_sampled_frames(normalized_frames, context.paths)
    validate_output_file_exists(sampled_frames_path)
    if context.project_state is not None:
        context.project_state.sampled_frames = normalized_frames
        context.project_state.paths.sampled_frames = storage.as_relative(sampled_frames_path)
    _register_artifact_once(
        context,
        "sampled_frames",
        storage.as_relative(sampled_frames_path),
        stage=WorkflowStage.PREPROCESSED.value,
    )
    for frame in normalized_frames.frames:
        _register_artifact_once(
            context,
            frame.frame_id,
            frame.image_path,
            stage=WorkflowStage.PREPROCESSED.value,
        )
    log_stage_transition(
        context,
        WorkflowStage.PREPROCESSED.value,
        f"Sampled {normalized_frames.frame_count} visual frames for video understanding.",
    )
    _save_project_state(storage, context)
    return normalized_frames


def _process_video_windows(
    storage: TaskStorage,
    context: TaskContext,
    video_info: VideoInfo,
    settings: AppSettings,
) -> list[VideoWindow]:
    """Create source-aligned proxy windows for the live long-video understanding path."""

    units = context.project_state.fine_grained_units if context.project_state is not None else []
    windows = create_video_windows(
        source_video_path=context.paths.source_video_path,
        output_dir=context.paths.understanding_windows_dir,
        video_info=video_info,
        fine_grained_units=units or [],
        window_duration_seconds=settings.video_understanding_window_duration_seconds,
        overlap_seconds=settings.video_understanding_window_overlap_seconds,
        proxy_height=settings.video_understanding_proxy_height,
        max_encoded_bytes=settings.video_understanding_max_base64_bytes,
    )
    storage.save_video_windows(
        {"windows": [window.model_dump() for window in windows]},
        context.paths,
    )
    _register_artifact_once(
        context,
        "video_windows",
        storage.as_relative(context.paths.video_windows_path),
        stage=WorkflowStage.PREPROCESSED.value,
    )
    log_stage_transition(
        context,
        WorkflowStage.PREPROCESSED.value,
        f"Created {len(windows)} overlapping proxy-video windows for live understanding.",
    )
    return windows


def _run_windowed_video_understanding(
    storage: TaskStorage,
    context: TaskContext,
    video_info: VideoInfo,
    windows: list[VideoWindow],
    settings: AppSettings,
) -> tuple[ContentAnalysis, VideoTimeline, list[LLMHighlightCandidate], dict, dict]:
    """Run bounded window and refinement jobs, persisting their artifacts on the main thread."""

    units = context.project_state.fine_grained_units if context.project_state is not None else []

    def _window_worker(window: VideoWindow):
        return understand_video_window(window, context.request, video_info, units or [], settings)

    window_jobs = run_bounded_jobs(
        windows,
        _window_worker,
        max_workers=settings.video_understanding_window_concurrency,
    )
    window_results = []
    window_raw: dict[str, Any] = {}
    for job in window_jobs:
        if job.error is not None or job.result is None:
            raise RuntimeError(f"Unexpected unhandled failure for {job.job.window_id}: {job.error}")
        result, request_payload, raw_response = job.result
        window_results.append(result)
        storage.save_window_request(job.job.window_id, request_payload, context.paths)
        storage.save_window_response(job.job.window_id, raw_response, context.paths)
        storage.save_window_result(job.job.window_id, result, context.paths)
        window_raw[job.job.window_id] = raw_response
    window_results.sort(key=lambda item: item.order)

    global_result, global_request, global_raw = aggregate_window_understandings(
        window_results,
        context.request,
        video_info,
        settings,
    )
    storage.save_global_understanding_request(global_request, context.paths)
    storage.save_global_understanding_response(global_raw, context.paths)
    storage.save_global_understanding(global_result, context.paths)

    coarse_candidates: list[CoarseHighlightCandidate] = [
        candidate for result in window_results for candidate in result.coarse_candidates
    ]
    groups = build_candidate_refinement_groups(
        coarse_candidates,
        global_result.refinement_candidate_ids,
        video_info.duration_seconds,
        settings,
    )
    coarse_by_id = {candidate.candidate_id: candidate for candidate in coarse_candidates}

    def _refinement_worker(group):
        group_candidates = [
            coarse_by_id[candidate_id]
            for candidate_id in group.candidate_ids
            if candidate_id in coarse_by_id
        ]
        return refine_candidate_group(
            group,
            group_candidates,
            context.paths.source_video_path,
            context.paths.refinement_dir,
            units or [],
            settings,
        )

    refinement_jobs = run_bounded_jobs(
        groups,
        _refinement_worker,
        max_workers=settings.video_understanding_refinement_concurrency,
    )
    refined_candidates = []
    refinement_raw: dict[str, Any] = {}
    for job in refinement_jobs:
        if job.error is not None or job.result is None:
            global_result.coverage_warnings.append(
                f"Refinement group {job.job.group_id} failed unexpectedly: {job.error}"
            )
            continue
        refined, rendered_group, request_payload, raw_response = job.result
        refined_candidates.extend(refined)
        storage.save_refinement_request(rendered_group.group_id, request_payload, context.paths)
        storage.save_refinement_response(rendered_group.group_id, raw_response, context.paths)
        refinement_raw[rendered_group.group_id] = raw_response
    storage.save_refined_candidates(
        {"candidates": [candidate.model_dump() for candidate in refined_candidates]},
        context.paths,
    )
    content_analysis, timeline, candidates = merge_windowed_video_understanding_outputs(
        global_result,
        coarse_candidates,
        refined_candidates,
        context.request,
        settings,
    )
    logical_request = {
        "mode": "windowed_video_understanding_v1",
        "window_count": len(windows),
        "window_ids": [window.window_id for window in windows],
        "global_request": global_request,
        "refinement_group_ids": [group.group_id for group in groups],
    }
    raw_response = {
        "mode": "windowed_video_understanding_v1",
        "window_results": [result.model_dump() for result in window_results],
        "window_responses": window_raw,
        "global_result": global_result.model_dump(),
        "global_response": global_raw,
        "refinement_responses": refinement_raw,
        "refined_candidates": [candidate.model_dump() for candidate in refined_candidates],
    }
    return content_analysis, timeline, candidates, logical_request, raw_response


def _process_video_understanding(
    storage: TaskStorage,
    context: TaskContext,
    video_info: VideoInfo,
    transcript: TranscriptResult,
    sampled_frames: SampledFramesResult | None = None,
    video_windows: list[VideoWindow] | None = None,
    settings: AppSettings | None = None,
) -> tuple[VideoTimeline, list[LLMHighlightCandidate]]:
    """Generate transcript-backed video understanding outputs and write them into the global state."""

    fine_grained_units = context.project_state.fine_grained_units if context.project_state is not None else None
    if video_windows and settings is not None:
        content_analysis, timeline, highlight_candidates_llm, request_payload, raw_response = (
            _run_windowed_video_understanding(storage, context, video_info, video_windows, settings)
        )
    else:
        content_analysis, timeline, highlight_candidates_llm, request_payload, raw_response = analyze_video_understanding(
            video_path=context.paths.source_video_path,
            user_request=context.request,
            video_info=video_info,
            transcript=transcript,
            fine_grained_units=fine_grained_units or [],
            sampled_frames=sampled_frames,
            settings=settings,
        )
    request_path = storage.save_video_understanding_request(request_payload, context.paths)
    response_raw_path = storage.save_video_understanding_response_raw(raw_response, context.paths)
    content_analysis_path = storage.save_content_analysis(content_analysis, context.paths)
    timeline_path = storage.save_timeline(timeline, context.paths)
    llm_candidates_path = storage.save_llm_candidates(
        {"candidates": [candidate.model_dump() for candidate in highlight_candidates_llm]},
        context.paths,
    )
    if context.project_state is not None:
        context.project_state.content_analysis = content_analysis
        context.project_state.timeline = timeline
        context.project_state.highlight_candidates_llm = highlight_candidates_llm
        context.project_state.paths.content_analysis = storage.as_relative(content_analysis_path)
        context.project_state.paths.timeline = storage.as_relative(timeline_path)
        context.project_state.paths.llm_candidates = storage.as_relative(llm_candidates_path)
    _register_artifact_once(
        context,
        "video_understanding_request",
        storage.as_relative(request_path),
        stage=WorkflowStage.VIDEO_UNDERSTOOD.value,
    )
    _register_artifact_once(
        context,
        "content_analysis",
        storage.as_relative(content_analysis_path),
        stage=WorkflowStage.VIDEO_UNDERSTOOD.value,
    )
    _register_artifact_once(
        context,
        "video_timeline",
        storage.as_relative(timeline_path),
        stage=WorkflowStage.VIDEO_UNDERSTOOD.value,
    )
    _register_artifact_once(
        context,
        "llm_highlight_candidates",
        storage.as_relative(llm_candidates_path),
        stage=WorkflowStage.VIDEO_UNDERSTOOD.value,
    )
    _register_artifact_once(
        context,
        "video_understanding_response_raw",
        storage.as_relative(response_raw_path),
        stage=WorkflowStage.VIDEO_UNDERSTOOD.value,
    )
    log_stage_transition(context, WorkflowStage.VIDEO_UNDERSTOOD.value, "Video understanding completed and timeline saved.")
    _save_project_state(storage, context)
    return timeline, highlight_candidates_llm


def _process_retrieved_context(
    storage: TaskStorage,
    context: TaskContext,
    settings: AppSettings,
) -> None:
    """Retrieve strategy context for the planner and persist it with diagnostics."""

    if context.project_state is None or not settings.rag_enabled:
        return

    retrieval_query = build_retrieval_query(
        user_request=context.request,
        fine_grained_units=context.project_state.fine_grained_units,
        settings=settings,
    )
    retrieved_context = retrieve_context(query=retrieval_query, settings=settings)
    retrieved_context_path = storage.save_retrieved_context(retrieved_context, context.paths)
    retrieval_trace_path = storage.save_retrieval_trace(retrieved_context.trace, context.paths)
    context.project_state.retrieved_context = retrieved_context
    context.project_state.paths.retrieved_context = storage.as_relative(retrieved_context_path)
    _register_artifact_once(
        context,
        "retrieved_context",
        storage.as_relative(retrieved_context_path),
        stage=WorkflowStage.RAG_CONTEXT_RETRIEVED.value,
    )
    _register_artifact_once(
        context,
        "retrieval_trace",
        storage.as_relative(retrieval_trace_path),
        stage=WorkflowStage.RAG_CONTEXT_RETRIEVED.value,
    )
    log_stage_transition(
        context,
        WorkflowStage.RAG_CONTEXT_RETRIEVED.value,
        f"Retrieved {len(retrieved_context.chunks)} strategy chunks for planner context.",
    )
    if context.planner_memory is not None:
        record_retrieved_context(context.planner_memory, retrieved_context)
        _save_planner_memory(storage, context)
    _save_project_state(storage, context)


def _process_highlight_candidates(
    storage: TaskStorage,
    context: TaskContext,
    settings: AppSettings,
    transcript: TranscriptResult,
) -> HighlightCandidatesResult:
    """Generate, validate, and persist highlight candidate artifacts."""

    if not transcript.has_content():
        candidates = HighlightCandidatesResult(video_id=transcript.video_id, candidates=[])
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
            "Highlight generation skipped because the transcript did not contain usable ASR text.",
        )
        _save_project_state(storage, context)
        return candidates

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
        fine_grained_units=context.project_state.fine_grained_units if context.project_state is not None else None,
        timeline=context.project_state.timeline if context.project_state is not None else None,
        llm_candidates=context.project_state.highlight_candidates_llm if context.project_state is not None else None,
        retrieved_context=context.project_state.retrieved_context if context.project_state is not None else None,
        planner_memory=context.planner_memory,
    )
    editing_plan_path = storage.save_editing_plan(editing_plan, context.paths)
    storage.save_plan_version(editing_plan, context.paths, editing_plan.plan_version)
    validate_output_file_exists(editing_plan_path)
    if context.planner_memory is not None:
        record_plan_generation(context.planner_memory, editing_plan)
        _save_planner_memory(storage, context)
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
    for item_result in execution_report.item_results:
        if item_result.success:
            _register_artifact_once(
                context,
                f"timeline_item_{item_result.item_id}",
                storage.as_relative(Path(item_result.item_path)),
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
    if context.planner_memory is not None:
        record_review_outcome(context.planner_memory, review_report)
        _save_planner_memory(storage, context)
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
        retrieved_context_path=(
            storage.as_relative(context.paths.retrieved_context_path) if context.paths.retrieved_context_path.exists() else None
        ),
        planner_memory_path=(
            storage.as_relative(context.paths.planner_memory_path) if context.paths.planner_memory_path.exists() else None
        ),
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
        transcript = _process_transcript(
            storage=storage,
            context=context,
            settings=active_settings,
            video_info=video_info,
        )
        _process_retrieved_context(storage=storage, context=context, settings=active_settings)
        if active_settings.video_understanding_enabled and active_settings.video_understanding_api_key:
            video_windows = _process_video_windows(
                storage=storage,
                context=context,
                video_info=video_info,
                settings=active_settings,
            )
            _process_video_understanding(
                storage=storage,
                context=context,
                video_info=video_info,
                transcript=transcript,
                video_windows=video_windows,
                settings=active_settings,
            )
        else:
            sampled_frames = _process_sampled_frames(
                storage=storage,
                context=context,
                video_info=video_info,
                settings=active_settings,
            )
            _process_video_understanding(
                storage=storage,
                context=context,
                video_info=video_info,
                transcript=transcript,
                sampled_frames=sampled_frames,
                settings=active_settings,
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
