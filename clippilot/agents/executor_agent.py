from pathlib import Path

from clippilot.core.exceptions import ClipPilotProcessingError
from clippilot.schemas.editing_plan import EditingPlan
from clippilot.schemas.execution_report import ExecutionClipResult, ExecutionReport
from clippilot.storage.path_manager import TaskPaths
from clippilot.tools.subtitle import burn_subtitles_to_video, generate_srt_from_editing_plan
from clippilot.tools.video_cut import cut_video_clip
from clippilot.tools.video_merge import merge_video_clips


def _clip_output_path(task_paths: TaskPaths, clip_id: str) -> Path:
    """Build the output path for one cut clip inside the task clips directory."""

    return task_paths.clips_dir / f"{clip_id}.mp4"


def _normalize_editing_plan(editing_plan: EditingPlan | dict) -> EditingPlan:
    """Normalize editing plan payloads into a validated EditingPlan model."""

    if isinstance(editing_plan, EditingPlan):
        return editing_plan
    return EditingPlan.model_validate(editing_plan)


def execute_editing_plan(
    task_id: str,
    source_video_path: str,
    editing_plan: EditingPlan | dict,
    task_paths: TaskPaths,
    need_burn_subtitle: bool,
) -> ExecutionReport:
    """Execute the editing plan by cutting clips, merging them, and generating subtitles."""

    plan = _normalize_editing_plan(editing_plan)
    clip_results: list[ExecutionClipResult] = []
    warnings: list[str] = []
    errors: list[str] = []
    successful_clip_paths: list[str] = []
    final_video_path = str(task_paths.final_video_path)
    subtitle_path = str(task_paths.subtitle_path)
    burned_video_path = str(task_paths.burned_video_path) if need_burn_subtitle else None

    for clip in plan.clips:
        output_clip_path = _clip_output_path(task_paths, clip.clip_id)
        try:
            cut_result = cut_video_clip(
                input_video_path=source_video_path,
                output_clip_path=str(output_clip_path),
                start_time=clip.source_start,
                end_time=clip.source_end,
            )
        except Exception as exc:
            errors.append(f"{clip.clip_id}: unexpected cut failure: {exc}")
            clip_results.append(
                ExecutionClipResult(
                    clip_id=clip.clip_id,
                    source_start=clip.source_start,
                    source_end=clip.source_end,
                    clip_path=str(output_clip_path),
                    duration=0.0,
                    success=False,
                    error=f"Unexpected cut failure: {exc}",
                )
            )
            continue

        clip_results.append(
            ExecutionClipResult(
                clip_id=clip.clip_id,
                source_start=clip.source_start,
                source_end=clip.source_end,
                clip_path=cut_result.clip_path,
                duration=cut_result.duration,
                success=cut_result.success,
                error=cut_result.error,
            )
        )
        if cut_result.success:
            successful_clip_paths.append(cut_result.clip_path)
        elif cut_result.error:
            errors.append(f"{clip.clip_id}: {cut_result.error}")

    if len(successful_clip_paths) < len(plan.clips):
        warnings.append("Some clips failed to cut and were excluded from the final merge.")

    if successful_clip_paths:
        try:
            merge_result = merge_video_clips(
                clip_paths=successful_clip_paths,
                output_video_path=str(task_paths.final_video_path),
            )
            final_video_path = merge_result.final_video_path
            if not merge_result.success and merge_result.error:
                errors.append(merge_result.error)
        except ClipPilotProcessingError as exc:
            errors.append(str(exc))
        except Exception as exc:
            errors.append(f"Unexpected merge failure: {exc}")
    else:
        errors.append("No clip was successfully cut, so final video merge could not run.")

    try:
        subtitle_result = generate_srt_from_editing_plan(
            editing_plan=plan,
            output_srt_path=str(task_paths.subtitle_path),
        )
        if subtitle_result.success:
            subtitle_path = subtitle_result.subtitle_path
        else:
            errors.append("Subtitle generation did not complete successfully.")
    except Exception as exc:
        errors.append(f"Subtitle generation failed: {exc}")

    if need_burn_subtitle:
        if Path(final_video_path).exists() and Path(subtitle_path).exists():
            try:
                burn_result = burn_subtitles_to_video(
                    input_video_path=final_video_path,
                    subtitle_path=subtitle_path,
                    output_video_path=str(task_paths.burned_video_path),
                )
                burned_video_path = burn_result.burned_video_path
                if not burn_result.success and burn_result.error:
                    warnings.append(f"Subtitle burning failed: {burn_result.error}")
            except Exception as exc:
                warnings.append(f"Subtitle burning failed unexpectedly: {exc}")
        else:
            warnings.append("Subtitle burning was skipped because final video or subtitle file was unavailable.")

    status = "completed" if Path(final_video_path).exists() and not errors else "failed"
    return ExecutionReport(
        task_id=task_id,
        status=status,
        clip_results=clip_results,
        final_video_path=final_video_path,
        subtitle_path=subtitle_path,
        burned_video_path=burned_video_path,
        warnings=warnings,
        errors=errors,
    )
