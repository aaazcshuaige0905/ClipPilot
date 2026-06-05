from pathlib import Path

from clippilot.core.exceptions import ClipPilotProcessingError
from clippilot.schemas.editing_plan import EditingPlan, TimelineItem
from clippilot.schemas.execution_report import ExecutionItemResult, ExecutionReport
from clippilot.storage.path_manager import TaskPaths
from clippilot.tools.subtitle import burn_subtitles_to_video, generate_srt_from_editing_plan
from clippilot.tools.video_cut import cut_video_clip
from clippilot.tools.video_merge import merge_video_clips


def _item_output_path(task_paths: TaskPaths, item_id: str) -> Path:
    """Build the output path for one rendered timeline item inside the task clips directory."""

    return task_paths.clips_dir / f"{item_id}.mp4"


def _item_part_output_path(task_paths: TaskPaths, item_id: str, part_index: int) -> Path:
    """Build the output path for one source-ref part within a montage timeline item."""

    return task_paths.clips_dir / f"{item_id}_part_{part_index:02d}.mp4"


def _normalize_editing_plan(editing_plan: EditingPlan | dict) -> EditingPlan:
    """Normalize editing plan payloads into a validated EditingPlan model."""

    if isinstance(editing_plan, EditingPlan):
        return editing_plan
    return EditingPlan.model_validate(editing_plan)


def _render_single_range_item(
    item: TimelineItem,
    source_video_path: str,
    output_item_path: Path,
) -> ExecutionItemResult:
    """Render one continuous-range timeline item into a standalone media file."""

    cut_result = cut_video_clip(
        input_video_path=source_video_path,
        output_clip_path=str(output_item_path),
        start_time=item.source_start,
        end_time=item.source_end,
    )
    return ExecutionItemResult(
        item_id=item.item_id,
        source_start=item.source_start,
        source_end=item.source_end,
        item_path=cut_result.clip_path,
        duration=cut_result.duration,
        success=cut_result.success,
        error=cut_result.error,
    )


def _render_montage_item(
    item: TimelineItem,
    source_video_path: str,
    task_paths: TaskPaths,
    output_item_path: Path,
) -> ExecutionItemResult:
    """Render one montage timeline item by cutting each source ref and concatenating them."""

    part_paths: list[str] = []
    for part_index, reference in enumerate(item.source_refs, start=1):
        part_output_path = _item_part_output_path(task_paths, item.item_id, part_index)
        part_result = cut_video_clip(
            input_video_path=source_video_path,
            output_clip_path=str(part_output_path),
            start_time=reference.start,
            end_time=reference.end,
        )
        if not part_result.success:
            return ExecutionItemResult(
                item_id=item.item_id,
                source_start=item.source_start,
                source_end=item.source_end,
                item_path=str(output_item_path),
                duration=0.0,
                success=False,
                error=part_result.error or f"Failed to cut montage part {part_index}.",
            )
        part_paths.append(part_result.clip_path)

    merge_result = merge_video_clips(
        clip_paths=part_paths,
        output_video_path=str(output_item_path),
    )
    return ExecutionItemResult(
        item_id=item.item_id,
        source_start=item.source_start,
        source_end=item.source_end,
        item_path=merge_result.final_video_path,
        duration=item.duration,
        success=merge_result.success,
        error=merge_result.error,
    )


def _render_timeline_item(
    item: TimelineItem,
    source_video_path: str,
    task_paths: TaskPaths,
) -> ExecutionItemResult:
    """Render one timeline item using the single planner-defined execution logic."""

    output_item_path = _item_output_path(task_paths, item.item_id)
    if item.assembly_mode == "montage" and len(item.source_refs) > 1:
        return _render_montage_item(
            item=item,
            source_video_path=source_video_path,
            task_paths=task_paths,
            output_item_path=output_item_path,
        )
    return _render_single_range_item(
        item=item,
        source_video_path=source_video_path,
        output_item_path=output_item_path,
    )


def execute_editing_plan(
    task_id: str,
    source_video_path: str,
    editing_plan: EditingPlan | dict,
    task_paths: TaskPaths,
    need_burn_subtitle: bool,
) -> ExecutionReport:
    """Execute the editing plan by rendering timeline items, merging them, and generating subtitles."""

    plan = _normalize_editing_plan(editing_plan)
    item_results: list[ExecutionItemResult] = []
    warnings: list[str] = []
    errors: list[str] = []
    successful_item_paths: list[str] = []
    final_video_path = str(task_paths.final_video_path)
    subtitle_path = str(task_paths.subtitle_path)
    burned_video_path = str(task_paths.burned_video_path) if need_burn_subtitle else None

    for item in plan.timeline_items:
        try:
            item_result = _render_timeline_item(
                item=item,
                source_video_path=source_video_path,
                task_paths=task_paths,
            )
        except Exception as exc:
            errors.append(f"{item.item_id}: unexpected render failure: {exc}")
            item_results.append(
                ExecutionItemResult(
                    item_id=item.item_id,
                    source_start=item.source_start,
                    source_end=item.source_end,
                    item_path=str(_item_output_path(task_paths, item.item_id)),
                    duration=0.0,
                    success=False,
                    error=f"Unexpected render failure: {exc}",
                )
            )
            continue

        item_results.append(item_result)
        if item_result.success:
            successful_item_paths.append(item_result.item_path)
        elif item_result.error:
            errors.append(f"{item.item_id}: {item_result.error}")

    if len(successful_item_paths) < len(plan.timeline_items):
        warnings.append("Some timeline items failed to render and were excluded from the final merge.")

    if successful_item_paths:
        try:
            merge_result = merge_video_clips(
                clip_paths=successful_item_paths,
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
        errors.append("No timeline item was successfully rendered, so final video merge could not run.")

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
        item_results=item_results,
        final_video_path=final_video_path,
        subtitle_path=subtitle_path,
        burned_video_path=burned_video_path,
        warnings=warnings,
        errors=errors,
    )
