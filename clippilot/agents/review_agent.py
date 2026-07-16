from pathlib import Path
import re

from clippilot.core.exceptions import ClipPilotProcessingError
from clippilot.schemas.editing_plan import EditingPlan
from clippilot.schemas.execution_report import ExecutionReport
from clippilot.schemas.review_report import ReviewCheck, ReviewReport
from clippilot.schemas.user_request import UserRequest
from clippilot.schemas.video_info import VideoInfo
from clippilot.tools.video_info import extract_video_info

CRITICAL = "critical"
WARNING = "warning"


def _check(name: str, passed: bool, severity: str, message: str) -> ReviewCheck:
    """Create one structured review check entry."""

    return ReviewCheck(name=name, passed=passed, severity=severity, message=message)


def _normalize_user_request(user_request: UserRequest | dict) -> UserRequest:
    """Normalize user request payloads into a validated model."""

    if isinstance(user_request, UserRequest):
        return user_request
    return UserRequest.model_validate(user_request)


def _normalize_original_video_info(original_video_info: VideoInfo | dict) -> VideoInfo:
    """Normalize original video metadata into a validated model."""

    if isinstance(original_video_info, VideoInfo):
        return original_video_info
    return VideoInfo.model_validate(original_video_info)


def _normalize_execution_report(execution_report: ExecutionReport | dict) -> ExecutionReport:
    """Normalize execution report payloads into a validated model."""

    if isinstance(execution_report, ExecutionReport):
        return execution_report
    return ExecutionReport.model_validate(execution_report)


def _resolve_task_id(editing_plan: EditingPlan | dict, execution_report: ExecutionReport) -> str:
    """Resolve task id from the editing plan when available, otherwise fall back to execution report."""

    if isinstance(editing_plan, EditingPlan):
        return editing_plan.task_id
    if isinstance(editing_plan, dict) and editing_plan.get("task_id"):
        return str(editing_plan["task_id"])
    return execution_report.task_id


def _extract_timeline_items(editing_plan: EditingPlan | dict) -> list:
    """Return raw timeline items from either a validated editing plan or a raw dictionary payload."""

    if isinstance(editing_plan, EditingPlan):
        return list(editing_plan.timeline_items)
    if isinstance(editing_plan, dict):
        raw_items = editing_plan.get("timeline_items") or []
        return list(raw_items) if isinstance(raw_items, list) else []
    return []


def _plan_item_field(item: object, field_name: str, default: object = None) -> object:
    """Read a field from either a timeline-item model or a raw dictionary."""

    if isinstance(item, dict):
        return item.get(field_name, default)
    return getattr(item, field_name, default)


def _contains_cjk(text: str) -> bool:
    """Return whether the text contains CJK characters and should use the tighter subtitle limit."""

    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text))


def _subtitle_limit(language: str, text: str) -> int:
    """Return the subtitle character limit based on language or script."""

    if language.lower().startswith("zh") or _contains_cjk(text):
        return 30
    return 80


def _append_suggestion(suggestions: list[str], message: str) -> None:
    """Append a suggestion once while preserving insertion order."""

    if message not in suggestions:
        suggestions.append(message)


def _plan_expects_subtitles(raw_items: list[object]) -> bool:
    """Return whether the current editing plan contains any actual subtitle text."""

    for item in raw_items:
        subtitle_text = str(_plan_item_field(item, "subtitle", "") or "").strip()
        if subtitle_text:
            return True
    return False


def _score_from_checks(checks: list[ReviewCheck]) -> float:
    """Calculate the review score from failed checks."""

    score = 1.0
    for check in checks:
        if check.passed:
            continue
        if check.severity == CRITICAL:
            score -= 0.25
        elif check.severity == WARNING:
            score -= 0.1
    return round(max(0.0, score), 2)


def review_task_output(
    user_request: UserRequest | dict,
    original_video_info: VideoInfo | dict,
    editing_plan: EditingPlan | dict,
    execution_report: ExecutionReport | dict,
) -> ReviewReport:
    """Review exported task outputs and return a structured quality report."""

    user = _normalize_user_request(user_request)
    original_video = _normalize_original_video_info(original_video_info)
    execution = _normalize_execution_report(execution_report)
    task_id = _resolve_task_id(editing_plan, execution)

    checks: list[ReviewCheck] = []
    suggestions: list[str] = []
    raw_items = _extract_timeline_items(editing_plan)

    final_video_path = Path(execution.final_video_path) if execution.final_video_path else None
    subtitle_path = Path(execution.subtitle_path) if execution.subtitle_path else None

    final_video_exists = final_video_path is not None and final_video_path.exists()
    checks.append(
        _check(
            name="final_video_exists",
            passed=final_video_exists,
            severity=CRITICAL,
            message=(
                f"Final video exists at {final_video_path}."
                if final_video_exists
                else "Final video is missing. Check the executor stage and ffmpeg output."
            ),
        )
    )
    if not final_video_exists:
        _append_suggestion(suggestions, "Final video is missing. Check the executor stage and ffmpeg command outputs.")

    final_video_non_empty = final_video_exists and final_video_path is not None and final_video_path.stat().st_size > 0
    checks.append(
        _check(
            name="final_video_non_empty",
            passed=final_video_non_empty,
            severity=CRITICAL,
            message=(
                f"Final video file size is {final_video_path.stat().st_size} bytes."
                if final_video_non_empty and final_video_path is not None
                else "Final video is empty or unavailable."
            ),
        )
    )
    if not final_video_non_empty:
        _append_suggestion(suggestions, "Final video output is empty. Re-run the executor and verify merge output.")

    final_video_duration_passed = False
    final_video_duration_message = "Final video duration could not be validated because the file is unavailable."
    if final_video_exists and final_video_path is not None:
        try:
            final_video_info = extract_video_info(final_video_path)
            delta = round(final_video_info.duration_seconds - user.target_duration, 2)
            final_video_duration_passed = abs(delta) <= 5
            final_video_duration_message = (
                f"Final video duration is {final_video_info.duration_seconds:.2f}s, target is {user.target_duration}s."
            )
            if not final_video_duration_passed:
                if delta > 5:
                    _append_suggestion(
                        suggestions,
                        "Final video is too long. Remove lower-scoring timeline items from the editing plan.",
                    )
                elif delta < -5:
                    _append_suggestion(
                        suggestions,
                        "Final video is too short. Add more relevant timeline items to the editing plan.",
                    )
        except (ClipPilotProcessingError, OSError) as exc:
            final_video_duration_message = f"Failed to read final video duration: {exc}"
            _append_suggestion(suggestions, "Final video metadata could not be read. Check the merged video integrity.")
    checks.append(
        _check(
            name="final_video_duration_within_target",
            passed=final_video_duration_passed,
            severity=CRITICAL,
            message=final_video_duration_message,
        )
    )

    subtitles_expected = _plan_expects_subtitles(raw_items)
    subtitle_exists = subtitle_path is not None and subtitle_path.exists()
    checks.append(
        _check(
            name="subtitle_exists",
            passed=(subtitle_exists if subtitles_expected else True),
            severity=CRITICAL,
            message=(
                f"Subtitle file exists at {subtitle_path}."
                if subtitle_exists
                else (
                    "Subtitle file is not required because the editing plan did not include subtitle text."
                    if not subtitles_expected
                    else "Subtitle file is missing."
                )
            ),
        )
    )
    if subtitles_expected and not subtitle_exists:
        _append_suggestion(suggestions, "Subtitle file is missing. Check subtitle generation in the executor stage.")

    editing_plan_has_items = bool(raw_items)
    checks.append(
        _check(
            name="editing_plan_has_timeline_items",
            passed=editing_plan_has_items,
            severity=CRITICAL,
            message=(
                f"Editing plan contains {len(raw_items)} timeline item(s)."
                if editing_plan_has_items
                else "Editing plan does not contain any timeline items."
            ),
        )
    )
    if not editing_plan_has_items:
        _append_suggestion(
            suggestions,
            "Editing plan is empty. Re-run the planner and ensure timeline items are generated.",
        )

    for index, item in enumerate(raw_items, start=1):
        item_id = str(_plan_item_field(item, "item_id", f"item_{index:02d}"))
        start_time = _plan_item_field(item, "source_start")
        end_time = _plan_item_field(item, "source_end")

        try:
            start_value = float(start_time)
            end_value = float(end_time)
            timing_passed = start_value < end_value
            bounds_passed = 0.0 <= start_value < end_value <= original_video.duration_seconds
        except (TypeError, ValueError):
            start_value = None
            end_value = None
            timing_passed = False
            bounds_passed = False

        checks.append(
            _check(
                name=f"{item_id}_source_timing",
                passed=timing_passed,
                severity=CRITICAL,
                message=(
                    f"{item_id} uses source_start={start_value} and source_end={end_value}."
                    if timing_passed
                    else f"{item_id} has invalid timing values: source_start={start_time}, source_end={end_time}."
                ),
            )
        )
        if not timing_passed:
            _append_suggestion(
                suggestions,
                "Some timeline item timing ranges are invalid. Rebuild the editing plan before execution.",
            )

        checks.append(
            _check(
                name=f"{item_id}_within_source_bounds",
                passed=bounds_passed,
                severity=CRITICAL,
                message=(
                    f"{item_id} stays within original duration {original_video.duration_seconds:.2f}s."
                    if bounds_passed
                    else f"{item_id} exceeds the original video duration of {original_video.duration_seconds:.2f}s."
                ),
            )
        )
        if not bounds_passed:
            _append_suggestion(
                suggestions,
                "Some timeline items exceed the original video duration. Regenerate the editing plan with valid bounds.",
            )

        subtitle_text = str(_plan_item_field(item, "subtitle", "") or "").strip()
        character_limit = _subtitle_limit(user.language, subtitle_text)
        subtitle_length_passed = len(subtitle_text) <= character_limit
        checks.append(
            _check(
                name=f"{item_id}_subtitle_length",
                passed=subtitle_length_passed,
                severity=WARNING,
                message=(
                    f"{item_id} subtitle length is {len(subtitle_text)} characters within the {character_limit}-character limit."
                    if subtitle_length_passed
                    else f"{item_id} subtitle length is {len(subtitle_text)} characters and exceeds the {character_limit}-character limit."
                ),
            )
        )
        if not subtitle_length_passed:
            _append_suggestion(suggestions, "Some subtitles are too long. Split long subtitle lines for better readability.")

    execution_errors_empty = not execution.errors
    checks.append(
        _check(
            name="execution_errors_empty",
            passed=execution_errors_empty,
            severity=CRITICAL,
            message=(
                "Execution report contains no errors."
                if execution_errors_empty
                else f"Execution report contains {len(execution.errors)} error(s)."
            ),
        )
    )
    if not execution_errors_empty:
        _append_suggestion(suggestions, "Execution report contains errors. Inspect the executor stage before publishing output.")

    score = _score_from_checks(checks)
    passed = not any(not check.passed and check.severity == CRITICAL for check in checks)
    return ReviewReport(
        task_id=task_id,
        passed=passed,
        score=score,
        checks=checks,
        suggestions=suggestions,
    )


def review_outputs(task_id: str) -> ReviewReport:
    """Preserve the old helper name for compatibility with earlier imports."""

    return ReviewReport(
        task_id=task_id,
        passed=False,
        score=0.0,
        checks=[_check("review_agent_invocation", False, CRITICAL, "review_outputs() no longer has enough context to run.")],
        suggestions=["Call review_task_output() with user_request, original_video_info, editing_plan, and execution_report."],
    )
