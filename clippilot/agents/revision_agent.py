from __future__ import annotations

from clippilot.agents.planner_agent import build_editing_plan
from clippilot.core.memory_manager import apply_revision_feedback, record_plan_generation
from clippilot.core.states import TaskStatus, WorkflowStage
from clippilot.core.exceptions import ClipPilotStorageError
from clippilot.schemas.editing_plan import EditingPlan
from clippilot.schemas.user_request import UserRequest
from clippilot.storage.path_manager import AppSettings, load_settings
from clippilot.storage.task_storage import TaskStorage


def _effective_user_request(base_request: UserRequest, overrides: dict[str, object]) -> UserRequest:
    """Build the effective user request after planner-memory overrides are applied."""

    allowed_keys = {
        "target_platform",
        "target_duration",
        "edit_style",
        "language",
        "need_burn_subtitle",
    }
    payload = base_request.model_dump()
    for key, value in overrides.items():
        if key in allowed_keys and value is not None:
            payload[key] = value
    return UserRequest.model_validate(payload)


def revise_editing_plan(
    task_id: str,
    feedback_text: str,
    settings: AppSettings | None = None,
) -> EditingPlan:
    """Apply revision feedback to planner memory and save a new editing-plan version."""

    active_settings = settings or load_settings()
    storage = TaskStorage(active_settings)
    task_paths = storage.task_paths_for(task_id)
    project_state = storage.load_project_state(task_id)
    planner_memory = storage.load_planner_memory(task_id)
    video_info = storage.load_video_info(task_id)
    transcript = storage.load_transcript(task_id)
    candidates = storage.load_candidates(task_id)
    current_plan = storage.load_editing_plan(task_id)

    turn_id = f"revision_{current_plan.plan_version + 1:02d}"
    parsed_feedback = apply_revision_feedback(
        planner_memory,
        feedback_text=feedback_text,
        turn_id=turn_id,
    )
    effective_request = _effective_user_request(project_state.user_request, planner_memory.active_constraints)

    revised_plan = build_editing_plan(
        task_id=task_id,
        user_request=effective_request,
        video_info=video_info,
        transcript=transcript,
        candidates=candidates,
        fine_grained_units=project_state.fine_grained_units,
        timeline=project_state.timeline,
        llm_candidates=project_state.highlight_candidates_llm,
        retrieved_context=project_state.retrieved_context,
        planner_memory=planner_memory,
    )
    revised_plan.editing_notes.append(f"Revision feedback applied: {feedback_text}")
    if parsed_feedback["constraints"]:
        revised_plan.editing_notes.append(
            "Revision constraint overrides: "
            + ", ".join(f"{key}={value}" for key, value in parsed_feedback["constraints"].items())
            + "."
        )

    storage.save_editing_plan(revised_plan, task_paths)
    storage.save_plan_version(revised_plan, task_paths, revised_plan.plan_version)
    record_plan_generation(planner_memory, revised_plan)
    planner_memory.touch()
    storage.save_planner_memory(planner_memory, task_paths)

    project_state.user_request = effective_request
    project_state.status = TaskStatus.PROCESSING.value.upper()
    project_state.current_step = WorkflowStage.EDITING_PLAN_GENERATED.value
    project_state.logs.append(
        f"[revision_requested] Applied feedback and regenerated plan v{revised_plan.plan_version}: {feedback_text}"
    )
    project_state.human_review = {
        "latest_feedback": feedback_text,
        "latest_turn_id": turn_id,
        "parsed_feedback": parsed_feedback,
    }
    project_state.paths.final_video = None
    project_state.render_output = None
    project_state.qc_report = None
    storage.save_project_state(project_state, task_paths)

    try:
        task_result = storage.load_task_result(task_id)
    except ClipPilotStorageError:
        return revised_plan

    updated_task_result = task_result.model_copy(
        update={
            "status": TaskStatus.PROCESSING.value,
            "stage": WorkflowStage.EDITING_PLAN_GENERATED.value,
            "user_params": effective_request,
            "editing_plan_path": storage.as_relative(task_paths.editing_plan_path),
            "planner_memory_path": storage.as_relative(task_paths.planner_memory_path),
            "execution_report_path": None,
            "final_video_path": None,
            "subtitle_path": None,
            "burned_video_path": None,
        }
    )
    storage.save_task_result(updated_task_result, task_paths)
    try:
        artifact_manifest = storage.load_artifact_manifest(task_id)
        updated_manifest = artifact_manifest.model_copy(update={"stage": WorkflowStage.EDITING_PLAN_GENERATED.value})
        storage.save_model(updated_manifest, task_paths.artifact_manifest_path)
    except ClipPilotStorageError:
        pass
    return revised_plan
