from pydantic import BaseModel, Field

from clippilot.core.states import TaskStatus, WorkflowStage
from clippilot.schemas.user_request import UserRequest
from clippilot.schemas.video_info import VideoInfo


class TaskArtifact(BaseModel):
    """Represent a single named artifact produced during task processing."""

    name: str
    stage: str
    relative_path: str


class TaskArtifactManifest(BaseModel):
    """Represent the saved artifact inventory for a task."""

    task_id: str
    stage: str
    artifacts: list[TaskArtifact] = Field(default_factory=list)


class TaskResult(BaseModel):
    """Represent the public API response returned after stage-1 processing."""

    task_id: str
    status: str = TaskStatus.COMPLETED.value
    stage: str = WorkflowStage.COMPLETED.value
    upload_file_path: str
    user_params: UserRequest
    metadata: VideoInfo
    transcript_path: str
    candidates_path: str
    editing_plan_path: str
    project_state_path: str | None = None
    timeline_path: str | None = None
    execution_report_path: str | None = None
    final_video_path: str | None = None
    subtitle_path: str | None = None
    burned_video_path: str | None = None


class TaskStatusResponse(BaseModel):
    """Represent a query response for an already processed task."""

    task_id: str
    status: str
    stage: str
    task_root_path: str
    task_result: TaskResult
    artifact_manifest: TaskArtifactManifest


class TaskSummary(BaseModel):
    """Represent a lightweight summary of a processed task."""

    task_id: str
    status: str
    stage: str
    task_root_path: str
    upload_file_path: str


class TaskListResponse(BaseModel):
    """Represent a paginated-style list of processed task summaries."""

    tasks: list[TaskSummary] = Field(default_factory=list)
