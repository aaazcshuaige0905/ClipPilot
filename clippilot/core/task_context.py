from dataclasses import dataclass, field

from clippilot.core.states import TaskStatus, WorkflowStage
from clippilot.schemas.agent_trace import AgentTrace
from clippilot.schemas.project_state import ProjectState
from clippilot.schemas.task_result import TaskArtifact, TaskArtifactManifest
from clippilot.schemas.user_request import UserRequest
from clippilot.storage.path_manager import TaskPaths


@dataclass
class TaskContext:
    """Keep task-scoped request, paths, stage, and artifact references together."""

    task_id: str
    request: UserRequest
    paths: TaskPaths
    stage: str = WorkflowStage.CREATED.value
    status: str = TaskStatus.PENDING.value
    project_state: ProjectState | None = None
    artifacts: list[TaskArtifact] = field(default_factory=list)
    traces: list[AgentTrace] = field(default_factory=list)

    def mark_stage(self, stage: str) -> None:
        """Update the current workflow stage for the task."""

        self.stage = stage
        if self.project_state is not None:
            self.project_state.current_step = stage

    def mark_status(self, status: str) -> None:
        """Update the coarse-grained task status."""

        self.status = status
        if self.project_state is not None:
            self.project_state.status = status.upper()

    def add_artifact(self, name: str, path: str, stage: str | None = None) -> None:
        """Register a saved artifact path under a stable name."""

        self.artifacts.append(
            TaskArtifact(
                name=name,
                stage=stage or self.stage,
                relative_path=path,
            )
        )

    def add_trace(self, stage: str, message: str) -> None:
        """Register a structured trace entry in memory."""

        self.traces.append(
            AgentTrace(
                agent_name="workflow",
                stage=stage,
                message=message,
            )
        )
        if self.project_state is not None:
            self.project_state.logs.append(f"[{stage}] {message}")

    def add_error(self, message: str) -> None:
        """Register a workflow error in the shared project state."""

        if self.project_state is not None:
            self.project_state.errors.append(message)

    def build_artifact_manifest(self) -> TaskArtifactManifest:
        """Build a manifest model for the current task artifacts."""

        return TaskArtifactManifest(
            task_id=self.task_id,
            stage=self.stage,
            artifacts=self.artifacts,
        )
