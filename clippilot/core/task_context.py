from dataclasses import dataclass, field

from clippilot.core.states import TaskStatus, WorkflowStage
from clippilot.schemas.agent_trace import AgentTrace
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
    artifacts: list[TaskArtifact] = field(default_factory=list)
    traces: list[AgentTrace] = field(default_factory=list)

    def mark_stage(self, stage: str) -> None:
        """Update the current workflow stage for the task."""

        self.stage = stage

    def mark_status(self, status: str) -> None:
        """Update the coarse-grained task status."""

        self.status = status

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

    def build_artifact_manifest(self) -> TaskArtifactManifest:
        """Build a manifest model for the current task artifacts."""

        return TaskArtifactManifest(
            task_id=self.task_id,
            stage=self.stage,
            artifacts=self.artifacts,
        )
