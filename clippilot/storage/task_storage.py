import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4
import json

from pydantic import BaseModel

if TYPE_CHECKING:
    from fastapi import UploadFile
else:
    UploadFile = Any

from clippilot.core.exceptions import ClipPilotStorageError
from clippilot.core.task_context import TaskContext
from clippilot.schemas.task_result import (
    TaskArtifactManifest,
    TaskListResponse,
    TaskResult,
    TaskStatusResponse,
    TaskSummary,
)
from clippilot.storage.json_io import read_json_file, write_json_file
from clippilot.storage.path_manager import AppSettings, TaskPaths, build_task_paths, ensure_base_directories, load_settings


class TaskStorage:
    """Create task folders and persist task artifacts in the structured task layout."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or load_settings()
        ensure_base_directories(self.settings)

    def create_task_id(self) -> str:
        """Create a unique task identifier."""

        return uuid4().hex

    def create_task_paths(self, task_id: str, original_file_name: str) -> TaskPaths:
        """Create and return all directories for a task."""

        task_paths = build_task_paths(self.settings, task_id, original_file_name)
        for directory in (
            task_paths.task_root,
            task_paths.input_dir,
            task_paths.metadata_dir,
            task_paths.transcript_dir,
            task_paths.highlights_dir,
            task_paths.clips_dir,
            task_paths.final_dir,
            task_paths.plan_dir,
            task_paths.review_dir,
            task_paths.trace_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return task_paths

    def save_upload_file(self, upload_file: UploadFile, task_paths: TaskPaths) -> Path:
        """Persist an uploaded file to the structured task input directory."""

        try:
            with task_paths.source_video_path.open("wb") as destination:
                shutil.copyfileobj(upload_file.file, destination)
        except OSError as exc:
            raise ClipPilotStorageError("Failed to save the uploaded video file.") from exc
        finally:
            upload_file.file.close()

        return task_paths.source_video_path

    def save_model(self, payload: BaseModel | dict, path: Path) -> Path:
        """Persist a model or dictionary artifact to the structured task directory."""

        return write_json_file(path, payload)

    def save_transcript(self, transcript: BaseModel | dict, task_paths: TaskPaths) -> Path:
        """Persist transcript artifacts in the structured task directory."""

        return self.save_model(transcript, task_paths.transcript_json_path)

    def save_candidates(self, candidates: BaseModel | dict, task_paths: TaskPaths) -> Path:
        """Persist highlight candidates in the structured task directory."""

        return self.save_model(candidates, task_paths.highlight_candidates_path)

    def save_video_info(self, video_info: BaseModel | dict, task_paths: TaskPaths) -> Path:
        """Persist extracted video metadata inside the structured task directory."""

        return self.save_model(video_info, task_paths.video_info_path)

    def save_task_result(self, task_result: BaseModel | dict, task_paths: TaskPaths) -> Path:
        """Persist the public workflow result inside the task root for debugging."""

        return self.save_model(task_result, task_paths.task_result_path)

    def save_editing_plan(self, editing_plan: BaseModel | dict, task_paths: TaskPaths) -> Path:
        """Persist an editing plan artifact inside the structured task directory."""

        return self.save_model(editing_plan, task_paths.editing_plan_path)

    def save_execution_report(self, execution_report: BaseModel | dict, task_paths: TaskPaths) -> Path:
        """Persist an execution report artifact inside the structured task directory."""

        return self.save_model(execution_report, task_paths.execution_report_path)

    def save_review_report(self, review_report: BaseModel | dict, task_paths: TaskPaths) -> Path:
        """Persist a review report artifact inside the structured task directory."""

        return self.save_model(review_report, task_paths.review_report_path)

    def save_artifact_manifest(self, context: TaskContext) -> Path:
        """Persist the current task artifact manifest inside the task root."""

        manifest = context.build_artifact_manifest()
        return self.save_model(manifest, context.paths.artifact_manifest_path)

    def append_trace_entries(self, context: TaskContext) -> Path:
        """Persist structured workflow trace entries in JSONL format."""

        trace_path = context.paths.trace_log_path
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", encoding="utf-8") as file:
            for trace in context.traces:
                file.write(json.dumps(trace.model_dump(), ensure_ascii=False) + "\n")
        return trace_path

    def task_paths_for(self, task_id: str) -> TaskPaths:
        """Rebuild task paths for an existing task using its saved input file if available."""

        task_root = self.settings.tasks_root_dir / task_id
        input_dir = task_root / "input"
        source_files = sorted(path for path in input_dir.glob("source.*") if path.is_file())
        source_name = source_files[0].name if source_files else "source.mp4"
        return build_task_paths(self.settings, task_id, source_name)

    def load_task_result(self, task_id: str) -> TaskResult:
        """Load a previously saved task result by task identifier."""

        task_paths = self.task_paths_for(task_id)
        if not task_paths.task_result_path.exists():
            raise ClipPilotStorageError(f"Task result was not found for task_id={task_id}.")
        return TaskResult.model_validate(read_json_file(task_paths.task_result_path))

    def load_artifact_manifest(self, task_id: str) -> TaskArtifactManifest:
        """Load a previously saved artifact manifest by task identifier."""

        task_paths = self.task_paths_for(task_id)
        if not task_paths.artifact_manifest_path.exists():
            raise ClipPilotStorageError(f"Artifact manifest was not found for task_id={task_id}.")
        return TaskArtifactManifest.model_validate(read_json_file(task_paths.artifact_manifest_path))

    def load_task_status(self, task_id: str) -> TaskStatusResponse:
        """Load a full task status snapshot including task result and artifact manifest."""

        task_paths = self.task_paths_for(task_id)
        task_result = self.load_task_result(task_id)
        artifact_manifest = self.load_artifact_manifest(task_id)
        return TaskStatusResponse(
            task_id=task_id,
            status=task_result.status,
            stage=task_result.stage,
            task_root_path=self.as_relative(task_paths.task_root),
            task_result=task_result,
            artifact_manifest=artifact_manifest,
        )

    def list_tasks(self, limit: int = 20) -> TaskListResponse:
        """List processed tasks by reading saved task result files under the tasks root."""

        task_summaries: list[TaskSummary] = []
        task_dirs = [path for path in self.settings.tasks_root_dir.iterdir() if path.is_dir()]
        task_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)

        for task_dir in task_dirs[:limit]:
            task_id = task_dir.name
            try:
                task_result = self.load_task_result(task_id)
                task_summaries.append(
                    TaskSummary(
                        task_id=task_result.task_id,
                        status=task_result.status,
                        stage=task_result.stage,
                        task_root_path=self.as_relative(task_dir),
                        upload_file_path=task_result.upload_file_path,
                    )
                )
            except ClipPilotStorageError:
                continue

        return TaskListResponse(tasks=task_summaries)

    def as_relative(self, path: Path) -> str:
        """Return a project-relative path string for API responses."""

        return str(path.relative_to(self.settings.project_root)).replace("\\", "/")
