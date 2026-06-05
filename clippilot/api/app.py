from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status

from clippilot.core.exceptions import (
    ClipPilotProcessingError,
    ClipPilotStorageError,
    ClipPilotValidationError,
)
from clippilot.agents.revision_agent import revise_editing_plan
from clippilot.core.workflow import run_stage1_workflow
from clippilot.schemas.revision_request import RevisionRequest
from clippilot.schemas.task_result import TaskListResponse, TaskResult, TaskStatusResponse
from clippilot.schemas.user_request import UserRequest
from clippilot.storage.path_manager import ensure_base_directories, load_settings
from clippilot.storage.task_storage import TaskStorage

settings = load_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Create required base directories before serving requests."""

    ensure_base_directories(settings)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> dict:
    """Return a simple health status for service verification."""

    return {"status": "ok"}


@app.get("/api/v1/tasks", response_model=TaskListResponse)
async def list_tasks(limit: int = 20) -> TaskListResponse:
    """List recently processed tasks from the structured task directory."""

    try:
        storage = TaskStorage(settings)
        return storage.list_tasks(limit=limit)
    except ClipPilotStorageError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@app.get("/api/v1/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """Load a previously processed task result and artifact manifest by task ID."""

    try:
        storage = TaskStorage(settings)
        return storage.load_task_status(task_id)
    except ClipPilotStorageError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@app.post("/api/v1/tasks/upload", response_model=TaskResult)
async def upload_video(
    file: UploadFile = File(..., description="Video file in MP4, MOV, or MKV format."),
    target_platform: str = Form(...),
    target_duration: int = Form(...),
    edit_style: str = Form(...),
    language: str = Form(...),
    need_burn_subtitle: bool = Form(...),
) -> TaskResult:
    """Accept a video upload and execute the current stage-1 processing workflow."""

    user_request = UserRequest(
        target_platform=target_platform,
        target_duration=target_duration,
        edit_style=edit_style,
        language=language,
        need_burn_subtitle=need_burn_subtitle,
    )
    try:
        return run_stage1_workflow(upload_file=file, user_request=user_request, settings=settings)
    except ClipPilotValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ClipPilotProcessingError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except ClipPilotStorageError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@app.post("/api/v1/tasks/{task_id}/revise", response_model=TaskStatusResponse)
async def revise_task(task_id: str, payload: RevisionRequest) -> TaskStatusResponse:
    """Apply revision feedback to an existing task and return its refreshed status snapshot."""

    try:
        revise_editing_plan(task_id=task_id, feedback_text=payload.feedback_text, settings=settings)
        storage = TaskStorage(settings)
        return storage.load_task_status(task_id)
    except ClipPilotValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ClipPilotStorageError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ClipPilotProcessingError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
