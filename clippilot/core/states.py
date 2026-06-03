from enum import StrEnum


class TaskStatus(StrEnum):
    """Represent the coarse-grained lifecycle status of a task."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowStage(StrEnum):
    """Represent the fine-grained workflow stage of a task."""

    CREATED = "created"
    UPLOADED = "uploaded"
    VIDEO_INFO_EXTRACTED = "video_info_extracted"
    TRANSCRIBED = "transcribed"
    HIGHLIGHT_CANDIDATES_GENERATED = "highlight_candidates_generated"
    EDITING_PLAN_GENERATED = "editing_plan_generated"
    EXECUTION_REPORT_GENERATED = "execution_report_generated"
    REVIEW_REPORT_GENERATED = "review_report_generated"
    COMPLETED = "completed"
