from clippilot.core.task_context import TaskContext


def log_stage_transition(context: TaskContext, stage: str, message: str) -> None:
    """Record a simple stage transition message in task memory."""

    context.mark_stage(stage)
    context.add_trace(stage=stage, message=message)
