from pathlib import Path


def export_final_video(task_id: str, output_dir: Path) -> Path:
    """Return the future final export path for a task."""

    return output_dir / f"{task_id}_final.mp4"
