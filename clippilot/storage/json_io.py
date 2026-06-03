import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def write_json_file(path: Path, payload: BaseModel | dict[str, Any]) -> Path:
    """Persist a model or dictionary to JSON using UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = payload.model_dump() if isinstance(payload, BaseModel) else payload
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return path


def read_json_file(path: Path) -> dict[str, Any]:
    """Load a JSON file from disk and return its parsed dictionary payload."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
