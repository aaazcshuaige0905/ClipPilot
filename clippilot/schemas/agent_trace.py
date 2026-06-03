from datetime import datetime, timezone

from pydantic import BaseModel, Field


class AgentTrace(BaseModel):
    """Represent a future trace item for agentic workflow execution."""

    agent_name: str
    stage: str
    message: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
