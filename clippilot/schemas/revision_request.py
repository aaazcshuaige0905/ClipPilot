from pydantic import BaseModel, Field


class RevisionRequest(BaseModel):
    """Represent one revision request sent for an existing task."""

    feedback_text: str = Field(..., min_length=1)
