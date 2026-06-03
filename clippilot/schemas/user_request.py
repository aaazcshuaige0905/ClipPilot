from pydantic import BaseModel, Field


class UserRequest(BaseModel):
    """Represent the editing request submitted together with an uploaded video."""

    target_platform: str = Field(..., min_length=1, max_length=50)
    target_duration: int = Field(..., gt=0, description="Target duration in seconds.")
    edit_style: str = Field(..., min_length=1, max_length=50)
    language: str = Field(..., min_length=1, max_length=20)
    need_burn_subtitle: bool
