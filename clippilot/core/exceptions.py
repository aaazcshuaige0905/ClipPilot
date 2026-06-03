class ClipPilotError(Exception):
    """Base exception for domain-level ClipPilot failures."""


class ClipPilotValidationError(ClipPilotError):
    """Raise when user input or intermediate results fail validation."""


class ClipPilotProcessingError(ClipPilotError):
    """Raise when a processing step cannot complete successfully."""


class ClipPilotStorageError(ClipPilotError):
    """Raise when an artifact cannot be persisted or copied."""
