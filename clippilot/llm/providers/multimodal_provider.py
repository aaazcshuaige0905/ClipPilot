from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class MultimodalProviderError(RuntimeError):
    """Raise when a multimodal provider request cannot be completed successfully."""


class OpenAICompatibleMultimodalProvider:
    """Call an OpenAI-compatible chat-completions endpoint for multimodal reasoning."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        api_path: str = "/chat/completions",
        timeout_seconds: int = 180,
    ) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.api_path = api_path if api_path.startswith("/") else f"/{api_path}"
        self.timeout_seconds = timeout_seconds

    def create_multimodal_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one multimodal chat-completions request and return the parsed JSON response."""

        if not self.api_key:
            raise MultimodalProviderError("Video-understanding API key is missing.")

        request = Request(
            f"{self.base_url}{self.api_path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:  # pragma: no cover - network-dependent path
            body = exc.read().decode("utf-8", errors="ignore")
            raise MultimodalProviderError(f"Multimodal request failed with HTTP {exc.code}: {body}") from exc
        except URLError as exc:  # pragma: no cover - network-dependent path
            raise MultimodalProviderError(f"Multimodal request failed: {exc}") from exc
