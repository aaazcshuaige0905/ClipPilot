from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from clippilot.storage.path_manager import AppSettings


class EmbeddingProviderError(RuntimeError):
    """Raise when the embedding provider cannot serve embeddings."""


@dataclass(frozen=True)
class EmbeddingUsage:
    """Track provider usage information for debugging."""

    prompt_tokens: int = 0
    total_tokens: int = 0


class BaseEmbeddingProvider(ABC):
    """Define the common embedding provider interface used by RAG modules."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed stored corpus documents."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed one user query for dense retrieval."""


class OpenAICompatibleEmbeddingProvider(BaseEmbeddingProvider):
    """Provide a generic OpenAI-compatible embeddings client over raw HTTP."""

    def __init__(self, api_key: str, base_url: str, model: str, dimensions: int = 1024) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts via the compatible embeddings endpoint."""

        return self._request_embeddings(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed one query text via the compatible embeddings endpoint."""

        vectors = self._request_embeddings([text])
        return vectors[0] if vectors else []

    def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Call the compatible endpoint and return vectors in input order."""

        if not self.api_key:
            raise EmbeddingProviderError("Embedding API key is missing.")
        payload = {
            "model": self.model,
            "input": texts,
            "dimensions": self.dimensions,
            "encoding_format": "float",
        }
        response = self._post_json(f"{self.base_url}/embeddings", payload)
        vectors = [item["embedding"] for item in sorted(response.get("data", []), key=lambda item: item.get("index", 0))]
        if len(vectors) != len(texts):
            raise EmbeddingProviderError("Embedding response length does not match request length.")
        return vectors

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one JSON POST request to the configured embedding endpoint."""

        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:  # pragma: no cover - network-dependent path
            body = exc.read().decode("utf-8", errors="ignore")
            raise EmbeddingProviderError(f"Embedding request failed with HTTP {exc.code}: {body}") from exc
        except URLError as exc:  # pragma: no cover - network-dependent path
            raise EmbeddingProviderError(f"Embedding request failed: {exc}") from exc


class QwenCloudEmbeddingProvider(BaseEmbeddingProvider):
    """Call Qwen Cloud's DashScope text embedding API with query/document modes."""

    def __init__(self, settings: AppSettings) -> None:
        self.api_key = settings.qwen_api_key
        self.model = settings.qwen_embedding_model
        self.dimensions = settings.qwen_embedding_dimensions
        self.base_url = settings.qwen_embedding_base_url.rstrip("/")
        self.api_path = settings.qwen_embedding_api_path

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed strategy documents using `text_type=document`."""

        return self._request_embeddings(texts, text_type="document")

    def embed_query(self, text: str) -> list[float]:
        """Embed one retrieval query using `text_type=query`."""

        vectors = self._request_embeddings([text], text_type="query")
        return vectors[0] if vectors else []

    def _request_embeddings(self, texts: list[str], text_type: str) -> list[list[float]]:
        """Call the DashScope embedding endpoint and return vectors in order."""

        if not self.api_key:
            raise EmbeddingProviderError("DASHSCOPE_API_KEY or CLIP_PILOT_QWEN_API_KEY is required for Qwen embeddings.")

        payload = {
            "model": self.model,
            "input": {"texts": texts},
            "parameters": {
                "text_type": text_type,
                "dimension": self.dimensions,
                "output_type": "dense",
            },
        }
        response = self._post_json(f"{self.base_url}{self.api_path}", payload)
        embeddings = response.get("output", {}).get("embeddings", [])
        vectors = [item["embedding"] for item in embeddings]
        if len(vectors) != len(texts):
            raise EmbeddingProviderError("DashScope embedding response length does not match request length.")
        return vectors

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one JSON POST request to the DashScope embedding endpoint."""

        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:  # pragma: no cover - network-dependent path
            body = exc.read().decode("utf-8", errors="ignore")
            raise EmbeddingProviderError(f"Qwen embedding request failed with HTTP {exc.code}: {body}") from exc
        except URLError as exc:  # pragma: no cover - network-dependent path
            raise EmbeddingProviderError(f"Qwen embedding request failed: {exc}") from exc
