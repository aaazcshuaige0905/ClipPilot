from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from clippilot.rag.schemas import KnowledgeChunk, RetrievalQuery


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text into lexical search terms."""

    if not text:
        return []
    normalized = text.lower()
    tokens: list[str] = []
    tokens.extend(re.findall(r"[a-z0-9_]+", normalized))
    tokens.extend(char for char in normalized if "\u4e00" <= char <= "\u9fff")
    return tokens


class BM25Store:
    """A lightweight persisted BM25 implementation for strategy chunks."""

    def __init__(self, chunks: list[KnowledgeChunk]) -> None:
        self.chunks = chunks
        self.doc_tokens = [tokenize(f"{chunk.title} {chunk.text} {' '.join(chunk.tags)}") for chunk in chunks]
        self.doc_lengths = [max(1, len(tokens)) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.term_document_frequency: dict[str, int] = defaultdict(int)
        self.term_frequencies: list[Counter[str]] = []

        for tokens in self.doc_tokens:
            counter = Counter(tokens)
            self.term_frequencies.append(counter)
            for token in counter:
                self.term_document_frequency[token] += 1

    def save(self, path: Path) -> Path:
        """Persist the BM25 corpus payload for later reuse."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump({"chunks": [chunk.model_dump() for chunk in self.chunks]}, file, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path: Path) -> "BM25Store":
        """Load a persisted BM25 corpus payload."""

        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [KnowledgeChunk.model_validate(item) for item in payload.get("chunks", [])]
        return cls(chunks)

    def search(self, query: RetrievalQuery, top_k: int | None = None) -> list[tuple[KnowledgeChunk, float]]:
        """Return BM25-ranked chunks filtered by basic metadata compatibility."""

        requested_top_k = top_k or query.top_k_bm25
        query_tokens = tokenize(query.text)
        if not query_tokens or not self.chunks:
            return []

        scored: list[tuple[KnowledgeChunk, float]] = []
        for index, chunk in enumerate(self.chunks):
            if not _matches_chunk_metadata(chunk, query):
                continue
            score = self._bm25_score(query_tokens, index)
            if score <= 0:
                continue
            scored.append((chunk, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:requested_top_k]

    def _bm25_score(self, query_tokens: list[str], index: int, k1: float = 1.5, b: float = 0.75) -> float:
        """Compute the BM25 score for one query-document pair."""

        score = 0.0
        doc_length = self.doc_lengths[index]
        frequencies = self.term_frequencies[index]
        for token in query_tokens:
            if token not in frequencies:
                continue
            document_frequency = self.term_document_frequency.get(token, 0)
            idf = math.log(1 + (len(self.chunks) - document_frequency + 0.5) / (document_frequency + 0.5))
            term_frequency = frequencies[token]
            numerator = term_frequency * (k1 + 1)
            denominator = term_frequency + k1 * (1 - b + b * (doc_length / max(self.avg_doc_length, 1.0)))
            score += idf * (numerator / denominator)
        return round(score, 6)


def _matches_chunk_metadata(chunk: KnowledgeChunk, query: RetrievalQuery) -> bool:
    """Apply strict-enough metadata filtering before lexical ranking."""

    if chunk.stage and chunk.stage != query.stage:
        return False
    if query.platform and chunk.platform and chunk.platform != query.platform:
        return False
    if query.language and chunk.language and chunk.language != query.language:
        return False
    if query.style and chunk.style and chunk.style != query.style:
        return False
    if query.duration_band and chunk.duration_band and chunk.duration_band != query.duration_band:
        return False
    return True
