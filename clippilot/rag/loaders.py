from __future__ import annotations

from pathlib import Path

from clippilot.rag.chunking import chunk_markdown_document
from clippilot.rag.schemas import KnowledgeChunk
from clippilot.storage.path_manager import AppSettings


def iter_knowledge_files(knowledge_dir: Path) -> list[Path]:
    """Return all markdown files under the configured knowledge directory."""

    if not knowledge_dir.exists():
        return []
    return sorted(path for path in knowledge_dir.rglob("*.md") if path.is_file())


def load_knowledge_chunks(settings: AppSettings) -> list[KnowledgeChunk]:
    """Load and chunk the complete strategy knowledge base."""

    chunks: list[KnowledgeChunk] = []
    for path in iter_knowledge_files(settings.rag_knowledge_dir):
        content = path.read_text(encoding="utf-8")
        chunks.extend(chunk_markdown_document(path, content))
    return chunks
