"""Retrieval-augmented planning helpers for ClipPilot."""

from clippilot.rag.build_index import build_local_index
from clippilot.rag.retrieve import retrieve_context, retrieve_knowledge
from clippilot.rag.schemas import KnowledgeChunk, RetrievedContext, RetrievalQuery

__all__ = [
    "KnowledgeChunk",
    "RetrievedContext",
    "RetrievalQuery",
    "build_local_index",
    "retrieve_context",
    "retrieve_knowledge",
]
