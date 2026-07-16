from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

from clippilot.rag.metadata_normalization import normalize_language, normalize_platform, normalize_style
from clippilot.rag.schemas import KnowledgeChunk

MAX_CHUNK_CHARS = 320
MIN_CHUNK_CHARS = 80

PLATFORM_KEYWORDS = {
    "douyin": ("douyin", "抖音"),
    "bilibili": ("bilibili", "哔哩哔哩", "b站"),
    "xiaohongshu": ("xiaohongshu", "小红书"),
}
STYLE_KEYWORDS = {
    "powerful": ("powerful", "强节奏", "冲击", "热血"),
    "healing": ("healing", "治愈", "舒缓", "温柔"),
    "educational": ("educational", "知识", "讲解", "科普"),
}
LANGUAGE_KEYWORDS = {
    "zh": ("中文", "字幕", "汉字"),
    "en": ("english", "英文"),
}
STRATEGY_HINTS = {
    "hook_rule": ("hook", "开头", "前三秒", "前3秒"),
    "subtitle_rule": ("subtitle", "字幕"),
    "pacing_rule": ("节奏", "pacing", "转折", "剪辑"),
    "ending_rule": ("结尾", "cta", "收束"),
}
DURATION_HINTS = {
    "0_15": ("15秒", "15 s", "0-15"),
    "15_30": ("30秒", "15-30"),
    "30_60": ("60秒", "30-60"),
}


class ChunkMetadata(TypedDict):
    """Represent inferred retrieval metadata for one chunk."""

    knowledge_type: str
    strategy_type: str
    platform: str | None
    language: str | None
    style: str | None
    duration_band: str | None
    priority: str
    tags: list[str]


def _slugify(text: str) -> str:
    """Convert a text fragment into a stable ASCII-friendly slug."""

    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return normalized.strip("_") or "chunk"


def _compact(text: str) -> str:
    """Collapse whitespace while preserving sentence content."""

    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> list[str]:
    """Split mixed Chinese and English content into sentence-like units."""

    parts = re.split(r"(?<=[。！？.!?])\s*", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _split_long_block(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split one oversized block while keeping rule text reasonably coherent."""

    sentences = _split_sentences(text)
    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > max_chars:
            chunks.append(current.strip())
            current = sentence
            continue
        current = candidate
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _infer_value(text: str, mapping: dict[str, tuple[str, ...]]) -> str | None:
    """Infer one metadata value from chunk text using a keyword mapping."""

    lowered = text.lower()
    for value, keywords in mapping.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            return value
    return None


def _infer_priority(text: str) -> str:
    """Infer rule priority from directive wording."""

    lowered = text.lower()
    if any(token in lowered for token in ("must", "必须", "avoid", "禁止", "不要")):
        return "must"
    if any(token in lowered for token in ("should", "建议", "prefer", "优先")):
        return "should"
    return "nice_to_have"


def _infer_chunk_metadata(
    source_name: str,
    title: str,
    text: str,
    base_metadata: dict[str, str],
) -> ChunkMetadata:
    """Infer retrieval metadata from file name, title, and chunk content."""

    combined = " ".join(item for item in (source_name, title, text) if item)
    knowledge_type = str(base_metadata.get("knowledge_type") or "strategy_rule")
    strategy_type = str(_infer_value(combined, STRATEGY_HINTS) or base_metadata.get("strategy_type") or knowledge_type)
    platform = normalize_platform(str(base_metadata.get("platform") or _infer_value(combined, PLATFORM_KEYWORDS) or "") or None)
    language = normalize_language(str(base_metadata.get("language") or _infer_value(combined, LANGUAGE_KEYWORDS) or "") or None)
    style = normalize_style(str(base_metadata.get("style") or _infer_value(combined, STYLE_KEYWORDS) or "") or None)
    duration_band = str(base_metadata.get("duration_band") or _infer_value(combined, DURATION_HINTS) or "") or None
    tags = sorted(
        {
            tag
            for tag in (
                knowledge_type,
                strategy_type,
                platform,
                language,
                style,
                duration_band,
            )
            if tag
        }
    )
    return {
        "knowledge_type": knowledge_type,
        "strategy_type": strategy_type,
        "platform": platform,
        "language": language,
        "style": style,
        "duration_band": duration_band,
        "priority": _infer_priority(combined),
        "tags": tags,
    }


def _base_metadata_for_source(path: Path) -> dict[str, str]:
    """Infer coarse metadata from the knowledge file name."""

    stem = path.stem.lower()
    if "platform" in stem:
        return {"knowledge_type": "platform_rule", "strategy_type": "platform_rule"}
    if "subtitle" in stem:
        return {"knowledge_type": "subtitle_rule", "strategy_type": "subtitle_rule"}
    if "editing" in stem:
        return {"knowledge_type": "pacing_template", "strategy_type": "pacing_rule"}
    if "title" in stem:
        return {"knowledge_type": "title_template", "strategy_type": "hook_rule"}
    return {"knowledge_type": "strategy_rule", "strategy_type": "strategy_rule"}


def chunk_markdown_document(source_path: Path, content: str) -> list[KnowledgeChunk]:
    """Split one markdown strategy document into retrieval-friendly chunks."""

    source_name = source_path.name
    base_metadata = _base_metadata_for_source(source_path)
    heading_stack: list[str] = []
    blocks: list[tuple[str, str]] = []
    current_lines: list[str] = []

    def flush_current() -> None:
        text = _compact("\n".join(current_lines))
        if text:
            title = " > ".join(heading_stack) if heading_stack else source_path.stem.replace("_", " ").title()
            blocks.append((title, text))
        current_lines.clear()

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            flush_current()
            continue
        if line.startswith("#"):
            flush_current()
            level = len(line) - len(line.lstrip("#"))
            heading_text = line[level:].strip() or source_path.stem
            heading_stack[:] = heading_stack[: max(0, level - 1)]
            heading_stack.append(heading_text)
            continue
        if line.startswith(("-", "*")):
            current_lines.append(line[1:].strip())
        else:
            current_lines.append(line)
    flush_current()

    chunks: list[KnowledgeChunk] = []
    counter = 0
    for title, text in blocks:
        chunk_parts = _split_long_block(text) if len(text) > MAX_CHUNK_CHARS else [text]
        for part in chunk_parts:
            compact_part = _compact(part)
            if len(compact_part) < MIN_CHUNK_CHARS and chunks and chunks[-1].title == title:
                previous = chunks[-1]
                previous.text = f"{previous.text} {compact_part}".strip()
                merged_metadata = _infer_chunk_metadata(source_name, title, compact_part, base_metadata)
                merged_tags = merged_metadata["tags"]
                previous.tags = sorted(set(previous.tags).union(set(merged_tags)))
                continue

            counter += 1
            metadata = _infer_chunk_metadata(source_name, title, compact_part, base_metadata)

            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{_slugify(source_path.stem)}_{counter:03d}",
                    text=compact_part,
                    title=title,
                    source_file=str(source_path).replace("\\", "/"),
                    knowledge_type=metadata["knowledge_type"],
                    strategy_type=metadata["strategy_type"],
                    platform=metadata["platform"],
                    language=metadata["language"],
                    style=metadata["style"],
                    duration_band=metadata["duration_band"],
                    priority=metadata["priority"],
                    tags=metadata["tags"],
                )
            )
    return chunks
