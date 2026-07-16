from __future__ import annotations


def _normalize_token(value: str | None) -> str:
    """Normalize one free-form metadata token into a compact comparison key."""

    if value is None:
        return ""
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def normalize_platform(value: str | None) -> str | None:
    """Map platform aliases onto the canonical values stored in the knowledge base."""

    normalized = _normalize_token(value)
    if not normalized:
        return None

    alias_map = {
        "bilibili": "bilibili",
        "bili": "bilibili",
        "douyin": "douyin",
        "xiaohongshu": "xiaohongshu",
        "xhs": "xiaohongshu",
        "rednote": "xiaohongshu",
    }
    return alias_map.get(normalized, normalized)


def normalize_language(value: str | None) -> str | None:
    """Map language aliases onto the canonical values stored in the knowledge base."""

    normalized = _normalize_token(value)
    if not normalized:
        return None

    alias_map = {
        "zh": "zh",
        "zh_cn": "zh",
        "zh_hans": "zh",
        "cn": "zh",
        "chinese": "zh",
        "mandarin": "zh",
        "en": "en",
        "en_us": "en",
        "en_gb": "en",
        "english": "en",
    }
    return alias_map.get(normalized, normalized)


def normalize_style(value: str | None) -> str | None:
    """Map style aliases onto the canonical values stored in the knowledge base."""

    normalized = _normalize_token(value)
    if not normalized:
        return None

    alias_map = {
        "powerful": "powerful",
        "strong": "powerful",
        "impactful": "powerful",
        "punchy": "powerful",
        "intense": "powerful",
        "healing": "healing",
        "gentle": "healing",
        "soft": "healing",
        "calm": "healing",
        "soothing": "healing",
        "educational": "educational",
        "education": "educational",
        "knowledge": "educational",
        "explainer": "educational",
        "tutorial": "educational",
    }
    return alias_map.get(normalized, normalized)
