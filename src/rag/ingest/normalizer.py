from __future__ import annotations

import re


_SPACED_HEADER_PATTERN = re.compile(r"\b(?:[A-Z]{1,2}\s+){2,}[A-Z]{1,2}\b")
_MULTISPACE_PATTERN = re.compile(r"[ \t]+")
_EXCESS_NEWLINES_PATTERN = re.compile(r"\n{3,}")
_BROKEN_PAGE_BREAK_PATTERN = re.compile(r"\n?\f+\n?")


def collapse_spaced_headers(text: str) -> str:
    """Collapse spaced uppercase headers (e.g. 'S U M M A R Y' -> 'SUMMARY')."""

    def _collapse(match: re.Match[str]) -> str:
        return match.group(0).replace(" ", "")

    return _SPACED_HEADER_PATTERN.sub(_collapse, text)


def normalize_visual_delimiters(text: str) -> str:
    """Normalize visual delimiters to standard punctuation."""
    normalized = text.replace("•", ", ").replace("·", ", ").replace("|", ", ")
    normalized = re.sub(r",\s*,+", ", ", normalized)
    normalized = re.sub(r"\s+,", ",", normalized)
    normalized = re.sub(r",\s{2,}", ", ", normalized)
    return normalized


def normalize_linebreaks(text: str) -> str:
    """Normalize line breaks and remove excessive empty lines."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _BROKEN_PAGE_BREAK_PATTERN.sub("\n", normalized)
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = _EXCESS_NEWLINES_PATTERN.sub("\n\n", normalized)
    return normalized.strip()


def normalize_spacing(text: str) -> str:
    """Normalize repeated spaces while preserving line breaks."""
    lines = text.split("\n")
    normalized_lines = [_MULTISPACE_PATTERN.sub(" ", line).strip() for line in lines]
    return "\n".join(normalized_lines)


def normalize_text(text: str) -> str:
    """Apply the full normalization pipeline preserving readability."""
    normalized = collapse_spaced_headers(text)
    normalized = normalize_visual_delimiters(normalized)
    normalized = normalize_linebreaks(normalized)
    normalized = normalize_spacing(normalized)
    normalized = _EXCESS_NEWLINES_PATTERN.sub("\n\n", normalized)
    return normalized.strip()
