from __future__ import annotations

import re

from dataclasses import dataclass

from rag.ingest.normalizer import normalize_text

from rag.shared.models import CVSection


@dataclass(frozen=True)
class SectioningResult:
    """Output of splitting normalized CV text into sections."""

    candidate_name: str | None

    sections: list[CVSection]


class CVSectioner:
    """Splits normalized CV text into structured sections."""

    _SECTION_PATTERN = re.compile(
        r"(?im)^\s*(SUMMARY|PROFILE|EXPERIENCE|WORK EXPERIENCE|EDUCATION|SKILLS)\s*$"
    )

    _SECTION_NAME_MAP: dict[str, str] = {
        "PROFILE": "SUMMARY",
        "WORK EXPERIENCE": "EXPERIENCE",
    }

    _EMAIL_PATTERN = re.compile(
        r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b", re.IGNORECASE
    )

    _PHONE_PATTERN = re.compile(
        r"(?:(?:\+?\d{1,3}[\s.\-]?)?(?:\(?\d{2,4}\)?[\s.\-]?){2,5}\d{2,4}(?:\s*(?:x|ext\.?)\s*\d+)?)",
        re.IGNORECASE,
    )

    _LOCATION_PATTERN = re.compile(
        r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){0,2},\s*[A-Z][a-z]+(?: [A-Z][a-z]+){0,2})\b"
    )

    _SECTION_KEYWORD_PATTERN = re.compile(
        r"^(SUMMARY|PROFILE|EXPERIENCE|WORK EXPERIENCE|EDUCATION|SKILLS)$",
        re.IGNORECASE,
    )

    _HEADER_FALLBACK_MAX_LINES = 8

    _LOCATION_MIN_SCORE = 15.0

    _TECH_TERMS: frozenset[str] = frozenset(
        {
            "microservices",
            "kubernetes",
            "docker",
            "django",
            "git",
            "react",
            "python",
            "java",
            "javascript",
            "typescript",
            "postgresql",
            "aws",
            "redis",
            "nodejs",
            "spark",
            "pandas",
            "numpy",
            "scikit",
            "apache",
            "visualization",
            "statistics",
            "system",
            "design",
            "api",
            "rest",
            "devops",
            "go",
            "csharp",
            "azure",
            "gcp",
            "terraform",
            "ansible",
            "jenkins",
            "graphql",
            "mongodb",
            "mysql",
            "linux",
            "windows",
            "agile",
            "scrum",
        }
    )

    _NARRATIVE_TERMS: frozenset[str] = frozenset(
        {
            "developed",
            "utilizing",
            "leveraging",
            "skilled",
            "proven",
            "roles",
            "applications",
            "based",
            "using",
            "teams",
            "software",
            "engineer",
            "developer",
            "experience",
            "years",
            "mid-level",
            "senior",
            "junior",
            "scalable",
            "deliver",
            "drive",
            "business",
            "value",
            "highly",
            "motivated",
            "detail-oriented",
        }
    )

    _BLOCKED_TERMS: frozenset[str] = frozenset(
        {
            "summary",
            "profile",
            "experience",
            "education",
            "skills",
            "solution",
            "objection",
            "management",
            "analysis",
        }
    )

    _ROLE_TITLE_TERMS: frozenset[str] = frozenset(
        {
            "engineer",
            "developer",
            "architect",
            "analyst",
            "scientist",
            "manager",
            "consultant",
            "specialist",
            "lead",
            "director",
            "officer",
            "administrator",
            "intern",
            "software",
            "data",
            "machine",
            "learning",
            "stack",
            "site",
            "reliability",
            "full",
            "senior",
            "junior",
            "principal",
            "solutions",
        }
    )

    _GEOGRAPHIC_SUFFIXES: frozenset[str] = frozenset(
        {
            "islands",
            "island",
            "republic",
            "kingdom",
            "federation",
            "states",
            "territory",
            "province",
            "region",
            "county",
        }
    )

    CANONICAL_SECTION_ORDER: tuple[str, ...] = (
        "CONTACT_INFO",
        "SUMMARY",
        "EXPERIENCE",
        "EDUCATION",
        "SKILLS",
    )

    def sectionize(self, clean_text: str, doc_id: str) -> SectioningResult:
        """Split clean_text into CV sections with CONTACT_INFO first when present."""

        candidate_name = self._infer_candidate_name(clean_text)

        contact_section, text_without_contact = self._extract_contact_section(
            clean_text, doc_id
        )

        sections = self._split_sections(text_without_contact, doc_id)

        if contact_section is not None:
            sections.insert(0, contact_section)

        return SectioningResult(candidate_name=candidate_name, sections=sections)

    def _infer_candidate_name(self, clean_text: str) -> str | None:

        lines = self._extract_header_lines(clean_text)[: self._HEADER_FALLBACK_MAX_LINES]

        for line in lines:
            if self._looks_like_name(line):
                return line

        for idx in range(len(lines) - 1):
            first = lines[idx]
            second = lines[idx + 1]
            if not self._looks_like_single_name_line(first):
                continue
            if not self._looks_like_single_name_line(second):
                continue

            full_name = f"{first} {second}"
            if self._looks_like_name(full_name):
                return full_name

        return None

    def _looks_like_name(self, line: str) -> bool:

        if len(line) > 60 or any(char.isdigit() for char in line):
            return False

        if "@" in line or self._SECTION_KEYWORD_PATTERN.match(line):
            return False

        tokens = line.split()

        if not (2 <= len(tokens) <= 4):
            return False

        if any(token.lower() in self._ROLE_TITLE_TERMS for token in tokens):
            return False

        return all(re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", token) for token in tokens)

    @staticmethod
    def _looks_like_single_name_line(line: str) -> bool:

        if "@" in line or "," in line or any(char.isdigit() for char in line):
            return False

        tokens = line.split()

        if len(tokens) != 1:
            return False

        return bool(re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", tokens[0]))

    def _extract_contact_section(
        self, clean_text: str, doc_id: str
    ) -> tuple[CVSection | None, str]:

        emails = self._dedupe_preserve_order(self._EMAIL_PATTERN.findall(clean_text))

        phones = self._dedupe_preserve_order(self._PHONE_PATTERN.findall(clean_text))

        location = self._infer_location(clean_text)
        removal_fragments = self._location_removal_fragments(clean_text, location)

        contact_items: list[str] = []

        contact_items.extend(f"EMAIL: {email}" for email in emails)

        contact_items.extend(f"PHONE: {phone}" for phone in phones)

        if location:
            contact_items.append(f"LOCATION: {location}")

        text_without_contact = clean_text

        if emails:
            text_without_contact = self._EMAIL_PATTERN.sub(" ", text_without_contact)

        if phones:
            text_without_contact = self._PHONE_PATTERN.sub(" ", text_without_contact)

        if removal_fragments:
            text_without_contact = self._remove_location_lines(
                text_without_contact, removal_fragments
            )

            for fragment in removal_fragments:
                text_without_contact = re.sub(
                    re.escape(fragment), " ", text_without_contact
                )

        text_without_contact = normalize_text(text_without_contact)

        if not contact_items:
            return None, text_without_contact

        contact_text = "\n".join(contact_items)

        return (
            CVSection(doc_id=doc_id, section_name="CONTACT_INFO", text=contact_text),
            text_without_contact,
        )

    def _extract_header_lines(self, clean_text: str) -> list[str]:
        """Return non-empty lines before the first canonical section header."""

        header_lines: list[str] = []

        for line in clean_text.splitlines():
            stripped = line.strip()

            if not stripped:
                continue

            if self._SECTION_KEYWORD_PATTERN.match(stripped):
                break

            header_lines.append(stripped)

        if header_lines:
            return header_lines

        all_lines = [line.strip() for line in clean_text.splitlines() if line.strip()]

        return all_lines[: self._HEADER_FALLBACK_MAX_LINES]

    def _infer_location(self, clean_text: str) -> str | None:
        """Return at most one validated location from the CV header."""

        header_lines = self._extract_header_lines(clean_text)

        if not header_lines:
            return None

        search_lines = self._merge_split_location_lines(header_lines)

        candidates: list[tuple[str, int]] = []

        seen: set[str] = set()

        def add_candidate(raw: str, line_index: int) -> None:

            normalized = self._normalize_location(raw)

            if not normalized:
                return

            key = normalized.casefold()

            if key in seen:
                return

            seen.add(key)

            candidates.append((normalized, line_index))

        for line_index, line in enumerate(search_lines):
            for match in self._LOCATION_PATTERN.findall(line):
                add_candidate(match, line_index)

            if self._line_may_contain_location(line):
                add_candidate(line, line_index)

        if not candidates:
            return None

        scored = [
            (
                self._score_location_candidate(
                    candidate, line_index, len(search_lines)
                ),
                line_index,
                candidate,
            )
            for candidate, line_index in candidates
        ]

        valid = [
            (score, idx, cand)
            for score, idx, cand in scored
            if score >= self._LOCATION_MIN_SCORE
        ]

        if not valid:
            return None

        _score, _idx, best = max(valid, key=lambda item: (item[0], -item[1]))

        return best

    def _location_removal_fragments(
        self, clean_text: str, location: str | None
    ) -> list[str]:
        """Return text fragments to strip from the body (including split header lines)."""
        if not location:
            return []

        fragments = [location]
        if "," not in location:
            return fragments

        city, region = (part.strip() for part in location.split(",", 1))
        header_lines = self._extract_header_lines(clean_text)
        for idx, line in enumerate(header_lines):
            if line.rstrip().rstrip(",").strip() != city:
                continue
            if not line.rstrip().endswith(","):
                continue
            if idx + 1 >= len(header_lines):
                continue
            if header_lines[idx + 1].strip() != region:
                continue
            fragments.append(line)
            fragments.append(header_lines[idx + 1])
            break

        return self._dedupe_preserve_order(fragments)

    @staticmethod
    def _merge_split_location_lines(header_lines: list[str]) -> list[str]:
        """Join city line ending with comma and region on the following line."""
        merged: list[str] = []
        idx = 0
        while idx < len(header_lines):
            line = header_lines[idx]
            if idx + 1 < len(header_lines) and CVSectioner._should_merge_location_pair(
                line, header_lines[idx + 1]
            ):
                city = line.rstrip().rstrip(",").strip()
                region = header_lines[idx + 1].strip()
                merged.append(f"{city}, {region}")
                idx += 2
                continue
            merged.append(line)
            idx += 1
        return merged

    @staticmethod
    def _should_merge_location_pair(first: str, second: str) -> bool:
        if not first.rstrip().endswith(","):
            return False

        second = second.strip()
        if not second or "@" in second or any(char.isdigit() for char in second):
            return False
        if len(second) > 40:
            return False

        words = second.split()
        if not 1 <= len(words) <= 3:
            return False
        if not all(re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", word) for word in words):
            return False

        city = first.rstrip().rstrip(",").strip()
        if not city or len(city) > 50 or "@" in city or not city[0].isupper():
            return False

        return True

    def _line_may_contain_location(self, line: str) -> bool:

        if "," not in line or "@" in line or any(char.isdigit() for char in line):
            return False

        if self._SECTION_KEYWORD_PATTERN.match(line):
            return False

        if len(line) > 60:
            return False

        lower = line.lower()

        return not lower.startswith("email:") and not lower.startswith("phone:")

    def _score_location_candidate(
        self, candidate: str, line_index: int, header_len: int
    ) -> float:

        if self._is_rejected_location(candidate):
            return -1000.0

        score = max(0.0, 25.0 - line_index * 5.0)

        if self._LOCATION_PATTERN.search(candidate):
            score += 20.0

        parts = [part.strip() for part in candidate.split(",", 1)]

        if len(parts) == 2:
            _city, region = parts

            region_words = region.split()

            if len(region_words) >= 2:
                score += 15.0

            if any(word.lower() in self._GEOGRAPHIC_SUFFIXES for word in region_words):
                score += 10.0

            if len(region) >= 5:
                score += 5.0

        if len(candidate) <= 45:
            score += 5.0

        if line_index <= 2:
            score += 5.0

        if header_len <= 6 and line_index == header_len - 1:
            score -= 5.0

        return score

    def _is_rejected_location(self, candidate: str) -> bool:

        text = candidate.strip()

        if not text or len(text) > 60:
            return True

        if text.count(",") != 1:
            return True

        if "@" in text or any(char.isdigit() for char in text):
            return True

        if self._SECTION_KEYWORD_PATTERN.match(text):
            return True

        lower = text.lower()

        if " and " in lower:
            return True

        words = {token.lower().strip(".,&") for token in text.replace(",", " ").split()}

        if words & self._BLOCKED_TERMS:
            return True

        if words & self._TECH_TERMS:
            return True

        for term in self._NARRATIVE_TERMS:
            if term in lower:
                return True

        if len(text.split()) > 8:
            return True

        city, region = (part.strip() for part in text.split(",", 1))

        if not city or not region:
            return True

        if self._looks_like_skill_pair(city, region):
            return True

        return False

    @staticmethod
    def _looks_like_skill_pair(city: str, region: str) -> bool:

        city_words = city.split()

        region_words = region.split()

        if len(city_words) == 1 and len(region_words) == 1:
            single_word_tech = {
                "go",
                "git",
                "java",
                "python",
                "react",
                "django",
                "docker",
                "aws",
                "rust",
                "php",
                "sql",
                "c",
                "r",
            }

            if city.lower() in single_word_tech or region.lower() in single_word_tech:
                return True

        tech_hints = {
            "microservices",
            "kubernetes",
            "system",
            "design",
            "data",
            "visualization",
            "pandas",
            "apache",
            "spark",
            "numpy",
            "scikit",
            "learn",
            "statistics",
        }

        combined = {w.lower() for w in city_words + region_words}

        if combined & tech_hints:
            return True

        return False

    @staticmethod
    def _normalize_location(candidate: str) -> str:

        text = candidate.strip()

        text = re.sub(r"\s+", " ", text)

        text = text.rstrip(".,&")

        text = re.sub(r",\s*$", "", text)

        return text.strip()

    @staticmethod
    def _remove_location_lines(text: str, locations: list[str]) -> str:

        location_set = {location.strip() for location in locations}

        filtered_lines = [
            line for line in text.splitlines() if line.strip() not in location_set
        ]

        return "\n".join(filtered_lines)

    @staticmethod
    def _dedupe_preserve_order(values: list[str]) -> list[str]:

        seen: set[str] = set()

        ordered: list[str] = []

        for value in values:
            clean_value = value.strip()

            if clean_value and clean_value not in seen:
                seen.add(clean_value)

                ordered.append(clean_value)

        return ordered

    def _split_sections(self, text: str, doc_id: str) -> list[CVSection]:

        sections: list[CVSection] = []

        prepared_text = re.sub(
            r"(?<!\n)\s+(SUMMARY|PROFILE|EXPERIENCE|WORK EXPERIENCE|EDUCATION|SKILLS)\b",
            r"\n\1",
            text,
        )

        matches = list(self._SECTION_PATTERN.finditer(prepared_text))

        if not matches:
            return sections

        for idx, match in enumerate(matches):
            raw_section_name = match.group(1).upper().strip()

            section_name = self._SECTION_NAME_MAP.get(
                raw_section_name, raw_section_name
            )

            start = match.end()

            end = (
                matches[idx + 1].start()
                if idx + 1 < len(matches)
                else len(prepared_text)
            )

            section_text = prepared_text[start:end].strip()

            if not section_text:
                continue

            sections.append(
                CVSection(
                    doc_id=doc_id,
                    section_name=section_name,
                    text=section_text,
                )
            )

        return sections
