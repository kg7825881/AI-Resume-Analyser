"""
jd_skills.py — deterministic safety net for atomic skill extraction from JD text.

jd_extractor.py's LLM prompt already instructs the model to split compound requirement
sentences into one atomic skill per array item (see the ATOMIC SKILL EXTRACTION section
of its system prompt). Confirmed in production: a 3B model doing this as one part of a
large multi-field JSON extraction doesn't reliably follow that instruction — several JD
lines came through as full unsplit sentences even with the instruction present (e.g.
"Strong SQL and good Python skills." stayed as one pill instead of splitting into SQL
and Python).

This module is a deterministic second pass, run AFTER the LLM call, that doesn't depend
on the model following the instruction. It's intentionally conservative: splitting a JD
sentence is a genuinely hard general problem (nested clauses, shared prepositional
phrases, nothing as clean as extractor.py's "Tools Used –" comma lists), so where a line
can't be split SAFELY, this keeps it as one item rather than guessing and producing
nonsense fragments. A single non-atomic item just semantic-matches a bit more weakly in
matcher.py; a fabricated fragment like "reliable datasets for ML" would never match
anything, permanently penalizing every candidate for a requirement that isn't real.
"""

import re

# Filler phrasing that precedes the actual skill(s) in a JD requirement sentence.
# Longest/most specific patterns first so a more specific match wins over a generic one.
_FILLER_PREFIXES = [
    r"hands[- ]on experience (?:with|in|building)\s+",
    r"strong understanding of\s+",
    r"strong knowledge of\s+",
    r"solid understanding of\s+",
    r"deep understanding of\s+",
    r"working knowledge of\s+",
    r"proven experience (?:with|in)\s+",
    r"demonstrated experience (?:with|in)\s+",
    r"experience working (?:with|in)\s+",
    r"experience (?:with|in|using|building|supporting)\s+",
    r"familiarity with\s+",
    r"exposure to\s+",
    r"knowledge of\s+",
    r"understanding of\s+",
    r"proficiency (?:with|in)\s+",
    r"expertise (?:with|in)\s+",
    r"comfort(?:able)? working with\s+",
    r"ability to\s+",
    r"strong\s+",
    r"good\s+",
]
_FILLER_PREFIX_RE = re.compile("^(?:" + "|".join(_FILLER_PREFIXES) + ")", re.IGNORECASE)

_YEARS_EXPERIENCE_RE = re.compile(
    r"^\s*\d+[\+\-]?\s*(?:to|-)?\s*\d*\+?\s*years?\s+of\s+experience", re.IGNORECASE
)

_EDUCATION_CLAUSE_RE = re.compile(
    r"^\s*(?:must have |requires |required: |education: |minimum |degree in )?(?:bachelor(?:'s)?|master(?:'s)?|b\.?\s*tech|m\.?\s*tech|b\.?\s*e\.?|b\.?\s*s\.?|b\.?\s*sc|m\.?\s*s\.?|m\.?\s*sc|mba|mca|bca|ph\.?d|degree|diploma)\b", 
    re.IGNORECASE
)

def _looks_like_education_clause(sentence: str) -> bool:
    return bool(_EDUCATION_CLAUSE_RE.match(sentence.strip()))

# Fragments ending in a bare role/team noun ("AI engineers", "product teams") — a
# collaborate-with-people clause, not a skill. Confirmed necessary: JDs commonly phrase
# soft requirements as "collaborate with X engineers, Y engineers, and Z teams", which
# splits grammatically cleanly but produces useless, permanently-unmatchable "skills".
_ROLE_NOUN_SUFFIX_RE = re.compile(
    r"^(?:[A-Za-z]+\s+)?(?:engineers?|developers?|scientists?|analysts?|managers?|"
    r"teams?|stakeholders?|leads?|panelists?|members?)$",
    re.IGNORECASE,
)

# A whole line phrased as "collaborate (closely/effectively) with <people/teams>" is a
# soft/people requirement, not a skill — drop the entire line rather than trying to
# split-and-filter it (splitting "collaborate closely with AI engineers, backend
# engineers, and product teams" still leaves "collaborate closely with AI engineers" as
# one fragment before a comma is even reached, which the role-noun suffix check alone
# doesn't catch).
_COLLABORATION_CLAUSE_RE = re.compile(
    r"^(?:ability to\s+|experience\s+)?collaborat\w*\s+(?:closely\s+|effectively\s+|"
    r"cross[- ]functionally\s+)?with\b",
    re.IGNORECASE,
)

# JD phrasing like "or similar tools" / "or equivalent technologies" names no real
# skill — it's a hedge word, and keeping it produces a fake, permanently-unmatchable
# "skill" (confirmed on "Spark, dbt, Airflow, Dagster, Kafka, or similar tools" ->
# 'similar tools' was coming through as its own item).
_GENERIC_HEDGE_ITEM_RE = re.compile(
    r"^(?:similar|equivalent|comparable|related|other|such)\b", re.IGNORECASE
)

# Generic trailing nouns that add nothing once a filler prefix has already been
# stripped from the same item (e.g. "good Python skills" -> after prefix-stripping
# "good " -> "Python skills"; this strips the trailing " skills" too -> "Python").
_TRAILING_GENERIC_SUFFIX_RE = re.compile(r"\s+skills?\s*$", re.IGNORECASE)

# A preposition surviving in the MIDDLE of a split fragment is the signal that the
# comma it was split on wasn't actually enumerating a list — it was part of a nested
# clause (adjectives, a shared trailing "for X, Y, Z" phrase, etc).
_MID_PREPOSITION_RE = re.compile(r"\b(for|with|in|of|using|on)\b", re.IGNORECASE)


def _strip_filler_prefix(text: str) -> str:
    return _FILLER_PREFIX_RE.sub("", text.strip()).strip()


def _looks_like_years_experience_clause(sentence: str) -> bool:
    return bool(_YEARS_EXPERIENCE_RE.match(sentence.strip()))


def _looks_like_role_or_team_name(item: str) -> bool:
    return bool(_ROLE_NOUN_SUFFIX_RE.match(item.strip()))


def _looks_like_collaboration_clause(sentence: str) -> bool:
    return bool(_COLLABORATION_CLAUSE_RE.match(sentence.strip()))


def _looks_like_generic_hedge(item: str) -> bool:
    return bool(_GENERIC_HEDGE_ITEM_RE.match(item.strip()))


def _clean_split_item(item: str) -> str:
    """Per-item cleanup applied AFTER splitting — the whole-line filler strip only
    catches filler at the very start of the original sentence, so a filler word
    attached to a LATER item in an "X and good Y" style list (confirmed: "Strong SQL
    and good Python skills." -> ['SQL', 'good Python skills'] without this) survives
    otherwise."""
    cleaned = _strip_filler_prefix(item)
    cleaned = _TRAILING_GENERIC_SUFFIX_RE.sub("", cleaned)
    return cleaned.strip(" .")


def _split_enumeration(text: str) -> list | None:
    """Splits on commas and a trailing and/or conjunction, but ONLY if (a) the text
    actually contains an and/or conjunction at all — a comma list with NO conjunction
    is almost always stacked adjectives on one noun, not an enumeration (confirmed:
    "document-heavy, workflow-heavy business data" has no and/or and was wrongly split
    into two fake "skills" before this check), and (b) every resulting fragment is
    clean — no leftover mid-fragment preposition, the signature of a shared trailing
    phrase a naive split would leave attached to only the last item. Returns None —
    meaning "do not split" — if either check fails."""
    if not re.search(r"\b(?:and|or)\b", text, re.IGNORECASE):
        return None
    normalized = re.sub(r"\s*,?\s+(?:and|or)\s+", ", ", text.strip().rstrip("."))
    parts = [p.strip() for p in normalized.split(",") if p.strip()]
    if len(parts) < 2:
        return None
    for p in parts:
        if _MID_PREPOSITION_RE.search(p):
            return None
    return parts


def atomize_skill_line(raw_line: str) -> list:
    """Deterministic atomic-skill extraction for ONE JD requirement line. Returns a
    list of atomic skill strings — length 0 (line filtered out entirely, e.g. a
    years-of-experience restatement or a collaborate-with-people clause), 1 (line
    couldn't be safely split; kept whole), or >1 (successfully split). Never raises."""
    if not raw_line or not isinstance(raw_line, str):
        return []
    line = raw_line.strip()
    if not line:
        return []

    if _looks_like_years_experience_clause(line):
        return []
    if _looks_like_education_clause(line): # NEW GUARD
        return []
    if _looks_like_collaboration_clause(line):
        return []

    stripped = _strip_filler_prefix(line).rstrip(".").strip()
    if not stripped:
        return []

    items = _split_enumeration(stripped)
    if items is None:
        items = [stripped]

    result = []
    for item in items:
        cleaned = _clean_split_item(item)
        if not cleaned:
            continue
        if _looks_like_role_or_team_name(cleaned):
            continue
        if _looks_like_generic_hedge(cleaned):
            continue
        result.append(cleaned)

    return result


def atomize_skill_list(raw_items: list) -> list:
    """Runs atomize_skill_line over a whole JD skill array (mandatory_skills,
    preferred_technical_skills, soft_preferred_skills, relevant_certifications) and
    flattens + dedups (case-insensitive) the result, preserving first-seen casing and
    order. NOT intended for responsibilities — those stay full sentences, since
    scorer.py uses them for semantic-similarity text comparison, not per-skill exact
    matching."""
    seen = set()
    out = []
    for raw in raw_items or []:
        for atom in atomize_skill_line(raw):
            key = atom.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(atom)
    return out
