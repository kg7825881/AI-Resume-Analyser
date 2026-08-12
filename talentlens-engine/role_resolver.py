"""
role_resolver.py — resolves a free-text role query ("the AI role") against
stored JD titles, per the Phase 1 dynamic role-handling design.

Uses word-level token overlap (Jaccard similarity), not character-level
matching — character similarity (e.g. difflib.SequenceMatcher) gives
misleading scores for short phrases: "sales executive" and "AI Engineer"
share enough individual letters to look deceptively similar, even though
they share zero real words. Token overlap avoids that failure mode.
"""

import re

AMBIGUOUS_THRESHOLD = 0.20     # minimum word-overlap to even be considered a candidate
CLEAR_WINNER_GAP = 0.25        # top match must beat the runner-up by this much to auto-select

STOPWORDS = {"the", "a", "an", "for", "role", "position", "job", "opening"}


def _tokenize(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def resolve_role(query: str, roles: list) -> dict:
    """
    roles: list of {"role_id": ..., "role_title": ..., ...} from db.get_all_roles().
    Returns one of:
      {"status": "matched", "role": {...}}
      {"status": "ambiguous", "candidates": [{...}, ...]}
      {"status": "no_match"}
    """
    if not roles:
        return {"status": "no_match"}

    query_tokens = _tokenize(query)
    scored = [(role, _jaccard(query_tokens, _tokenize(role["role_title"]))) for role in roles]
    scored.sort(key=lambda x: x[1], reverse=True)

    candidates = [(role, score) for role, score in scored if score >= AMBIGUOUS_THRESHOLD]

    if not candidates:
        return {"status": "no_match"}

    if len(candidates) == 1:
        return {"status": "matched", "role": candidates[0][0]}

    top_role, top_score = candidates[0]
    second_score = candidates[1][1]

    if (top_score - second_score) >= CLEAR_WINNER_GAP:
        return {"status": "matched", "role": top_role}

    return {"status": "ambiguous", "candidates": [role for role, _ in candidates[:5]]}