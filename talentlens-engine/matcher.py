"""
matcher.py — embedding + similarity utilities.

Provides per-skill matching (exact match = 1.0, semantic band = 0.6-0.9,
no match = 0.0) as specified in the scoring methodology. Now supports an 
`exact_only` flag to strictly gate mandatory requirements while still 
identifying semantic near-misses for UI display.
"""

import math
import re
import threading
import ollama

EMBED_MODEL = "nomic-embed-text"

# Cosine similarity thresholds for the semantic match band.
# Below SEM_LOW -> no match (0.0). Above SEM_HIGH -> top of band (0.9).
# Between the two -> linearly interpolated between 0.6 and 0.9.
SEM_LOW = 0.46
SEM_HIGH = 0.70

# Minimum contribution a match must reach to satisfy a hard gate when 
# exact_only is False.
GATE_MIN_CONTRIBUTION = 0.8

# --- Embedding cache ---
_embedding_cache: dict[str, list[float]] = {}
_cache_lock = threading.Lock()


def get_embedding(text: str) -> list[float]:
    """Generates (or returns a cached) vector embedding for a string via the local embedding model."""
    with _cache_lock:
        cached = _embedding_cache.get(text)
    if cached is not None:
        return cached

    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    vec = response["embedding"]

    with _cache_lock:
        _embedding_cache[text] = vec
    return vec


def cache_stats() -> dict:
    """Useful for confirming the cache is actually being hit during a batch run."""
    with _cache_lock:
        return {"cached_strings": len(_embedding_cache)}


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)


def _normalize(text: str) -> str:
    return text.strip().lower()


def _word_boundary_contains(needle: str, haystack: str) -> bool:
    """
    True if `needle` appears in `haystack` as a whole token/phrase, not just as a
    raw substring. Prevents false positives like "java" matching inside
    "javascript", while still allowing compound-phrase matches like "aws"
    matching "aws lambda" or "react" matching "react native".

    Boundary is defined as "not immediately adjacent to another letter/digit",
    so punctuation (spaces, +, #, /, etc.) counts as a valid edge. Note this
    means single/short tokens glued to symbols (e.g. required "c" against
    candidate "c++") can still match - that ambiguity is inherent to
    substring-based compound matching and isn't fully resolved by word
    boundaries alone.
    """
    if not needle:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])"
    return re.search(pattern, haystack) is not None


def score_single_skill(required_skill: str, candidate_skills: list[str], embed_fn=get_embedding, exact_only: bool = False) -> dict:
    """
    Scores one required skill against a candidate's skill list.
    If exact_only is True, semantic matches are recorded for UI display but contribute 0.0.
    """
    required_norm = _normalize(required_skill)

    # 1. Exact match check first (cheap, no embedding call needed)
    for cand in candidate_skills:
        if _normalize(cand) == required_norm:
            return {"skill": required_skill, "contribution": 1.0, "match_type": "exact",
                    "gate_satisfied": True, "matched_against": cand}

    # Word-boundary phrase match also counts as exact for compound skill phrases
    # (e.g. required "AWS" satisfied by candidate "AWS Lambda"), but does NOT
    # allow bare substrings like required "java" being satisfied by "javascript".
    for cand in candidate_skills:
        cand_norm = _normalize(cand)
        if _word_boundary_contains(required_norm, cand_norm) or _word_boundary_contains(cand_norm, required_norm):
            return {"skill": required_skill, "contribution": 1.0, "match_type": "exact",
                    "gate_satisfied": True, "matched_against": cand}

    if not candidate_skills:
        return {"skill": required_skill, "contribution": 0.0, "match_type": "none",
                "gate_satisfied": False, "matched_against": None}

    # 2. No exact match -> semantic similarity against each candidate skill, take the best
    required_vec = embed_fn(required_skill)
    best_cosine = 0.0
    best_candidate = None
    for cand in candidate_skills:
        cand_vec = embed_fn(cand)
        sim = cosine_similarity(required_vec, cand_vec)
        if sim > best_cosine:
            best_cosine = sim
            best_candidate = cand

    if best_cosine < SEM_LOW:
        return {"skill": required_skill, "contribution": 0.0, "match_type": "none",
                "gate_satisfied": False, "matched_against": None}
                
    if exact_only:
        # We found a semantic match, but we only accept exact matches.
        # Zero contribution ensures it fails the gate and renders red.
        return {
            "skill": required_skill,
            "contribution": 0.0,
            "match_type": "semantic",
            "gate_satisfied": False,
            "matched_against": best_candidate,
        }

    if best_cosine >= SEM_HIGH:
        contribution = 0.9
    else:
        # Linear interpolation between 0.6 and 0.9 across the SEM_LOW-SEM_HIGH band
        fraction = (best_cosine - SEM_LOW) / (SEM_HIGH - SEM_LOW)
        contribution = round(0.6 + fraction * 0.3, 3)

    return {
        "skill": required_skill,
        "contribution": contribution,
        "match_type": "semantic",
        "gate_satisfied": contribution >= GATE_MIN_CONTRIBUTION,
        "matched_against": best_candidate,
    }


def score_skill_list(required_skills: list[str], candidate_skills: list[str], embed_fn=get_embedding, exact_only: bool = False) -> dict:
    """
    Scores an entire list of required skills against a candidate's skill list. 
    """
    if not required_skills:
        return {"results": [], "matched": [], "missing": [], "gate_missing": [], "average_contribution": 1.0}

    results = [score_single_skill(skill, candidate_skills, embed_fn, exact_only) for skill in required_skills]
    matched = [r["skill"] for r in results if r["contribution"] > 0.0]
    missing = [r["skill"] for r in results if r["contribution"] == 0.0]
    gate_missing = [r["skill"] for r in results if not r["gate_satisfied"]]
    avg = sum(r["contribution"] for r in results) / len(results)

    return {"results": results, "matched": matched, "missing": missing, "gate_missing": gate_missing, "average_contribution": avg}


def evidence_status(result: dict) -> str:
    """
    Maps one score_single_skill result to a 3-state UI status for the evidence view.
    Zero contribution (including near-miss semantic matches under exact_only) maps to "missing" (red).
    """
    if result["contribution"] == 0.0:
        return "missing"
    if not result["gate_satisfied"]:
        return "weak_match"
    return "matched"