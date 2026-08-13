"""
matcher.py — embedding + similarity utilities.

Provides per-skill matching (exact match = 1.0, semantic band = 0.6-0.9,
no match = 0.0) as specified in the scoring methodology, instead of the
previous whole-JD-blob comparison.
"""

import math
import threading
import ollama

EMBED_MODEL = "nomic-embed-text"

# Cosine similarity thresholds for the semantic match band.
# Below SEM_LOW -> no match (0.0). Above SEM_HIGH -> top of band (0.9).
# Between the two -> linearly interpolated between 0.6 and 0.9.
#
# Calibrated for nomic-embed-text via calibrate_embeddings.py (2026-08-12) against real
# skill-pair and JD/resume-text-pair data. Non-match skill pairs (Docker/Kubernetes,
# Python/Adobe Photoshop) both landed ~0.42; genuine semantic match (Machine Learning/
# Deep Learning) landed at 0.699 — SEM_LOW/HIGH bracket that with margin. Free-text
# irrelevant pair landed at 0.396 vs. relevant pairs at 0.562/0.591 — note those last two
# are close together, so TEXT_SIM_LOW/HIGH mainly separates "irrelevant" from "relevant",
# not "partially" from "fully" relevant; revisit with more sample pairs if the
# projects/experience category ever looks miscalibrated in practice.
SEM_LOW = 0.46
SEM_HIGH = 0.70
TEXT_SIM_LOW = 0.43
TEXT_SIM_HIGH = 0.60

# Minimum contribution a match must reach to satisfy a hard mandatory-skill gate. A weak
# semantic hit (contribution near 0.6, i.e. cosine barely above SEM_LOW) still earns partial
# score for ranking purposes, but shouldn't be able to silently pass a hard gate the same way
# an exact match does. 0.8 back-solves to cosine ~0.62 — comfortably above the ~0.42 non-match
# noise floor from calibration, comfortably below the 0.699 genuine ML/DL match — so confident
# semantic matches still clear the gate, boundary guesses don't.
GATE_MIN_CONTRIBUTION = 0.8

# --- Embedding cache ---
# The same strings (JD skills, common candidate skills like "Python") get embedded
# repeatedly — once per candidate scored, sometimes multiple times per candidate.
# Caching by exact text avoids re-querying Ollama for identical input, which is a
# meaningful chunk of the redundant work in a batch scoring run.
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


def score_single_skill(required_skill: str, candidate_skills: list[str], embed_fn=get_embedding) -> dict:
    """
    Scores one required skill against a candidate's skill list.
    Returns {"skill": ..., "contribution": 0.0-1.0, "match_type": "exact"|"semantic"|"none",
    "gate_satisfied": bool, "matched_against": str|None} per the methodology: exact match =
    1.0, semantic band = 0.6-0.9, no match = 0.0. gate_satisfied is True for exact matches and
    for semantic matches whose contribution clears GATE_MIN_CONTRIBUTION — it's what hard
    mandatory-skill gates should check instead of "contribution > 0", so a weak semantic guess
    can't silently satisfy a gate the way a confident match does. matched_against is the
    specific candidate skill string that produced the match (None if no match) — used to build
    a human-readable evidence view ("required X, candidate has Y") and to identify which
    candidate skills weren't consumed by any requirement.
    """
    required_norm = _normalize(required_skill)

    # 1. Exact match check first (cheap, no embedding call needed)
    for cand in candidate_skills:
        if _normalize(cand) == required_norm:
            return {"skill": required_skill, "contribution": 1.0, "match_type": "exact",
                    "gate_satisfied": True, "matched_against": cand}

    # Substring match also counts as exact for compound skill phrases
    # (e.g. required "Python" vs candidate "Python (Advanced)")
    for cand in candidate_skills:
        cand_norm = _normalize(cand)
        if required_norm in cand_norm or cand_norm in required_norm:
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


def score_skill_list(required_skills: list[str], candidate_skills: list[str], embed_fn=get_embedding) -> dict:
    """
    Scores an entire list of required skills (e.g. mandatory_skills, preferred_technical_skills)
    against a candidate's skill list. Returns per-skill results plus matched/missing/gate_missing
    summaries. "missing" (contribution == 0.0) is for score-contribution and explainability
    display. "gate_missing" (gate_satisfied == False) is what a hard mandatory gate should check
    instead — it also excludes skills that only scraped a weak semantic match, not just skills
    with zero match.
    """
    if not required_skills:
        return {"results": [], "matched": [], "missing": [], "gate_missing": [], "average_contribution": 1.0}

    results = [score_single_skill(skill, candidate_skills, embed_fn) for skill in required_skills]
    matched = [r["skill"] for r in results if r["contribution"] > 0.0]
    missing = [r["skill"] for r in results if r["contribution"] == 0.0]
    gate_missing = [r["skill"] for r in results if not r["gate_satisfied"]]
    avg = sum(r["contribution"] for r in results) / len(results)

    return {"results": results, "matched": matched, "missing": missing, "gate_missing": gate_missing, "average_contribution": avg}


def semantic_similarity_score(text_a: str, text_b: str, embed_fn=get_embedding) -> float:
    """Returns raw cosine similarity between two free-text blocks (used for projects/experience relevance)."""
    if not text_a or not text_b:
        return 0.0
    return cosine_similarity(embed_fn(text_a), embed_fn(text_b))


def evidence_status(result: dict) -> str:
    """
    Maps one score_single_skill result to a 3-state UI status for the evidence view:
      "matched"     — exact or confident semantic match (green)
      "weak_match"  — nonzero contribution but below GATE_MIN_CONTRIBUTION (amber) —
                       worth a human glance, since this is exactly the band that used to
                       silently pass the hard gate before gate_satisfied was added
      "missing"     — zero contribution (red)
    """
    if result["contribution"] == 0.0:
        return "missing"
    if not result["gate_satisfied"]:
        return "weak_match"
    return "matched"