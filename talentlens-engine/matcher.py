"""
matcher.py — embedding + similarity utilities.

Provides per-skill matching (exact match = 1.0, semantic band = 0.6-0.9,
no match = 0.0) as specified in the scoring methodology, instead of the
previous whole-JD-blob comparison.
"""

import math
import ollama

# Cosine similarity thresholds for the semantic match band.
# Below SEM_LOW -> no match (0.0). Above SEM_HIGH -> top of band (0.9).
# Between the two -> linearly interpolated between 0.6 and 0.9.
# Threshold band for short SKILL-TO-SKILL matching (e.g. "Docker" vs "Docker", or vs another
# short skill phrase). Single terms compared to single terms sit in a higher cosine range.
SEM_LOW = 0.55
SEM_HIGH = 0.80

# Threshold band for FREE-TEXT comparisons (a short JD phrase/role description vs a full
# paragraph-length project or experience description). Comparing short text to long text
# naturally produces lower cosine similarity regardless of actual relevance, so this band
# is intentionally lower — reusing the bounds already empirically found to work for this
# embedding model in the original calibrate_score() implementation.
TEXT_SIM_LOW = 0.35
TEXT_SIM_HIGH = 0.65


def get_embedding(text: str) -> list[float]:
    """Generates a vector embedding for a string using the local Qwen embedding model."""
    response = ollama.embeddings(model='qwen3-embedding:8b', prompt=text)
    return response['embedding']


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
    Returns {"skill": ..., "contribution": 0.0-1.0, "match_type": "exact"|"semantic"|"none"}
    per the methodology: exact match = 1.0, semantic band = 0.6-0.9, no match = 0.0.
    """
    required_norm = _normalize(required_skill)

    # 1. Exact match check first (cheap, no embedding call needed)
    for cand in candidate_skills:
        if _normalize(cand) == required_norm:
            return {"skill": required_skill, "contribution": 1.0, "match_type": "exact"}

    # Substring match also counts as exact for compound skill phrases
    # (e.g. required "Python" vs candidate "Python (Advanced)")
    for cand in candidate_skills:
        cand_norm = _normalize(cand)
        if required_norm in cand_norm or cand_norm in required_norm:
            return {"skill": required_skill, "contribution": 1.0, "match_type": "exact"}

    if not candidate_skills:
        return {"skill": required_skill, "contribution": 0.0, "match_type": "none"}

    # 2. No exact match -> semantic similarity against each candidate skill, take the best
    required_vec = embed_fn(required_skill)
    best_cosine = 0.0
    for cand in candidate_skills:
        cand_vec = embed_fn(cand)
        sim = cosine_similarity(required_vec, cand_vec)
        if sim > best_cosine:
            best_cosine = sim

    if best_cosine < SEM_LOW:
        return {"skill": required_skill, "contribution": 0.0, "match_type": "none"}
    if best_cosine >= SEM_HIGH:
        return {"skill": required_skill, "contribution": 0.9, "match_type": "semantic"}

    # Linear interpolation between 0.6 and 0.9 across the SEM_LOW-SEM_HIGH band
    fraction = (best_cosine - SEM_LOW) / (SEM_HIGH - SEM_LOW)
    contribution = round(0.6 + fraction * 0.3, 3)
    return {"skill": required_skill, "contribution": contribution, "match_type": "semantic"}


def score_skill_list(required_skills: list[str], candidate_skills: list[str], embed_fn=get_embedding) -> dict:
    """
    Scores an entire list of required skills (e.g. mandatory_skills, preferred_technical_skills)
    against a candidate's skill list. Returns per-skill results plus matched/missing summaries,
    which the scoring engine uses both for the category score and for hard-gate / explainability.
    """
    if not required_skills:
        return {"results": [], "matched": [], "missing": [], "average_contribution": 1.0}

    results = [score_single_skill(skill, candidate_skills, embed_fn) for skill in required_skills]
    matched = [r["skill"] for r in results if r["contribution"] > 0.0]
    missing = [r["skill"] for r in results if r["contribution"] == 0.0]
    avg = sum(r["contribution"] for r in results) / len(results)

    return {"results": results, "matched": matched, "missing": missing, "average_contribution": avg}


def semantic_similarity_score(text_a: str, text_b: str, embed_fn=get_embedding) -> float:
    """Returns raw cosine similarity between two free-text blocks (used for projects/experience relevance)."""
    if not text_a or not text_b:
        return 0.0
    return cosine_similarity(embed_fn(text_a), embed_fn(text_b))