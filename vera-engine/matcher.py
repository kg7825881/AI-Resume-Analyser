"""
matcher.py — skill/requirement matching for TalentLens.

Two-stage matching per requirement:
  1. Exact / word-boundary match against the candidate's own skill list
     (cheap, deterministic, no model calls at all).
  2. If no exact match: BM25 retrieves the candidate's most relevant resume
     evidence for the requirement (retrieval.py), then a small local LLM
     classifies how well that evidence actually supports the requirement
     (judge.py). The LLM is never asked for a score — only a classification
     ("direct" / "related" / "weak" / "none") — and that classification is
     converted into a numeric contribution deterministically, right here.

exact_only is a generic strictness flag, not tied to any one category. As of
this revision, scorer.py calls score_skill_list with exact_only=False for
BOTH mandatory and preferred skills — an exact match still earns free full
credit with no model call either way, but a non-exact match is now scored via
BM25 + judge evidence for both, rather than mandatory skills being forced to
zero out. The flag remains available for a caller that wants literal-match-
only behavior for some other requirement category in the future.

This replaces the previous embedding-cosine-similarity semantic stage.
Retrieval + judgment is more inspectable than a bare cosine number: every
non-exact match now carries the actual evidence quote it was judged against
and a one-sentence reason, which is what actually gets shown to HR. It also
removes an entire class of problems that came from embedding-model-specific
similarity thresholds (SEM_LOW/SEM_HIGH) needing separate recalibration
every time the embedding model changed — see calibrate_embeddings.py, which
existed only because of that.

Embeddings aren't gone from the codebase, just no longer part of the default
scoring path — see embeddings.py.
"""

import re

from judge import judge_evidence
from retrieval import CandidateEvidenceIndex

# Minimum contribution a match must reach to satisfy a hard gate. Only
# "direct" evidence (1.0) clears this bar — "related" (0.7) is deliberately
# kept below it: a plausible-but-not-explicit match earns real partial score
# toward a category, but shouldn't by itself silently satisfy a hard
# mandatory-skill gate. Same intent as the old SEM_HIGH embedding band.
GATE_MIN_CONTRIBUTION = 0.8

# How many BM25-retrieved evidence chunks to hand the judge per requirement.
# Kept small: the judge only needs the strongest evidence, not the whole
# resume, and every extra chunk is more tokens per call across what can be
# dozens of requirements x candidates in one /analyze run.
DEFAULT_TOP_K = 3

# Deterministic mapping from the judge's 4-level classification to a numeric
# contribution. Tuned here, not in judge.py — the judge only classifies, it
# never sees or produces these numbers (same "LLM never does arithmetic"
# principle as experience.py / extractor.py). Shared by mandatory and
# preferred skills alike; split this into two mappings if mandatory and
# preferred should ever weight evidence confidence differently.
MATCH_LEVEL_CONTRIBUTION = {
    "direct": 1.0,
    "related": 0.7,
    "weak": 0.35,
    "none": 0.0,
}

# Interchangeable skill acronyms/terms that should satisfy exact matching bidirectionally
EQUIVALENT_SKILLS = {
    "etl": {"etl", "elt"},
    "elt": {"etl", "elt"},
}


def _normalize(text: str) -> str:
    return text.strip().lower()


def _word_boundary_contains(needle: str, haystack: str) -> bool:
    """
    True if `needle` appears in `haystack` as a whole token/phrase, not just
    as a raw substring. Prevents false positives like "java" matching inside
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


def _exact_match(required_skill: str, candidate_skills: list[str]) -> str | None:
    """Returns the candidate skill string that satisfies an exact/word-boundary
    match, or None if there isn't one."""
    required_norm = _normalize(required_skill)

    # 1. Direct equality & interchangeable term checks (e.g., ETL <-> ELT)
    for cand in candidate_skills:
        cand_norm = _normalize(cand)
        if cand_norm == required_norm:
            return cand
        if required_norm in EQUIVALENT_SKILLS and cand_norm in EQUIVALENT_SKILLS[required_norm]:
            return cand

    # 2. Word-boundary containment checks
    req_variants = EQUIVALENT_SKILLS.get(required_norm, {required_norm})

    for cand in candidate_skills:
        cand_norm = _normalize(cand)

        # Direction 1: Required term (or its equivalent variant) appears inside candidate skill
        for req_var in req_variants:
            if _word_boundary_contains(req_var, cand_norm):
                return cand

        # Direction 2: Multi-token candidate skill appears inside required phrase
        if len(cand_norm.split()) > 1 and _word_boundary_contains(cand_norm, required_norm):
            return cand

    return None


def score_single_skill(
    required_skill: str,
    candidate_skills: list[str],
    evidence_index: CandidateEvidenceIndex,
    judge_fn=judge_evidence,
    exact_only: bool = False,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    Scores one required skill/requirement against a candidate.

    evidence_index: a CandidateEvidenceIndex built ONCE per candidate (see
    retrieval.py / scorer.py) and reused across every requirement scored
    against that candidate — the BM25 index doesn't need rebuilding per-skill.

    If exact_only is True, evidence-based matches are still retrieved and
    judged — so the UI can show *why* something is a near-miss — but
    contribute 0.0 either way: only a literal exact/word-boundary match is
    accepted. As of this revision neither mandatory nor preferred skills call
    this with exact_only=True by default; it's kept available for any future
    requirement category that should behave that strictly.
    """
    matched_against = _exact_match(required_skill, candidate_skills)
    if matched_against is not None:
        return {
            "skill": required_skill,
            "contribution": 1.0,
            "match_type": "exact",
            "gate_satisfied": True,
            "matched_against": matched_against,
            "evidence": [],
            "judge_reason": "",
        }

    evidence_chunks = evidence_index.retrieve(required_skill, top_k=top_k)
    judgment = judge_fn(required_skill, evidence_chunks)
    match_level = judgment["match"]
    base_contribution = MATCH_LEVEL_CONTRIBUTION.get(match_level, 0.0)
    found_evidence = base_contribution > 0.0
    evidence_label = evidence_chunks[0]["source_label"] if (found_evidence and evidence_chunks) else None

    if exact_only:
        # Evidence was found and judged (useful for the UI's "near miss" view),
        # but exact_only means only a literal skill-list match is accepted —
        # zero contribution ensures it fails the gate and renders red.
        return {
            "skill": required_skill,
            "contribution": 0.0,
            "match_type": "evidence" if found_evidence else "none",
            "gate_satisfied": False,
            "matched_against": evidence_label,
            "evidence": evidence_chunks,
            "judge_reason": judgment.get("reason", ""),
        }

    return {
        "skill": required_skill,
        "contribution": base_contribution,
        "match_type": "evidence" if found_evidence else "none",
        "gate_satisfied": base_contribution >= GATE_MIN_CONTRIBUTION,
        "matched_against": evidence_label,
        "evidence": evidence_chunks,
        "judge_reason": judgment.get("reason", ""),
    }


def score_skill_list(
    required_skills: list[str],
    candidate_skills: list[str],
    evidence_index: CandidateEvidenceIndex,
    judge_fn=judge_evidence,
    exact_only: bool = False,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    Scores an entire list of required skills/requirements against a candidate's skill list.
    """
    if not required_skills:
        return {"results": [], "matched": [], "missing": [], "gate_missing": [], "average_contribution": 1.0}

    results = [
        score_single_skill(skill, candidate_skills, evidence_index, judge_fn, exact_only, top_k)
        for skill in required_skills
    ]
    matched = [r["skill"] for r in results if r["contribution"] > 0.0]
    missing = [r["skill"] for r in results if r["contribution"] == 0.0]
    gate_missing = [r["skill"] for r in results if not r["gate_satisfied"]]
    avg = sum(r["contribution"] for r in results) / len(results)

    return {"results": results, "matched": matched, "missing": missing, "gate_missing": gate_missing, "average_contribution": avg}


def evidence_status(result: dict) -> str:
    """
    Maps one score_single_skill result to a 3-state UI status for the evidence view.
    Zero contribution (including near-miss evidence under exact_only) maps to "missing" (red).
    """
    if result["contribution"] == 0.0:
        return "missing"
    if not result["gate_satisfied"]:
        return "weak_match"
    return "matched"