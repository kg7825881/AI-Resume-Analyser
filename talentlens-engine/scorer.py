"""
scorer.py — scoring engine implementing the simplified methodology:
  - Mandatory Skills (50%): exact match only, hard gate if >0 missing
  - Relevant Experience (30%): purely checks total_years >= JD min_years
  - Education (10%): degree match against JD education_requirements
  - Preferred Skills (10%): semantic/exact match combining JD's technical and soft preferred lists
"""

from datetime import datetime, timezone
from matcher import score_skill_list, get_embedding, evidence_status

WEIGHTS = {
    "mandatory_skills": 0.50,
    "relevant_experience": 0.30,
    "education": 0.10,
    "preferred_skills": 0.10,
}

HARD_GATE_MAX_MISSING_MANDATORY = 0  # Fail the gate if ANY mandatory exact-matches are missing


def _score_experience(candidate_data: dict, jd_data: dict) -> tuple[float, str, dict]:
    """
    Only evaluates if min_years_experience is in the JD. 
    Matches purely on total_years_experience >= min_years_required.
    """
    min_years = jd_data.get("min_years_experience")
    
    # If not mentioned in JD, give full credit to avoid penalizing the candidate
    if min_years is None:
        return WEIGHTS["relevant_experience"] * 100, "No minimum experience specified in JD", {"years": None, "roles": []}

    total_years = candidate_data.get("total_years_experience", 0) or 0
    years_fraction = 1.0 if total_years >= min_years else 0.0
    
    score = years_fraction * WEIGHTS["relevant_experience"] * 100
    notes = f"{total_years} yrs total vs {min_years} yrs required"

    years_evidence = {
        "total_years_experience": total_years,
        "min_years_required": min_years,
        "status": "matched" if years_fraction == 1.0 else "missing",
    }

    return round(score, 2), notes, {"roles": [], "years": years_evidence}


def _score_education(candidate_data: dict, jd_data: dict) -> tuple[float, str, list]:
    """
    Degree match against JD education_requirements.
    Only considers education if it is explicitly mentioned in the JD.
    """
    requirements = jd_data.get("education_requirements", [])
    candidate_edu = candidate_data.get("education", [])
    
    if not requirements:
        return WEIGHTS["education"] * 100, "No specific education requirement in JD", []

    candidate_fields = [_norm(e.get("field", "")) for e in candidate_edu]
    candidate_levels = [_norm(e.get("degree_level", "")) for e in candidate_edu]

    education_evidence = []
    matched_required = 0
    total_required = 0
    
    for req in requirements:
        if not req.get("required", False):
            continue
        total_required += 1
        req_field_norm = _norm(req.get("field", ""))
        field_match = req_field_norm in candidate_fields or any(
            req_field_norm in cf for cf in candidate_fields
        )
        level_match = _norm(req.get("degree_level", "")) in candidate_levels
        matched = field_match and level_match
        if matched:
            matched_required += 1

        education_evidence.append({
            "required_field": req.get("field", ""),
            "required_degree_level": req.get("degree_level", ""),
            "status": "matched" if matched else "missing",
        })

    if total_required == 0:
        return WEIGHTS["education"] * 100, "No hard-required education entries", []

    fraction = matched_required / total_required
    score = fraction * WEIGHTS["education"] * 100
    notes = f"{matched_required}/{total_required} required education criteria met"
    return round(score, 2), notes, education_evidence


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _build_evidence(result: dict) -> list[dict]:
    """
    Converts one score_skill_list result into the per-skill rows for the UI.
    """
    rows = []
    for r in result["results"]:
        rows.append({
            "skill": r["skill"],
            "status": evidence_status(r),
            "match_type": r["match_type"],
            "matched_against": r.get("matched_against"),
            "contribution": r["contribution"],
        })
    return rows


def _extra_candidate_skills(candidate_skills: list[str], *results: dict) -> list[str]:
    """
    Extracts candidate skills not consumed by any JD requirement.
    """
    used = set()
    for result in results:
        for r in result["results"]:
            if r.get("matched_against"):
                used.add(r["matched_against"].strip().lower())
    return [s for s in candidate_skills if s.strip().lower() not in used]


def calculate_job_fit(candidate_data: dict, jd_data: dict, embed_fn=get_embedding) -> dict:
    """
    Main scoring entry point using exact-match gating for mandatory skills.
    """
    candidate_skills = candidate_data.get("skills_all_sources") or candidate_data.get("skills", [])

    # --- Mandatory Skills (EXACT MATCH ONLY) ---
    mandatory_result = score_skill_list(jd_data.get("mandatory_skills", []), candidate_skills, embed_fn, exact_only=True)
    mandatory_score = mandatory_result["average_contribution"] * WEIGHTS["mandatory_skills"] * 100
    
    gate_missing_mandatory = mandatory_result["gate_missing"]
    hard_gate_failed = len(gate_missing_mandatory) > HARD_GATE_MAX_MISSING_MANDATORY
    hard_gate_reason = (
        f"Missing {len(gate_missing_mandatory)} mandatory exact-match skills: {', '.join(gate_missing_mandatory)}"
        if hard_gate_failed else ""
    )

    # --- Preferred Skills (Combined technical + soft) ---
    jd_preferred_skills = jd_data.get("preferred_technical_skills", []) + jd_data.get("soft_preferred_skills", [])
    pref_result = score_skill_list(jd_preferred_skills, candidate_skills, embed_fn, exact_only=False)
    pref_score = pref_result["average_contribution"] * WEIGHTS["preferred_skills"] * 100

    # --- Relevant Experience ---
    experience_score, experience_notes, experience_evidence = _score_experience(candidate_data, jd_data)

    # --- Education ---
    education_score, education_notes, education_evidence = _score_education(candidate_data, jd_data)

    category_scores = {
        "mandatory_skills": {
            "score": round(mandatory_score, 2),
            "matched": mandatory_result["matched"],
            "missing": mandatory_result["missing"],
            "gate_missing": gate_missing_mandatory,
        },
        "relevant_experience": {"score": experience_score, "notes": experience_notes},
        "preferred_skills": {
            "score": round(pref_score, 2), 
            "matched": pref_result["matched"], 
            "missing": pref_result["missing"],
        },
        "education": {"score": education_score, "notes": education_notes},
    }

    final_score = round(sum(c["score"] for c in category_scores.values()), 2)

    evidence = {
        "mandatory_skills": _build_evidence(mandatory_result),
        "preferred_skills": _build_evidence(pref_result),
        "experience": experience_evidence, 
        "education": education_evidence,
        "additional_candidate_skills": _extra_candidate_skills(candidate_skills, mandatory_result, pref_result),
    }

    return {
        "category_scores": category_scores,
        "evidence": evidence,
        "final_score": final_score,
        "hard_gate_failed": hard_gate_failed,
        "hard_gate_reason": hard_gate_reason,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }