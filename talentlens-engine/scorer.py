"""
scorer.py — scoring engine implementing the agreed methodology:
  - Mandatory Skills (35%): per-skill exact/semantic/none, hard gate if >1 missing
  - Relevant Experience (25%): years matched/exceeded + role-title semantic relevance
  - Technical/Preferred Skills (15%): per-skill match against JD's preferred_technical_skills
  - Projects (10%): semantic similarity of project descriptions vs JD responsibilities
  - Education (5%): degree match against JD education_requirements
  - Preferred Skills/soft (5%): per-skill match against JD's soft_preferred_skills
  - Certifications/Other (5%): per-skill match against JD's relevant_certifications

Produces a score record matching the Phase 1 schema, including hard_gate_failed
and hard_gate_reason for explainability.
"""

from datetime import datetime, timezone
from matcher import score_skill_list, semantic_similarity_score, get_embedding, evidence_status

WEIGHTS = {
    "mandatory_skills": 0.35,
    "relevant_experience": 0.25,
    "technical_preferred_skills": 0.15,
    "projects": 0.10,
    "education": 0.05,
    "preferred_skills_soft": 0.05,
    "certifications_other": 0.05,
}

HARD_GATE_MAX_MISSING_MANDATORY = 1  # fail the gate if MORE than this many are missing


def _jd_context_text(jd_data: dict) -> str:
    """
    Builds the text used to judge project/experience relevance against the JD.
    Prefers 'responsibilities' (most descriptive), but falls back to role title +
    skill lists for sparse JDs that don't list responsibilities explicitly —
    otherwise Projects/Experience-relevance would unfairly score 0 on lean JDs.
    """
    responsibilities_text = " ".join(jd_data.get("responsibilities", []))
    if responsibilities_text.strip():
        return responsibilities_text

    role_title = jd_data.get("role_title", "")
    skills = jd_data.get("mandatory_skills", []) + jd_data.get("preferred_technical_skills", [])
    return f"{role_title}. Key skills: {', '.join(skills)}." if (role_title or skills) else ""


def _score_experience(candidate_data: dict, jd_data: dict, embed_fn) -> tuple[float, str]:
    """Years matched/exceeded (50%) + role-title/domain semantic relevance (50%) of the category weight."""
    min_years = jd_data.get("min_years_experience", 0) or 0
    total_years = candidate_data.get("total_years_experience", 0) or 0

    years_fraction = 1.0 if min_years == 0 else min(total_years / min_years, 1.0)

    experience_entries = candidate_data.get("experience", [])
    jd_role_text = f"{jd_data.get('role_title', '')}. {_jd_context_text(jd_data)}"

    best_relevance = 0.0
    for entry in experience_entries:
        entry_text = f"{entry.get('title', '')} in {entry.get('domain', '')}: {entry.get('description', '')}"
        sim = semantic_similarity_score(jd_role_text, entry_text, embed_fn)
        if sim > best_relevance:
            best_relevance = sim
    # Free-text comparison (JD context vs a full experience description) -> use the
    # TEXT_SIM band, not the tighter SEM band used for short skill-to-skill matching.
    from matcher import TEXT_SIM_LOW, TEXT_SIM_HIGH
    if best_relevance < TEXT_SIM_LOW:
        relevance_fraction = 0.0
    elif best_relevance >= TEXT_SIM_HIGH:
        relevance_fraction = 1.0
    else:
        relevance_fraction = (best_relevance - TEXT_SIM_LOW) / (TEXT_SIM_HIGH - TEXT_SIM_LOW)

    category_fraction = 0.5 * years_fraction + 0.5 * relevance_fraction
    score = category_fraction * WEIGHTS["relevant_experience"] * 100
    notes = f"{total_years} yrs vs {min_years} required; best role-relevance similarity {round(best_relevance, 3)}"
    return round(score, 2), notes


def _score_projects(candidate_data: dict, jd_data: dict, embed_fn) -> tuple[float, str]:
    context_text = _jd_context_text(jd_data)
    projects = candidate_data.get("projects", [])
    if not projects or not context_text:
        return 0.0, "No projects or JD context to compare"

    from matcher import TEXT_SIM_LOW, TEXT_SIM_HIGH
    best_sim = 0.0
    for p in projects:
        sim = semantic_similarity_score(context_text, p.get("description", ""), embed_fn)
        if sim > best_sim:
            best_sim = sim
    fraction = 0.0 if best_sim < TEXT_SIM_LOW else (1.0 if best_sim >= TEXT_SIM_HIGH else (best_sim - TEXT_SIM_LOW) / (TEXT_SIM_HIGH - TEXT_SIM_LOW))
    score = fraction * WEIGHTS["projects"] * 100
    return round(score, 2), f"Best project-to-JD-context similarity {round(best_sim, 3)}"


def _score_education(candidate_data: dict, jd_data: dict) -> tuple[float, str]:
    requirements = jd_data.get("education_requirements", [])
    candidate_edu = candidate_data.get("education", [])
    if not requirements:
        return WEIGHTS["education"] * 100, "No specific education requirement in JD"

    candidate_fields = [_norm(e.get("field", "")) for e in candidate_edu]
    candidate_levels = [_norm(e.get("degree_level", "")) for e in candidate_edu]

    matched_required = 0
    total_required = 0
    for req in requirements:
        if not req.get("required", False):
            continue
        total_required += 1
        field_match = _norm(req.get("field", "")) in candidate_fields or any(
            _norm(req.get("field", "")) in cf for cf in candidate_fields
        )
        level_match = _norm(req.get("degree_level", "")) in candidate_levels
        if field_match and level_match:
            matched_required += 1

    if total_required == 0:
        return WEIGHTS["education"] * 100, "No hard-required education entries"

    fraction = matched_required / total_required
    score = fraction * WEIGHTS["education"] * 100
    return round(score, 2), f"{matched_required}/{total_required} required education criteria met"


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _build_evidence(category_key: str, result: dict) -> list[dict]:
    """
    Converts one score_skill_list result into the per-skill rows an evidence UI renders
    directly: skill name, 3-state status (matched/weak_match/missing) for
    green/amber/red styling, and — when matched — which candidate skill it matched
    against, so HR can see e.g. "required: Kubernetes -> candidate has: Docker"
    for a semantic match rather than just a bare checkmark.
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
    Candidate skills that weren't consumed as a matched_against by any requirement across
    all categories passed in. Shown as a neutral/gray 'additional skills' bucket in the
    evidence view — not required by the JD, but useful context (e.g. shows breadth, or a
    tool that might matter for a different req down the line). Case-insensitive comparison
    since matched_against preserves the candidate's original casing.
    """
    used = set()
    for result in results:
        for r in result["results"]:
            if r.get("matched_against"):
                used.add(r["matched_against"].strip().lower())
    return [s for s in candidate_skills if s.strip().lower() not in used]


def calculate_job_fit(candidate_data: dict, jd_data: dict, embed_fn=get_embedding) -> dict:
    """
    Main scoring entry point. Takes structured candidate + JD data (from extractor.py /
    jd_extractor.py) and returns a full score record per the Phase 1 schema.
    """
    # skills_all_sources (built by extractor.py's _aggregate_skills) merges the resume's own
    # Skills-section list with every technologies_used entry from experience roles and projects.
    # Matching against just "skills" misses tools a candidate demonstrably used on the job but
    # didn't also relist in a dedicated Skills section (e.g. Docker/Kubernetes/RAG mentioned only
    # under a specific role's Tools line). Falls back to "skills" for older records that predate
    # skills_all_sources.
    candidate_skills = candidate_data.get("skills_all_sources") or candidate_data.get("skills", [])

    # --- Mandatory Skills (with hard gate) ---
    mandatory_result = score_skill_list(jd_data.get("mandatory_skills", []), candidate_skills, embed_fn)
    mandatory_score = mandatory_result["average_contribution"] * WEIGHTS["mandatory_skills"] * 100
    # Gate on gate_missing, not missing: a weak semantic match (contribution as low as 0.6,
    # barely above the calibrated noise floor) still counts as "not missing" for score purposes,
    # but shouldn't be able to single-handedly satisfy a hard mandatory-skill gate the way an
    # exact or confident match does. See GATE_MIN_CONTRIBUTION in matcher.py.
    gate_missing_mandatory = mandatory_result["gate_missing"]
    hard_gate_failed = len(gate_missing_mandatory) > HARD_GATE_MAX_MISSING_MANDATORY
    hard_gate_reason = (
        f"Missing {len(gate_missing_mandatory)} mandatory skills (below gate confidence): {', '.join(gate_missing_mandatory)}"
        if hard_gate_failed else ""
    )

    # --- Technical / Preferred Skills ---
    tech_result = score_skill_list(jd_data.get("preferred_technical_skills", []), candidate_skills, embed_fn)
    tech_score = tech_result["average_contribution"] * WEIGHTS["technical_preferred_skills"] * 100

    # --- Preferred Skills (soft) ---
    soft_result = score_skill_list(jd_data.get("soft_preferred_skills", []), candidate_skills, embed_fn)
    soft_score = soft_result["average_contribution"] * WEIGHTS["preferred_skills_soft"] * 100

    # --- Certifications / Other ---
    cert_result = score_skill_list(
        jd_data.get("relevant_certifications", []), candidate_data.get("certifications", []), embed_fn
    )
    cert_score = cert_result["average_contribution"] * WEIGHTS["certifications_other"] * 100

    # --- Relevant Experience ---
    experience_score, experience_notes = _score_experience(candidate_data, jd_data, embed_fn)

    # --- Projects ---
    projects_score, projects_notes = _score_projects(candidate_data, jd_data, embed_fn)

    # --- Education ---
    education_score, education_notes = _score_education(candidate_data, jd_data)

    category_scores = {
        "mandatory_skills": {
            "score": round(mandatory_score, 2),
            "matched": mandatory_result["matched"],
            "missing": mandatory_result["missing"],          # zero-contribution skills, for display
            "gate_missing": gate_missing_mandatory,           # zero OR weak-below-threshold, drove the gate decision
        },
        "relevant_experience": {"score": experience_score, "notes": experience_notes},
        "technical_preferred_skills": {
            "score": round(tech_score, 2), "matched": tech_result["matched"], "missing": tech_result["missing"],
        },
        "projects": {"score": projects_score, "notes": projects_notes},
        "education": {"score": education_score, "notes": education_notes},
        "preferred_skills_soft": {"score": round(soft_score, 2), "matched": soft_result["matched"]},
        "certifications_other": {"score": round(cert_score, 2), "matched": cert_result["matched"]},
    }

    final_score = round(sum(c["score"] for c in category_scores.values()), 2)

    # Evidence view for HR-facing candidate detail pages: per-category matched/weak/missing
    # rows with 3-state status for green/amber/red display, plus a neutral "extra skills"
    # bucket for candidate skills the JD never asked about. Built from the same per-skill
    # results already computed above — no extra scoring work, just reshaped for display.
    evidence = {
        "mandatory_skills": _build_evidence("mandatory_skills", mandatory_result),
        "technical_preferred_skills": _build_evidence("technical_preferred_skills", tech_result),
        "preferred_skills_soft": _build_evidence("preferred_skills_soft", soft_result),
        "certifications_other": _build_evidence("certifications_other", cert_result),
        "additional_candidate_skills": _extra_candidate_skills(
            candidate_skills, mandatory_result, tech_result, soft_result
        ),
    }

    return {
        "category_scores": category_scores,
        "evidence": evidence,
        "final_score": final_score,
        "hard_gate_failed": hard_gate_failed,
        "hard_gate_reason": hard_gate_reason,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }