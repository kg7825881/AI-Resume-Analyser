"""
scorer.py — scoring engine implementing the simplified methodology:
  - Mandatory Skills (40%): exact match earns full credit for free; anything without an
    exact match is retrieved via BM25 (retrieval.py) and judged by a small local LLM
    (judge.py) into direct/related/weak/none, then converted to a numeric contribution
    by matcher.py — same pipeline preferred skills already use. A skill still needs to
    clear GATE_MIN_CONTRIBUTION (currently only a "direct" judge match does) to satisfy
    the hard mandatory-skill gate; a "related"/"weak" evidence match earns partial score
    toward the category but does NOT by itself clear the gate.
  - Relevant Experience (20%): purely checks total_years >= JD min_years
  - Job Title Match (15%): candidate's past titles (experience entries) plus, if
    extractor.py found one, the current title stated in the resume's Summary/Profile
    section (current_role_title_from_summary) — judged via the SAME judge (judge.py)
    used for skill evidence, in one call covering all titles at once. This replaced a
    pure token-overlap approach specifically because it missed genuinely related titles
    that share no words (e.g. "AI Architect" vs required "Data Engineer") — see
    job_title_matcher.py's module docstring. Not a hard gate.
  - Industry Keywords (10%): JD-stated industry/domain terms (e.g. "Fintech",
    "Healthcare"), matched via the same BM25 + judge evidence pipeline as skills, guarded
    against fabrication the same way min_years_experience is (see jd_extractor.py).
  - Soft Skills (5%): the JD's soft_preferred_skills, now scored as its own category
    instead of being folded into Preferred Skills — same evidence pipeline.
  - Education (5%): degree match against JD education_requirements
  - Preferred Skills (5%): technical preferred skills only (soft skills split out above);
    exact match, or BM25-retrieved evidence + LLM judgment

Non-exact skill/keyword matching (mandatory, industry, soft, and preferred) all go
through matcher.py's evidence pipeline: BM25 retrieval (retrieval.py) finds the
candidate's most relevant resume text for a requirement, then a small local LLM
(judge.py) classifies how well that evidence supports it. See matcher.py's module
docstring for why this replaced the previous embedding-cosine-similarity approach.

Weight rebalance note: the previous version of this module weighted Mandatory Skills
50% / Relevant Experience 30% / Education 10% / Preferred Skills 10%. Job Title Match,
Industry Keywords, and Soft Skills were added as new categories by shrinking the
existing four proportionally rather than exceeding 100% — these are plain constants in
WEIGHTS below and are meant to be retuned once real scoring output is reviewed.
"""
import re
from datetime import datetime, timezone

from job_title_matcher import score_job_titles
from judge import judge_evidence
from matcher import score_skill_list, evidence_status
from retrieval import CandidateEvidenceIndex

WEIGHTS = {
    "mandatory_skills": 0.25,
    "relevant_experience": 0.25,
    "job_title_match": 0.05,
    "industry_keywords": 0.10,
    "soft_skills": 0.10,
    "education": 0.20,
    "preferred_skills": 0.05,
}

HARD_GATE_MAX_MISSING_MANDATORY = 6 # Fail the gate if ANY mandatory skill doesn't clear GATE_MIN_CONTRIBUTION


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


def _score_job_title(candidate_data: dict, jd_data: dict, judge_fn=judge_evidence) -> tuple[float, str, dict]:
    """
    Job-title relevance via job_title_matcher.score_job_titles — a single judge call
    covering ALL of the candidate's past titles (plus their summary-stated current
    title, if extractor.py found one) as evidence together, classified against the
    JD's role_title the same way skill evidence is classified. Not a hard gate — a
    title mismatch costs score but never fails the candidate outright, since a
    genuinely qualified candidate can carry an unconventional past title (e.g. "Data
    Wrangler" instead of "Data Engineer").
    """
    role_title = jd_data.get("role_title", "")
    if not role_title.strip():
        return WEIGHTS["job_title_match"] * 100, "No role title on JD to compare against", {}

    result = score_job_titles(candidate_data, role_title, judge_fn)
    score = result["contribution"] * WEIGHTS["job_title_match"] * 100

    if result["best_match"]:
        notes = (
            f"Best match: \"{result['best_match']['title']}\" at {result['best_match']['company']} "
            f"— judged {result['match_level']} against required \"{role_title}\""
            + (f" ({result['judge_reason']})" if result.get("judge_reason") else "")
        )
    else:
        notes = f"No past job title to compare against required \"{role_title}\""

    return round(score, 2), notes, result

# Degree hierarchy & equivalent acronyms
DEGREE_TIERS = {
    "bachelor": {"bachelor", "btech", "b.tech", "be", "b.e", "bs", "b.s", "bsc", "b.sc", "bca", "undergraduate", "ug"},
    "master": {"master", "mtech", "m.tech", "me", "m.e", "ms", "m.s", "msc", "m.sc", "mca", "mba", "postgraduate", "pg"},
    "phd": {"phd", "ph.d", "doctorate"},
    "diploma": {"diploma", "associate"},
}

def _get_degree_tier(text: str) -> str:
    norm = re.sub(r"[^a-z0-9]", "", (text or "").lower())
    for tier, keywords in DEGREE_TIERS.items():
        if any(re.sub(r"[^a-z0-9]", "", kw) in norm for kw in keywords):
            return tier
    return "other"

def _fields_overlap(req_field: str, cand_edu_entries: list[dict]) -> bool:
    """True if candidate field/degree text contains any major keywords from the required field."""
    # 1. Lowercase and remove all punctuation from the JD requirement
    req_clean = re.sub(r"[^a-z0-9\s]", " ", (req_field or "").lower()).strip()
    if not req_clean or any(k in req_clean for k in ["any", "related field", "relevant field", "all fields"]):
        return True

    # 2. Extract distinct keywords/phrases from the cleaned requirement
    raw_keywords = [w.strip() for w in re.split(r"[,/|]|\bor\b|\band\b", req_clean) if len(w.strip()) >= 2]
    
    # 3. Lowercase AND remove all punctuation from the candidate's education text
    cand_texts = [
        re.sub(r"[^a-z0-9\s]", " ", f"{e.get('degree_level', '')} {e.get('field', '')}".lower())
        for e in cand_edu_entries
    ]

    # 4. Compare the clean, lowercased keywords against the clean, lowercased candidate text
    for kw in raw_keywords:
        kw_tokens = kw.split()
        for cand_text in cand_texts:
            if kw in cand_text or all(tok in cand_text for tok in kw_tokens):
                return True
    return False

def _score_education(candidate_data: dict, jd_data: dict) -> tuple[float, str, list]:
    """
    Evaluates education against JD requirements using keyword matching.
    If no education is specified in JD -> 100% full credit automatically.
    If JD specifies education -> evaluates degrees & fields flexibly.
    """
    requirements = jd_data.get("education_requirements", [])
    candidate_edu = candidate_data.get("education", [])

    # Case 1: No education required in JD -> Full score
    if not requirements:
        return WEIGHTS["education"] * 100, "No specific education requirement in JD", []

    education_evidence = []
    any_matched = False
    
    cand_tiers = {_get_degree_tier(e.get("degree_level", "")) for e in candidate_edu}
    # A Master's or PhD degree also fulfills a Bachelor's requirement
    if "master" in cand_tiers or "phd" in cand_tiers:
        cand_tiers.add("bachelor")

    for req in requirements:
        req_level = req.get("degree_level", "")
        req_field = req.get("field", "")
        req_tier = _get_degree_tier(req_level)

        level_match = (req_tier == "other") or (req_tier in cand_tiers)
        field_match = _fields_overlap(req_field, candidate_edu)
        matched = level_match and field_match

        if matched:
            any_matched = True

        education_evidence.append({
            "required_degree_level": req_level or "Degree",
            "required_field": req_field or "Any Field",
            "status": "matched" if matched else "missing",
        })

    # If any acceptable degree requirement from JD is met, award full credit (OR logic)
    score_fraction = 1.0 if any_matched else 0.0
    score = score_fraction * WEIGHTS["education"] * 100
    notes = "Education criteria met" if any_matched else "Required education not found in resume"

    return round(score, 2), notes, education_evidence

def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _build_evidence(result: dict) -> list[dict]:
    """
    Converts one score_skill_list result into the per-skill rows for the UI.
    For evidence-based (non-exact) matches, this also carries the retrieved
    resume text and the judge's one-line reason — that's the whole point of
    the BM25 + judge pipeline over a bare similarity score: "why did this
    match?" now has an actual, inspectable answer to show HR. This applies
    to mandatory skills too now, not just preferred.
    """
    rows = []
    for r in result["results"]:
        rows.append({
            "skill": r["skill"],
            "status": evidence_status(r),
            "match_type": r["match_type"],
            "matched_against": r.get("matched_against"),
            "contribution": r["contribution"],
            "evidence": r.get("evidence", []),
            "judge_reason": r.get("judge_reason", ""),
        })
    return rows


def _extra_candidate_skills(candidate_skills: list[str], *results: dict) -> list[str]:
    """
    Extracts candidate skills not consumed by any JD requirement. Only exact
    matches actually "consume" an entry from candidate_skills — evidence-based
    matches point at a resume text chunk (matched_against is a source label
    like "Data Engineer at Foo Corp"), not a specific skill string, so they
    can't be excluded from this list the same way.
    """
    used = set()
    for result in results:
        for r in result["results"]:
            if r["match_type"] == "exact" and r.get("matched_against"):
                used.add(r["matched_against"].strip().lower())
    return [s for s in candidate_skills if s.strip().lower() not in used]


def calculate_job_fit(candidate_data: dict, jd_data: dict, judge_fn=judge_evidence) -> dict:
    """
    Main scoring entry point.

    Builds one CandidateEvidenceIndex (BM25 over this candidate's experience/
    project/skills text — see retrieval.py) up front and reuses it for every
    mandatory, preferred, industry-keyword, and soft-skill requirement below,
    rather than rebuilding it per requirement.

    Mandatory skills (exact_only=False): an exact/word-boundary match still earns
    free full credit with no model call. Anything without one goes through BM25 +
    judge evidence scoring — same pipeline preferred/industry/soft skills use — so a
    mandatory skill genuinely described in different words on the resume (e.g. "set up
    ETL pipelines" vs required "ETL") can now earn credit and clear the hard gate,
    rather than being automatically zeroed out just for not being a literal
    skills-list entry.

    Job Title Match uses the judge too now (see job_title_matcher.py), but in one
    call covering all of the candidate's titles together, not the BM25 retrieval
    pipeline — titles are short enough that retrieval doesn't add anything.
    """
    candidate_skills = candidate_data.get("skills_all_sources") or candidate_data.get("skills", [])
    evidence_index = CandidateEvidenceIndex(candidate_data)

    # --- Mandatory Skills (exact match earns free credit; otherwise evidence-based) ---
    jd_mandatory_skills = jd_data.get("mandatory_skills", [])
    mandatory_result = score_skill_list(
        jd_mandatory_skills, candidate_skills, evidence_index, judge_fn, exact_only=False
    )
    mandatory_total = len(jd_mandatory_skills)
    mandatory_score = mandatory_result["average_contribution"] * WEIGHTS["mandatory_skills"] * 100

    gate_missing_mandatory = mandatory_result["gate_missing"]
    hard_gate_failed = len(gate_missing_mandatory) > HARD_GATE_MAX_MISSING_MANDATORY
    hard_gate_reason = (
        f"{len(gate_missing_mandatory)} of {mandatory_total} mandatory skills missing or not "
        f"confidently evidenced: {', '.join(gate_missing_mandatory)}"
        if hard_gate_failed else ""
    )

    # --- Preferred Skills (technical only — soft skills are their own category below) ---
    jd_preferred_skills = jd_data.get("preferred_technical_skills", [])
    pref_result = score_skill_list(
        jd_preferred_skills, candidate_skills, evidence_index, judge_fn, exact_only=False
    )
    pref_score = pref_result["average_contribution"] * WEIGHTS["preferred_skills"] * 100

    # --- Soft Skills (split out of the old combined preferred_skills bucket) ---
    jd_soft_skills = jd_data.get("soft_preferred_skills", [])
    soft_result = score_skill_list(
        jd_soft_skills, candidate_skills, evidence_index, judge_fn, exact_only=False
    )
    soft_score = soft_result["average_contribution"] * WEIGHTS["soft_skills"] * 100

    # --- Industry Keywords (JD-stated industry/domain terms, fabrication-guarded on
    # the JD side — see jd_extractor.py's _guard_against_fabricated_industry_keywords) ---
    jd_industry_keywords = jd_data.get("industry_keywords", [])
    industry_result = score_skill_list(
        jd_industry_keywords, candidate_skills, evidence_index, judge_fn, exact_only=False
    )
    industry_score = industry_result["average_contribution"] * WEIGHTS["industry_keywords"] * 100

    # --- Job Title Match (judge-based, one call over all titles — see job_title_matcher.py) ---
    job_title_score, job_title_notes, job_title_evidence = _score_job_title(candidate_data, jd_data, judge_fn)

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
        "job_title_match": {"score": job_title_score, "notes": job_title_notes},
        "industry_keywords": {
            "score": round(industry_score, 2),
            "matched": industry_result["matched"],
            "missing": industry_result["missing"],
        },
        "soft_skills": {
            "score": round(soft_score, 2),
            "matched": soft_result["matched"],
            "missing": soft_result["missing"],
        },
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
        "soft_skills": _build_evidence(soft_result),
        "industry_keywords": _build_evidence(industry_result),
        "job_title": job_title_evidence,
        "experience": experience_evidence,
        "education": education_evidence,
        "additional_candidate_skills": _extra_candidate_skills(
            candidate_skills, mandatory_result, pref_result, soft_result, industry_result
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