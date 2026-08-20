"""
debug_candidate_match.py — pinpoints WHY each requirement was scored the way it was
for one candidate, straight from the live pipeline, across EVERY scored category —
not just mandatory/preferred skills.

The UI only shows a checkmark or an x. This prints the actual `matched_against`
value behind every checkmark — the literal candidate skill string (for an
exact match) or the resume evidence chunk + judge reason (for an evidence-
based match) — so a false positive can be traced to its root cause instead
of guessed at from a screenshot.

Categories with a per-item skill list (mandatory_skills, preferred_skills, soft_skills,
industry_keywords — all scored via score_skill_list / matcher.py) get the full
matched_against / judge_reason / evidence-chunk breakdown. Categories with a different
evidence shape (experience, education, job_title_match) get a generic key/value dump —
this script deliberately doesn't hardcode their exact field names, so it keeps working
even if those evidence shapes change later; only the skill-row shape (a list of dicts
with a "skill" key) is treated specially.

USAGE:
    python3 debug_candidate_match.py <resume_path> <jd_path>

Also prints the candidate's full extracted skills_all_sources list up front —
worth checking by eye for any single long un-split entry that happens to
contain several required phrases as literal substrings (a run-on skills
blob), since that would satisfy the *unrestricted* direction of
_word_boundary_contains (required-phrase-found-inside-candidate-skill) even
after the reverse-direction fix, and would look identical to the original
"Data" bug from the outside.
"""

import json
import sys

from extractor import ingest_resume
from jd_extractor import ingest_jd
from scorer import calculate_job_fit, WEIGHTS

# Human-readable labels for whatever categories actually exist in WEIGHTS — printed in
# this order when present, so the output has a stable, predictable layout regardless of
# how many scoring categories the pipeline currently has.
_CATEGORY_LABELS = {
    "mandatory_skills": "MANDATORY SKILLS",
    "job_title_match": "JOB TITLE MATCH",
    "relevant_experience": "EXPERIENCE",
    "industry_keywords": "INDUSTRY KEYWORDS",
    "soft_skills": "SOFT SKILLS",
    "education": "EDUCATION",
    "preferred_skills": "PREFERRED SKILLS",
}


def _is_skill_row_list(evidence) -> bool:
    """True if `evidence` is the shape _build_evidence() in scorer.py produces —
    a list of dicts each with a "skill" key. mandatory_skills, preferred_skills,
    soft_skills, and industry_keywords all take this shape since they're all scored
    via matcher.score_skill_list."""
    return isinstance(evidence, list) and bool(evidence) and isinstance(evidence[0], dict) and "skill" in evidence[0]


def _print_skill_rows(rows: list):
    for r in rows:
        mark = "✓" if r.get("status") != "missing" else "✗"
        print(f"\n{mark} {r['skill']!r}")
        print(f"    status: {r.get('status')}   match_type: {r.get('match_type')}   contribution: {r.get('contribution')}")
        print(f"    matched_against: {r.get('matched_against')!r}")
        if r.get("judge_reason"):
            print(f"    judge_reason: {r['judge_reason']!r}")
        if r.get("match_type") == "evidence" and r.get("evidence"):
            for chunk in r["evidence"]:
                print(f"    evidence chunk [{chunk.get('source_type')} — {chunk.get('source_label')}] "
                      f"(bm25={chunk.get('bm25_score')}): {chunk.get('text', '')[:160]}...")


def _print_generic(evidence):
    """Fallback for any category whose evidence isn't a skill-row list (experience,
    education, job_title_match, or anything added later with a different shape) —
    deliberately doesn't assume field names, just pretty-prints whatever is there."""
    if evidence is None:
        print("  (no evidence recorded)")
        return
    try:
        print(json.dumps(evidence, indent=2, default=str))
    except TypeError:
        print(f"  {evidence!r}")


def _print_evidence_section(title: str, evidence):
    print(f"\n{'=' * 90}\n{title}\n{'=' * 90}")
    if evidence is None or evidence == [] or evidence == {}:
        print("  (empty — not applicable for this JD, or nothing scored)")
        return
    if _is_skill_row_list(evidence):
        _print_skill_rows(evidence)
    else:
        _print_generic(evidence)


def _print_category_summary(category_scores: dict):
    print(f"\n{'=' * 90}\nCATEGORY SCORES (of current WEIGHTS)\n{'=' * 90}")
    for key in _CATEGORY_LABELS:
        if key not in category_scores:
            continue
        detail = category_scores[key]
        weight_pct = round(WEIGHTS.get(key, 0) * 100, 2)
        score = detail.get("score")
        notes = detail.get("notes", "")
        line = f"  {_CATEGORY_LABELS[key]:<22} {score}/{weight_pct}"
        if notes:
            line += f"   — {notes}"
        print(line)
    # Catch any category present in the real output that this script's label map
    # doesn't know about yet, rather than silently dropping it from the summary.
    unknown = [k for k in category_scores if k not in _CATEGORY_LABELS]
    for key in unknown:
        detail = category_scores[key]
        print(f"  {key:<22} {detail.get('score')}   — {detail.get('notes', '')}")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python3 {sys.argv[0]} <resume_path> <jd_path>")
        return 1

    resume_path, jd_path = sys.argv[1], sys.argv[2]

    print(f"Ingesting resume: {resume_path}")
    candidate = ingest_resume(resume_path)
    print(f"Ingesting JD: {jd_path}")
    jd = ingest_jd(jd_path)

    print(f"\nCandidate: {candidate.get('candidate_name')}")
    print(f"extraction_warnings: {candidate.get('extraction_warnings')}")

    skills = candidate.get("skills_all_sources") or candidate.get("skills", [])
    print(f"\nskills_all_sources ({len(skills)} entries):")
    for s in skills:
        flag = "  <-- LONG / MULTI-CLAUSE, INSPECT THIS" if len(s.split()) > 4 or "," in s else ""
        print(f"  - {s!r}{flag}")

    result = calculate_job_fit(candidate, jd)

    _print_category_summary(result["category_scores"])

    evidence = result.get("evidence", {})
    for key, label in _CATEGORY_LABELS.items():
        if key in evidence:
            _print_evidence_section(label, evidence[key])
    # Any evidence key not in the known label map (e.g. a brand-new category) still
    # gets printed, just under its raw key name instead of a friendly label.
    for key in evidence:
        if key not in _CATEGORY_LABELS and key != "additional_candidate_skills":
            _print_evidence_section(key.upper(), evidence[key])

    if evidence.get("additional_candidate_skills"):
        print(f"\n{'=' * 90}\nADDITIONAL CANDIDATE SKILLS (not consumed by any requirement)\n{'=' * 90}")
        print(f"  {', '.join(evidence['additional_candidate_skills'])}")

    print(f"\n{'=' * 90}")
    print(f"final_score: {result['final_score']}   hard_gate_failed: {result['hard_gate_failed']}")
    print(f"hard_gate_reason: {result['hard_gate_reason']}")
    print(f"{'=' * 90}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())