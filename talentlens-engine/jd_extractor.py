import os
import re
import json
import logging
import ollama
from extractor import extract_text
from common import new_id, now_iso, slugify
from jd_skills import atomize_skill_list

logger = logging.getLogger("talentlens.jd_extractor")

# JD fields that hold discrete, individually-matchable skill/tool/cert names — these
# get the deterministic atomic-splitting pass. "responsibilities" is deliberately
# excluded: scorer.py treats it as free text for semantic-similarity comparison against
# candidate projects/experience, not as a per-item exact-match list, so splitting it
# would only lose the sentence context that comparison relies on.
_ATOMIC_SKILL_FIELDS = (
    "mandatory_skills",
    "preferred_technical_skills",
    "soft_preferred_skills",
    "relevant_certifications",
)

# Data path set to local Kaggle directory
DATA_DIR = os.path.expanduser("~/kaggle/input/talentlens-batch/")


# --- Fabrication guard: min_years_experience / education_requirements ---
#
# Confirmed on a real JD (Sr Java Lead Engineer — 2026.docx, which has no Qualifications/
# Requirements section at all): the model returned "10+ yrs experience" and a full
# Bachelor's/Master's-in-CS education requirement even though NEITHER appears anywhere
# in the source document (verified against the raw docx XML, not just the parsed
# paragraphs). It filled in what's typical for this kind of role from its training
# prior, not what the JD actually says — a much riskier failure than a missed skill,
# since a fabricated "10+ yrs required" can hard-filter out qualified candidates against
# a bar the employer never set.
#
# This mirrors the guard already built in extractor.py for fabricated company names
# (_looks_like_non_company_label / _companies_match): don't trust a claim the model
# makes about the source document without deterministically checking the source
# document actually supports it.

_YEARS_EXPERIENCE_MENTION_RE = re.compile(r"\b\d{1,2}\s*\+?\s*(?:-|to)?\s*\d{0,2}\s*\+?\s*years?\b", re.IGNORECASE)

_DEGREE_KEYWORDS = (
    "bachelor", "master", "b.tech", "btech", "b.e.", "m.tech", "mtech",
    "b.s.", "bsc", "b.sc", "m.s.", "msc", "m.sc", "mba", "phd", "ph.d",
    "degree", "diploma", "graduate", "undergraduate",
)


def _text_mentions_years_experience(raw_text: str) -> bool:
    return bool(_YEARS_EXPERIENCE_MENTION_RE.search(raw_text or ""))


def _text_mentions_education_requirement(raw_text: str) -> bool:
    lowered = (raw_text or "").lower()
    return any(keyword in lowered for keyword in _DEGREE_KEYWORDS)


def _guard_against_fabricated_requirements(structured: dict, jd_text: str) -> list:
    """Zeroes out min_years_experience / education_requirements if the raw JD text
    contains no language supporting them, since a nonzero/nonempty value here can only
    have come from the model, not the document. Returns warnings describing what (if
    anything) was discarded, so this stays reviewable rather than silently corrected."""
    warnings = []

    years = structured.get("min_years_experience")
    if years:
        if not _text_mentions_years_experience(jd_text):
            warnings.append(
                f"min_years_experience ({years}) discarded — no years-of-experience phrase "
                f"found anywhere in the JD text. This looks inferred from the role title/"
                f"seniority rather than stated in the document. Verify manually if a minimum "
                f"years requirement should apply to this role."
            )
            structured["min_years_experience"] = 0

    edu_reqs = structured.get("education_requirements")
    if edu_reqs:
        if not _text_mentions_education_requirement(jd_text):
            warnings.append(
                f"education_requirements ({len(edu_reqs)} entries) discarded — no degree/"
                f"education keyword (Bachelor's, Master's, degree, etc.) found anywhere in "
                f"the JD text. This looks inferred from the role type rather than stated in "
                f"the document. Verify manually if an education requirement should apply."
            )
            structured["education_requirements"] = []

    return warnings


def extract_structured_jd(jd_text):
    """Sends raw job description text to Qwen to extract structured JSON with strict limits."""

    system_prompt = (
        "You are an expert AI Job Description Parser. Your task is to extract information from the provided job description "
        "and output it EXACTLY matching the following JSON schema. Do not include any conversational text outside the JSON.\n\n"
        "IMPORTANT RULES:\n"
        "- Extract EVERY item explicitly listed under each JD section (every mandatory skill, every preferred skill, "
        "every responsibility, every certification, every education requirement) — do not stop early.\n"
        "- Read the ENTIRE document before producing output, including sections near the end (responsibilities, "
        "certifications, and education requirements are often listed later in a JD and are frequently missed — "
        "check specifically for them).\n"
        "- min_years_experience and education_requirements are especially high-risk to invent: many JDs — including "
        "ones for senior roles — simply do NOT state a minimum years figure or a degree requirement at all. Do NOT "
        "fill in a number or degree that seems typical for this kind of role based on its title or seniority. If "
        "the JD text does not contain an explicit years-of-experience phrase, min_years_experience MUST be 0. If "
        "the JD text does not contain an explicit degree/education requirement, education_requirements MUST be an "
        "empty array. A missing requirement in the output is correct and expected when the source document doesn't "
        "state one — it is not something to fill in.\n"
        "- If a field genuinely has no information in the JD, use an empty array or empty string — never invent content.\n\n"
        "ATOMIC SKILL EXTRACTION (applies to mandatory_skills, preferred_technical_skills, "
        "soft_preferred_skills, and relevant_certifications):\n"
        "- Each array item must be ONE atomic skill, tool, technology, or named concept — NEVER a full "
        "requirement sentence. A JD line commonly bundles several skills into one sentence; split it into "
        "one entry per named skill/tool/concept instead of keeping the sentence whole. Examples:\n"
        "    \"Strong SQL and good Python skills.\" -> [\"SQL\", \"Python\"]\n"
        "    \"Experience building ETL or ELT pipelines in production.\" -> [\"ETL\", \"ELT\"]\n"
        "    \"Hands-on experience with data validation, deduplication, consistency checks, and data "
        "quality control\" -> [\"Data Validation\", \"Deduplication\", \"Consistency Checks\", \"Data "
        "Quality Control\"]\n"
        "    \"Familiarity with vector databases, embedding pipelines, or retrieval datasets.\" -> "
        "[\"Vector Databases\", \"Embedding Pipelines\", \"Retrieval Datasets\"]\n"
        "- Strip filler phrasing around the skill itself — \"experience with\", \"strong understanding "
        "of\", \"ability to\", \"hands-on\", \"familiarity with\", \"knowledge of\" — keep only the "
        "skill/tool/concept name, not the surrounding sentence.\n"
        "- This matters for downstream matching: each entry is compared individually against a "
        "candidate's own skill list. A full sentence can never exact-match or closely semantic-match a "
        "short skill name, so keeping requirements un-split silently breaks matching even when the "
        "candidate genuinely has the skill.\n"
        "- Still extract EVERY distinct skill/tool/concept mentioned anywhere in the JD's requirements — "
        "atomizing is about granularity, not about dropping items.\n\n"
        "SCHEMA:\n"
        "{\n"
        '  "role_title": "string",\n'
        '  "department": "string",\n'
        '  "mandatory_skills": ["string", "string"],\n'
        '  "preferred_technical_skills": ["string", "string"],\n'
        '  "soft_preferred_skills": ["string", "string"],\n'
        '  "min_years_experience": number,\n'
        '  "education_requirements": [\n'
        '    { "degree_level": "string", "field": "string", "required": boolean }\n'
        '  ],\n'
        '  "relevant_certifications": ["string", "string"],\n'
        '  "responsibilities": ["string", "string"]\n'
        "}"
    )

    response = ollama.chat(
        model='qwen2.5:3b-instruct',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Job Description Text:\n{jd_text}"}
        ],
        format='json',
        options={
            'temperature': 0.0,
            'num_predict': 4096
        }
    )

    raw_output = response['message']['content']

    try:
        structured = json.loads(raw_output)
    except json.JSONDecodeError:
        print("\n[WARNING] Model failed to return a complete JSON object. Returning raw text.")
        return {"error": "Incomplete JSON", "raw_output": raw_output}

    # Deterministic safety net #1, run regardless of whether the model followed the
    # ATOMIC SKILL EXTRACTION instructions above — confirmed in production it doesn't
    # always (e.g. "Strong SQL and good Python skills." coming through as one unsplit
    # item). See jd_skills.py for what this does and does not attempt to split.
    for field in _ATOMIC_SKILL_FIELDS:
        if field in structured and isinstance(structured[field], list):
            structured[field] = atomize_skill_list(structured[field])

    # Deterministic safety net #2 — confirmed in production the model fabricates
    # min_years_experience/education_requirements for terse JDs that don't state them.
    # Stashed under a leading-underscore key rather than changed as a second return
    # value, so this stays a drop-in match for the existing return-a-dict signature
    # (and every existing caller/test that does extract_structured_jd(...)["field"]) —
    # ingest_jd() below pops it back out before attaching real extraction_warnings.
    structured["_fabrication_guard_warnings"] = _guard_against_fabricated_requirements(structured, jd_text)

    return structured


def ingest_jd(file_path: str) -> dict:
    """
    Full JD ingestion: extract text (with OCR fallback, via extractor.extract_text) ->
    structure via LLM, with a deterministic pass to atomize skill lists and to strip
    any min_years_experience/education_requirements the source text doesn't actually
    support -> attach document_id/role_id (slug)/file_name/extraction metadata.
    role_id is auto-generated from role_title per the dynamic role-handling design —
    no fixed taxonomy needed.
    """
    file_name = os.path.basename(file_path)
    raw_text, warnings, method = extract_text(file_path)
    structured = extract_structured_jd(raw_text)
    warnings = warnings + structured.pop("_fabrication_guard_warnings", [])

    # Same reasoning as extractor.py: extraction-quality warnings go to server logs,
    # not into the API response the frontend renders — see api.py's /jds/upload, which
    # deliberately does not forward extraction_warnings from the DB record.
    for w in warnings:
        logger.warning("[%s] %s", file_name, w)

    structured["document_id"] = new_id()
    structured["role_id"] = slugify(structured.get("role_title", ""))
    structured["file_name"] = file_name
    structured["extraction_method"] = method
    structured["extraction_warnings"] = warnings
    structured["uploaded_at"] = now_iso()
    return structured


if __name__ == "__main__":
    jd_path = os.path.join(DATA_DIR, "job_description.pdf")

    if os.path.exists(jd_path):
        print("Ingesting Job Description...")
        structured_jd = ingest_jd(jd_path)
        print(json.dumps(structured_jd, indent=2))
    else:
        print(f"Please ensure job_description.pdf is placed in: {DATA_DIR}")