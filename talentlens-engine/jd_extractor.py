import os
import json
import ollama
from extractor import extract_text
from common import new_id, now_iso, slugify

# Data path set to local Kaggle directory
DATA_DIR = os.path.expanduser("~/kaggle/input/talentlens-batch/")


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
        return json.loads(raw_output)
    except json.JSONDecodeError:
        print("\n[WARNING] Model failed to return a complete JSON object. Returning raw text.")
        return {"error": "Incomplete JSON", "raw_output": raw_output}


def ingest_jd(file_path: str) -> dict:
    """
    Full JD ingestion: extract text (with OCR fallback, via extractor.extract_text) ->
    structure via LLM -> attach document_id/role_id (slug)/file_name/extraction metadata.
    role_id is auto-generated from role_title per the dynamic role-handling design —
    no fixed taxonomy needed.
    """
    file_name = os.path.basename(file_path)
    raw_text, warnings, method = extract_text(file_path)
    structured = extract_structured_jd(raw_text)

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