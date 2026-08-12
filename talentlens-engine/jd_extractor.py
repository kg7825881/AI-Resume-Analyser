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