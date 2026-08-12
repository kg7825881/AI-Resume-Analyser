import os
import json
import pdfplumber
import docx2txt
import ollama
from pdf2image import convert_from_path
import pytesseract

from common import new_id, now_iso

# Setting the data source to read directly from a local Kaggle directory path
DATA_DIR = os.path.expanduser("~/kaggle/input/talentlens-batch/")

MIN_TEXT_DENSITY_CHARS_PER_PAGE = 40  # below this, assume scanned/image PDF, fall back to OCR


def extract_text(file_path: str) -> tuple[str, list[str], str]:
    """
    Extracts raw text from PDF, DOCX, or TXT files.
    Returns (raw_text, extraction_warnings, extraction_method).
    PDFs with little/no extractable text (scanned/image PDFs) fall back to OCR.
    """
    ext = os.path.splitext(file_path)[1].lower()
    warnings = []

    if ext == '.pdf':
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            num_pages = len(pdf.pages)
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text_parts.append(extracted)
        raw_text = "\n".join(text_parts)
        avg_density = len(raw_text) / max(num_pages, 1)

        if avg_density >= MIN_TEXT_DENSITY_CHARS_PER_PAGE:
            return raw_text, warnings, "pdf_text"

        # Fall back to OCR — low text density strongly suggests a scanned/image PDF
        warnings.append("Low text density in PDF text layer — used OCR fallback.")
        images = convert_from_path(file_path)
        ocr_parts = [pytesseract.image_to_string(img) for img in images]
        ocr_text = "\n".join(ocr_parts).strip()
        if not ocr_text:
            warnings.append("OCR fallback also produced no text — file may be blank, corrupt, or unreadable.")
        return ocr_text, warnings, "pdf_ocr"

    elif ext == '.docx':
        text = docx2txt.process(file_path)
        if not text or not text.strip():
            warnings.append("No extractable text found in DOCX — file may be empty or malformed.")
        return text, warnings, "docx"

    elif ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text, warnings, "txt"

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def extract_structured_evidence(resume_text):
    """Sends raw resume text to Qwen to extract structured JSON with strict limits."""

    system_prompt = (
        "You are an expert AI Resume Extractor. Your task is to extract information from the provided resume "
        "and output it EXACTLY matching the following JSON schema. Do not include any conversational text outside the JSON.\n\n"
        "IMPORTANT RULES:\n"
        "- Extract EVERY item explicitly listed under each resume section (every skill, every experience entry, "
        "every project, every certification, every education entry) — do not stop early or summarize a subset.\n"
        "- Only place an item in \"projects\" if it appears under a section literally titled Projects (or similar). "
        "Do NOT create a project entry from an Experience/Internship bullet point, and do not duplicate the same "
        "content across two fields.\n"
        "- If a field has no information in the resume, use an empty array or empty string — never invent content.\n\n"
        "SCHEMA:\n"
        "{\n"
        '  "candidate_name": "string (extracted from resume)",\n'
        '  "skills": ["string", "string"],\n'
        '  "total_years_experience": number,\n'
        '  "experience": [\n'
        '    {\n'
        '      "title": "string",\n'
        '      "company": "string",\n'
        '      "years": number,\n'
        '      "domain": "string (inferred domain/industry)",\n'
        '      "description": "string"\n'
        '    }\n'
        '  ],\n'
        '  "education": [\n'
        '    { "degree_level": "string", "field": "string", "institution": "string" }\n'
        '  ],\n'
        '  "certifications": ["string", "string"],\n'
        '  "projects": [\n'
        '    { "title": "string", "description": "string" }\n'
        '  ]\n'
        "}"
    )

    response = ollama.chat(
        model='qwen3:8b',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Resume Text:\n{resume_text}"}
        ],
        format='json',
        think=False,   # Qwen3 thinks by default, which eats into num_predict and can starve
                       # the actual JSON output on longer documents — disable it here.
        options={
            'temperature': 0.0,
            'num_predict': 4096   # Raised from 2500 now that thinking tokens aren't competing for budget
        }
    )

    raw_output = response['message']['content']

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        print("\n[WARNING] Model failed to return a complete JSON object. Returning raw text.")
        return {"error": "Incomplete JSON", "raw_output": raw_output}


def ingest_resume(file_path: str) -> dict:
    """
    Full resume ingestion: extract text (with OCR fallback) -> structure via LLM ->
    attach candidate_id/document_id/file_name/extraction metadata per the Phase 1 schema.
    """
    file_name = os.path.basename(file_path)
    raw_text, warnings, method = extract_text(file_path)
    structured = extract_structured_evidence(raw_text)

    structured["candidate_id"] = new_id()
    structured["document_id"] = new_id()
    structured["file_name"] = file_name
    structured["extraction_method"] = method
    structured["extraction_warnings"] = warnings
    structured["uploaded_at"] = now_iso()
    return structured


if __name__ == "__main__":
    sample_resume = os.path.join(DATA_DIR, "candidate_1.pdf")

    if os.path.exists(sample_resume):
        print("Ingesting resume...")
        structured_data = ingest_resume(sample_resume)
        print(json.dumps(structured_data, indent=2))
    else:
        print(f"Please ensure your test resume is placed in: {DATA_DIR}")