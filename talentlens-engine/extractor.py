import os
import json
import logging

import ollama
import pymupdf as fitz  # PyMuPDF — `import fitz` directly is deprecated, `import pymupdf as fitz` is the current form
import pymupdf4llm
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from pdf2image import convert_from_path
import pytesseract

from common import new_id, now_iso
from experience import compute_total_years

logger = logging.getLogger("talentlens.extractor")

# Setting the data source to read directly from a local Kaggle directory path
DATA_DIR = os.path.expanduser("~/kaggle/input/talentlens-batch/")

MIN_TEXT_DENSITY_CHARS_PER_PAGE = 40  # below this, assume scanned/image PDF, fall back to OCR


# --- Stage 1: deterministic extraction (no LLM) ---

def _iter_block_items(doc: Document):
    """
    Yields paragraphs and tables in true document order. python-docx's .paragraphs and
    .tables are separate flat lists with no ordering between them, so a resume with e.g.
    a skills table in the middle of prose would come out with the table's content moved
    to the end — this walks the underlying XML body directly to preserve real order.
    """
    for child in doc.element.body.iterchildren():
        if child.tag == qn('w:p'):
            yield Paragraph(child, doc)
        elif child.tag == qn('w:tbl'):
            yield Table(child, doc)


def _docx_to_markdown(file_path: str) -> str:
    """Structure-aware DOCX -> markdown: headings become #, list items become -, tables
    become markdown tables — all exact text, no summarization or rephrasing."""
    doc = Document(file_path)
    lines = []

    for block in _iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            style_name = (block.style.name or "").lower()
            if "heading" in style_name or "title" in style_name:
                level = next((int(ch) for ch in style_name if ch.isdigit()), 1)
                lines.append(f"{'#' * min(max(level, 1), 6)} {text}")
            elif "list" in style_name:
                lines.append(f"- {text}")
            else:
                lines.append(text)

        elif isinstance(block, Table):
            rows = block.rows
            if not rows:
                continue
            header = [c.text.strip() for c in rows[0].cells]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("| " + " | ".join("---" for _ in header) + " |")
            for row in rows[1:]:
                cells = [c.text.strip() for c in row.cells]
                lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def extract_text(file_path: str) -> tuple[str, list[str], str]:
    """
    Extracts exact text from PDF, DOCX, or TXT files as markdown (PDF, DOCX) or plain
    text (TXT) — deterministic, no LLM involved at this stage.
    Returns (raw_text, extraction_warnings, extraction_method).
    PDFs with little/no extractable text (scanned/image PDFs) fall back to OCR.
    """
    ext = os.path.splitext(file_path)[1].lower()
    warnings = []

    if ext == '.pdf':
        with fitz.open(file_path) as doc:
            num_pages = doc.page_count

        raw_text = pymupdf4llm.to_markdown(file_path)
        avg_density = len(raw_text) / max(num_pages, 1)

        if avg_density >= MIN_TEXT_DENSITY_CHARS_PER_PAGE:
            return raw_text, warnings, "pdf_markdown"

        # Fall back to OCR — low text density strongly suggests a scanned/image PDF
        warnings.append("Low text density in PDF text layer — used OCR fallback.")
        images = convert_from_path(file_path)
        ocr_parts = [pytesseract.image_to_string(img) for img in images]
        ocr_text = "\n".join(ocr_parts).strip()
        if not ocr_text:
            warnings.append("OCR fallback also produced no text — file may be blank, corrupt, or unreadable.")
        return ocr_text, warnings, "pdf_ocr"

    elif ext == '.docx':
        raw_text = _docx_to_markdown(file_path)
        if not raw_text or not raw_text.strip():
            warnings.append("No extractable text found in DOCX — file may be empty or malformed.")
        return raw_text, warnings, "docx_markdown"

    elif ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text, warnings, "txt"

    else:
        raise ValueError(f"Unsupported file type: {ext}")


# --- Stage 2: LLM structuring only (exact text -> JSON, no arithmetic) ---

def extract_structured_evidence(resume_text):
    """Sends already-exact resume markdown to Qwen to extract structured JSON. The model's
    only job is re-formatting text that's already in front of it — it does not compute
    durations or a total, and is not asked to."""

    system_prompt = (
        "You are an expert AI Resume Extractor. Your task is to extract information from the provided resume "
        "and output it EXACTLY matching the following JSON schema. Do not include any conversational text outside the JSON.\n\n"
        "IMPORTANT RULES:\n"
        "- Extract EVERY item explicitly listed under each resume section (every skill, every experience entry, "
        "every project, every certification, every education entry) — do not stop early or summarize a subset.\n"
        "- Only place an item in \"projects\" if it appears under a section literally titled Projects (or similar). "
        "Do NOT create a project entry from an Experience/Internship bullet point, and do not duplicate the same "
        "content across two fields.\n"
        "- For each experience entry, copy start_date_raw and end_date_raw EXACTLY as written in the resume text "
        "(e.g. \"Oct 2023\", \"Mar 2022\", \"2019\"). Do NOT calculate a duration, a number of years, or a total — "
        "that is computed separately, outside your output.\n"
        "- If the resume says the role is ongoing — using ANY wording such as \"Present\", \"Current\", \"Till "
        "date\", \"Ongoing\", or similar — copy that exact word as end_date_raw. This is a literal copy of text "
        "that IS on the page, not a guess. Only use an empty string for end_date_raw in the rare case where the "
        "resume shows a start date but truly no end date and no ongoing-indicator word at all.\n"
        "- For each experience entry, also extract technologies_used: the specific tools/technologies/platforms "
        "listed for THAT role specifically — e.g. a line like \"Tools and Techniques Used – Python, Docker, "
        "Kubernetes\" appearing under that job. Keep this separate from the top-level skills list below.\n"
        "- For each project entry, extract technologies_used the same way — tools/technologies mentioned for "
        "that specific project. Keep this separate from both the top-level skills list and from any experience "
        "entry's technologies_used.\n"
        "- The top-level \"skills\" field is ONLY for items listed under a resume section literally titled "
        "Skills / Technical Skills / Skills and Technologies (or similar) — not items pulled from job or project "
        "descriptions, even if they'd fit there too.\n"
        "- If a field has no information in the resume, use an empty array or empty string — never invent content.\n\n"
        "SCHEMA:\n"
        "{\n"
        '  "candidate_name": "string (extracted from resume)",\n'
        '  "skills": ["string", "string"],\n'
        '  "experience": [\n'
        '    {\n'
        '      "title": "string",\n'
        '      "company": "string",\n'
        '      "start_date_raw": "string — copied exactly as written, e.g. \'Oct 2023\'",\n'
        '      "end_date_raw": "string — copied exactly as written, e.g. \'Present\' or \'Mar 2022\'",\n'
        '      "domain": "string (inferred domain/industry)",\n'
        '      "description": "string",\n'
        '      "technologies_used": ["string", "string"]\n'
        '    }\n'
        '  ],\n'
        '  "education": [\n'
        '    { "degree_level": "string", "field": "string", "institution": "string" }\n'
        '  ],\n'
        '  "certifications": ["string", "string"],\n'
        '  "projects": [\n'
        '    { "title": "string", "description": "string", "technologies_used": ["string", "string"] }\n'
        '  ]\n'
        "}"
    )

    response = ollama.chat(
        model='qwen2.5:3b-instruct',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Resume Text:\n{resume_text}"}
        ],
        format='json',
        # No think=False here — qwen2.5 doesn't have qwen3's thinking-token mode,
        # so there's nothing to disable.
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


def _aggregate_skills(structured: dict) -> list:
    """Merges the resume's own Skills-section list with every technologies_used entry
    from experience roles and projects, deduped case-insensitively, preserving first-seen
    casing/order. Kept as a separate field (skills_all_sources) rather than overwriting
    "skills" so "skills" still faithfully reflects just the resume's own Skills section —
    useful for display/explainability even after scoring uses the broader set."""
    seen = set()
    aggregated = []

    def _add(item):
        if not item or not isinstance(item, str):
            return
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            aggregated.append(item)

    for skill in structured.get("skills", []) or []:
        _add(skill)
    for entry in structured.get("experience", []) or []:
        for tech in entry.get("technologies_used", []) or []:
            _add(tech)
    for proj in structured.get("projects", []) or []:
        for tech in proj.get("technologies_used", []) or []:
            _add(tech)

    return aggregated


def ingest_resume(file_path: str) -> dict:
    """
    Full resume ingestion:
      Stage 1 — extract exact text (pymupdf4llm / docx-to-markdown, OCR fallback)
      Stage 2 — structure via LLM (text/dates only, no arithmetic)
      Stage 3 — compute total_years_experience deterministically from the extracted
                date ranges (experience.py), via range union
    then attach candidate_id/document_id/file_name/extraction metadata.
    """
    file_name = os.path.basename(file_path)
    raw_text, warnings, method = extract_text(file_path)
    structured = extract_structured_evidence(raw_text)

    total_years, experience_warnings = compute_total_years(structured.get("experience", []))
    structured["total_years_experience"] = total_years
    structured["skills_all_sources"] = _aggregate_skills(structured)
    warnings = warnings + experience_warnings

    # Warnings are extraction-quality signals for whoever runs this pipeline, not
    # candidate-facing data — they belong in server logs (visible in the terminal
    # running uvicorn), not in an API response a frontend might render as-is. api.py
    # deliberately does NOT forward extraction_warnings from the DB record to the
    # frontend for exactly this reason.
    for w in warnings:
        logger.warning("[%s] %s", file_name, w)

    structured["candidate_id"] = new_id()
    structured["document_id"] = new_id()
    structured["file_name"] = file_name
    structured["extraction_method"] = method
    structured["extraction_warnings"] = warnings
    structured["uploaded_at"] = now_iso()
    return structured


if __name__ == "__main__":
    sample_resume = os.path.join(DATA_DIR, "hirist_kajal_nimje_BA.pdf")

    if os.path.exists(sample_resume):
        print("Ingesting resume...")
        structured_data = ingest_resume(sample_resume)
        print(json.dumps(structured_data, indent=2))
    else:
        print(f"Please ensure your test resume is placed in: {DATA_DIR}")