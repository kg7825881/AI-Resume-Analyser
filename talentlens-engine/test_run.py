"""
test_run.py — end-to-end smoke test for Phase 3.
Run this after wiring in the updated files to confirm ingestion + structuring + scoring
work together with your real Ollama models before moving to Phase 5.

Usage: put a resume file and a JD file in your Kaggle-style data folder
(candidate_1.pdf/.docx and job_description.pdf/.docx), then run:
    python3 test_run.py
"""

import os
import json
from extractor import ingest_resume
from jd_extractor import ingest_jd
from scorer import calculate_job_fit

DATA_DIR = os.path.expanduser("~/kaggle/input/talentlens-batch/")

RESUME_FILE = "candidate_1.pdf"   # change to match your actual sample file
JD_FILE = "job_description.pdf"   # change to match your actual sample file


def main():
    resume_path = os.path.join(DATA_DIR, RESUME_FILE)
    jd_path = os.path.join(DATA_DIR, JD_FILE)

    if not os.path.exists(resume_path) or not os.path.exists(jd_path):
        print(f"Missing test files. Expected:\n  {resume_path}\n  {jd_path}")
        return

    print("Ingesting resume (this calls Ollama — may take a bit)...")
    resume_record = ingest_resume(resume_path)
    print(f"  candidate_name: {resume_record.get('candidate_name')}")
    print(f"  extraction_method: {resume_record['extraction_method']}  warnings: {resume_record['extraction_warnings']}")

    print("\nIngesting JD (this calls Ollama)...")
    jd_record = ingest_jd(jd_path)
    print(f"  role_title: {jd_record.get('role_title')}  ->  role_id: {jd_record['role_id']}")
    print(f"  extraction_method: {jd_record['extraction_method']}  warnings: {jd_record['extraction_warnings']}")

    print("\nScoring (this calls Ollama for embeddings — may take a bit)...")
    result = calculate_job_fit(resume_record, jd_record)

    print("\n=== RESULT ===")
    print(f"Candidate: {resume_record.get('candidate_name')}  |  Role: {jd_record.get('role_title')}")
    print(f"Final Score: {result['final_score']}%")
    print(f"Hard Gate Failed: {result['hard_gate_failed']}  {result['hard_gate_reason']}")
    print("\nCategory breakdown:")
    for cat, detail in result['category_scores'].items():
        line = f"  {cat}: {detail['score']}"
        if 'notes' in detail:
            line += f"  ({detail['notes']})"
        if 'matched' in detail:
            line += f"  matched={detail['matched']}"
        if 'missing' in detail and detail['missing']:
            line += f"  missing={detail['missing']}"
        print(line)

    # Dump full structured output too, useful for spot-checking extraction quality
    print("\nFull resume structuring output:")
    print(json.dumps(resume_record, indent=2))


if __name__ == "__main__":
    main()