"""
api.py — FastAPI layer for TalentLens.

Endpoints:
  POST /jds/upload         — upload a JD (PDF/DOCX), ingest, store
  GET  /jds                — list all JD roles in the library
  POST /resumes/upload      — upload one or more resumes, ingest, store (partial-failure tolerant)
  POST /analyze              — resolve a role query, score resumes against it, store results
  GET  /results/{role_id}   — fetch the ranked list of scores for a role

No auth — built for a single internal HR user against the Next.js frontend.

Run locally with: uvicorn api:app --reload
"""

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import db
from common import new_id
from extractor import ingest_resume
from jd_extractor import ingest_jd
from scorer import calculate_job_fit
from role_resolver import resolve_role

# Ollama serves requests over HTTP, so ingestion/scoring calls are I/O-bound — a thread pool
# lets several run concurrently even though each individual call is a normal blocking function.
# This does NOT by itself guarantee the Ollama *server* processes them in parallel — check
# OLLAMA_NUM_PARALLEL (and that you have RAM headroom for it) if wall-clock time doesn't improve.
MAX_WORKERS = int(os.environ.get("TALENTLENS_MAX_WORKERS", "4"))

app = FastAPI(title="TalentLens Resume Analyzer")

# Allows the Next.js dev server (and same-origin prod builds you configure later) to call this API.
FRONTEND_ORIGINS = os.environ.get("TALENTLENS_FRONTEND_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_JDS = "storage/jds"
STORAGE_RESUMES = "storage/resumes"
os.makedirs(STORAGE_JDS, exist_ok=True)
os.makedirs(STORAGE_RESUMES, exist_ok=True)


@app.on_event("startup")
def on_startup():
    db.init_db()


def _save_upload(upload_file: UploadFile, target_dir: str) -> str:
    dest_path = os.path.join(target_dir, upload_file.filename)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    return dest_path


# --- JDs ---

@app.post("/jds/upload")
def upload_jd(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".pdf", ".docx"):
        raise HTTPException(400, f"Unsupported file type: {ext}. Only .pdf and .docx are supported.")

    path = _save_upload(file, STORAGE_JDS)
    try:
        record = ingest_jd(path)
    except Exception as e:
        raise HTTPException(500, f"Failed to process JD: {str(e)}")

    db.insert_jd(record)
    return {
        "role_id": record["role_id"],
        "role_title": record.get("role_title", ""),
        "document_id": record["document_id"],
        "extraction_method": record["extraction_method"],
        "extraction_warnings": record["extraction_warnings"],
    }


@app.get("/jds")
def list_jds():
    return db.get_all_roles()


@app.get("/jds/{role_id}")
def get_jd(role_id: str):
    """Full extracted JD record (mandatory_skills, responsibilities, etc.) — used by the frontend
    to display the raw structured JD data, not just the summary from GET /jds."""
    jd = db.get_jd_by_role_id(role_id)
    if not jd:
        raise HTTPException(404, f"No JD found for role_id '{role_id}'.")
    return jd


# --- Resumes ---

@app.post("/resumes/upload")
def upload_resumes(files: List[UploadFile] = File(...)):
    """Batch upload — individual file failures don't stop the rest (per Phase 2 design).
    Streams results back as newline-delimited JSON (NDJSON) so the frontend can add each
    candidate to the list the moment ITS ingestion finishes, instead of waiting for the
    entire batch. Files are still ingested concurrently — this changes *when the client
    finds out*, not the concurrency itself (that was already fixed).

    Response shape: one JSON object per line —
      {"type": "meta", "total": N}                                   — sent first
      {"type": "result", "file_name": ..., "status": "ok"|"failed", ...}  — one per file,
                                                                            in COMPLETION order,
                                                                            not upload order.
    """

    # Validate + save synchronously first (fast, and keeps UploadFile handling on the main
    # thread — file.read() isn't safe to call concurrently from a worker thread).
    to_process = []  # (file_name, saved_path)
    immediate_failures = []
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in (".pdf", ".docx"):
            immediate_failures.append({
                "type": "result", "file_name": file.filename,
                "status": "failed", "error": f"Unsupported file type: {ext}",
            })
            continue
        path = _save_upload(file, STORAGE_RESUMES)
        to_process.append((file.filename, path))

    total = len(files)

    def _process(item):
        file_name, path = item
        try:
            record = ingest_resume(path)
            db.insert_resume(record)
            return {
                "type": "result",
                "file_name": file_name,
                "status": "ok",
                "candidate_id": record["candidate_id"],
                "candidate_name": record.get("candidate_name", ""),
                "extraction_warnings": record["extraction_warnings"],
            }
        except Exception as e:
            return {"type": "result", "file_name": file_name, "status": "failed", "error": str(e)}

    def stream():
        yield json.dumps({"type": "meta", "total": total}) + "\n"

        for failure in immediate_failures:
            yield json.dumps(failure) + "\n"

        if to_process:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = [pool.submit(_process, item) for item in to_process]
                # as_completed yields whichever finishes first — that's the whole point:
                # a fast resume shouldn't wait behind a slow one before reaching the client.
                for future in as_completed(futures):
                    yield json.dumps(future.result()) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


# --- Analyze ---

class AnalyzeRequest(BaseModel):
    role_query: str
    candidate_ids: Optional[List[str]] = None  # if omitted, scores ALL stored resumes


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    roles = db.get_all_roles()
    resolution = resolve_role(req.role_query, roles)

    if resolution["status"] == "no_match":
        raise HTTPException(404, f"No JD found matching '{req.role_query}'. Upload a JD for this role first.")

    if resolution["status"] == "ambiguous":
        return {
            "status": "ambiguous",
            "message": f"'{req.role_query}' matches multiple roles — please specify which one.",
            "candidates": [{"role_id": c["role_id"], "role_title": c["role_title"]} for c in resolution["candidates"]],
        }

    role = resolution["role"]
    jd_data = db.get_jd_by_role_id(role["role_id"])
    resumes = db.get_resumes(req.candidate_ids)

    if not resumes:
        raise HTTPException(404, "No resumes found to analyze (upload resumes first, or check candidate_ids).")

    run_id = new_id()  # ties every score from this /analyze call together as one run

    def _score(resume):
        result = calculate_job_fit(resume, jd_data)
        db.insert_score(resume["candidate_id"], role["role_id"], run_id, result)
        return {
            "candidate_id": resume["candidate_id"],
            "candidate_name": resume.get("candidate_name", ""),
            "final_score": result["final_score"],
            "hard_gate_failed": result["hard_gate_failed"],
            "hard_gate_reason": result["hard_gate_reason"],
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        scored_results = list(pool.map(_score, resumes))

    scored_results.sort(key=lambda r: r["final_score"], reverse=True)
    ranked = [r for r in scored_results if not r["hard_gate_failed"]]
    excluded = [r for r in scored_results if r["hard_gate_failed"]]

    return {
        "status": "scored",
        "role_id": role["role_id"],
        "role_title": role["role_title"],
        "ranked": ranked,
        "excluded_hard_gate_failed": excluded,
    }


# --- Results ---

@app.get("/results/{role_id}")
def get_results(role_id: str):
    scores = db.get_scores_by_role(role_id)
    if not scores:
        raise HTTPException(404, f"No scores found for role_id '{role_id}'. Run /analyze first.")

    ranked = [s for s in scores if not s["hard_gate_failed"]]
    excluded = [s for s in scores if s["hard_gate_failed"]]
    return {"role_id": role_id, "ranked": ranked, "excluded_hard_gate_failed": excluded}