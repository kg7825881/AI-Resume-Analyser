"""
api.py — FastAPI layer for TalentLens.

Endpoints:
  POST /jds/upload         — upload a JD (PDF/DOCX), ingest, store
  GET  /jds                — list all JD roles in the library
  POST /resumes/upload      — upload one or more resumes, ingest, store (partial-failure tolerant)
  POST /analyze              — resolve a role query, score resumes against it, store results
  GET  /results/{role_id}   — fetch the ranked list of scores for a role

Run locally with: uvicorn api:app --reload
"""

import os
import shutil
from typing import List, Optional

import jwt
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr

import db
import auth
from common import new_id, now_iso
from extractor import ingest_resume
from jd_extractor import ingest_jd
from scorer import calculate_job_fit
from role_resolver import resolve_role

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


# --- Auth ---

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> dict:
    if credentials is None:
        raise HTTPException(401, "Missing bearer token. Log in via POST /auth/login.")
    try:
        payload = auth.decode_access_token(credentials.credentials)
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Session expired, please log in again.")
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid session token.")

    user = db.get_user_by_id(payload["sub"])
    if not user:
        raise HTTPException(401, "User no longer exists.")
    return {"user_id": user["user_id"], "email": user["email"], "name": user["name"]}


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


@app.post("/auth/register")
def register(req: RegisterRequest):
    if len(req.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")
    if db.get_user_by_email(req.email):
        raise HTTPException(409, "An account with this email already exists.")

    user_id = new_id()
    db.insert_user(user_id, req.email, req.name, auth.hash_password(req.password), now_iso())
    token = auth.create_access_token(user_id, req.email.strip().lower())
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"user_id": user_id, "email": req.email.strip().lower(), "name": req.name},
    }


@app.post("/auth/login")
def login(req: LoginRequest):
    user = db.get_user_by_email(req.email)
    if not user or not auth.verify_password(req.password, user["password_hash"]):
        raise HTTPException(401, "Incorrect email or password.")

    token = auth.create_access_token(user["user_id"], user["email"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"user_id": user["user_id"], "email": user["email"], "name": user["name"]},
    }


@app.get("/auth/me")
def me(current_user: dict = Depends(get_current_user)):
    return current_user


def _save_upload(upload_file: UploadFile, target_dir: str) -> str:
    dest_path = os.path.join(target_dir, upload_file.filename)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    return dest_path


# --- JDs ---

@app.post("/jds/upload")
def upload_jd(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
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
def list_jds(current_user: dict = Depends(get_current_user)):
    return db.get_all_roles()


# --- Resumes ---

@app.post("/resumes/upload")
def upload_resumes(files: List[UploadFile] = File(...), current_user: dict = Depends(get_current_user)):
    """Batch upload — individual file failures don't stop the rest (per Phase 2 design)."""
    results = []
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in (".pdf", ".docx"):
            results.append({"file_name": file.filename, "status": "failed", "error": f"Unsupported file type: {ext}"})
            continue

        try:
            path = _save_upload(file, STORAGE_RESUMES)
            record = ingest_resume(path)
            db.insert_resume(record)
            results.append({
                "file_name": file.filename,
                "status": "ok",
                "candidate_id": record["candidate_id"],
                "candidate_name": record.get("candidate_name", ""),
                "extraction_warnings": record["extraction_warnings"],
            })
        except Exception as e:
            results.append({"file_name": file.filename, "status": "failed", "error": str(e)})

    return {"results": results}


# --- Analyze ---

class AnalyzeRequest(BaseModel):
    role_query: str
    candidate_ids: Optional[List[str]] = None  # if omitted, scores ALL stored resumes


@app.post("/analyze")
def analyze(req: AnalyzeRequest, current_user: dict = Depends(get_current_user)):
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

    scored_results = []
    for resume in resumes:
        result = calculate_job_fit(resume, jd_data)
        db.insert_score(resume["candidate_id"], role["role_id"], result)
        scored_results.append({
            "candidate_id": resume["candidate_id"],
            "candidate_name": resume.get("candidate_name", ""),
            "final_score": result["final_score"],
            "hard_gate_failed": result["hard_gate_failed"],
            "hard_gate_reason": result["hard_gate_reason"],
        })

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
def get_results(role_id: str, current_user: dict = Depends(get_current_user)):
    scores = db.get_scores_by_role(role_id)
    if not scores:
        raise HTTPException(404, f"No scores found for role_id '{role_id}'. Run /analyze first.")

    ranked = [s for s in scores if not s["hard_gate_failed"]]
    excluded = [s for s in scores if s["hard_gate_failed"]]
    return {"role_id": role_id, "ranked": ranked, "excluded_hard_gate_failed": excluded}