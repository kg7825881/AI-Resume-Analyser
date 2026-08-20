"""
db.py — lightweight persistence layer using SQLite (stdlib, zero setup).
Stores JDs, resumes, and score records per the Phase 1 schema.

Swappable later: if you outgrow SQLite (concurrent HR users, larger scale),
migrate to Postgres by replacing this module with a SQLAlchemy version —
the function signatures below (insert_jd, insert_resume, etc.) can stay
the same so api.py doesn't need to change.
"""

import sqlite3
import json
import os
from contextlib import contextmanager

DB_PATH = os.environ.get("TALENTLENS_DB_PATH", "VERA.db")


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jds (
                role_id TEXT PRIMARY KEY,
                document_id TEXT,
                role_title TEXT,
                department TEXT,
                mandatory_skills TEXT,
                preferred_technical_skills TEXT,
                soft_preferred_skills TEXT,
                min_years_experience REAL,
                education_requirements TEXT,
                relevant_certifications TEXT,
                responsibilities TEXT,
                file_name TEXT,
                extraction_method TEXT,
                extraction_warnings TEXT,
                uploaded_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS resumes (
                candidate_id TEXT PRIMARY KEY,
                document_id TEXT,
                candidate_name TEXT,
                skills TEXT,
                total_years_experience REAL,
                experience TEXT,
                education TEXT,
                certifications TEXT,
                projects TEXT,
                file_name TEXT,
                extraction_method TEXT,
                extraction_warnings TEXT,
                uploaded_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT,
                role_id TEXT,
                run_id TEXT,
                category_scores TEXT,
                evidence TEXT,
                final_score REAL,
                hard_gate_failed INTEGER,
                hard_gate_reason TEXT,
                scored_at TEXT
            )
        """)

        # Migration: a talentlens.db created before this change has a "scores" table
        # WITHOUT the evidence column — CREATE TABLE IF NOT EXISTS only applies to a
        # brand-new table, it does nothing to an existing one with a different shape.
        # This adds the column in place so existing databases don't need a manual reset.
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(scores)").fetchall()}
        if "evidence" not in existing_cols:
            conn.execute("ALTER TABLE scores ADD COLUMN evidence TEXT")


@contextmanager
def _connect():
    # timeout: with parallel resume ingestion/scoring, multiple threads may write around
    # the same time — without a timeout, SQLite raises "database is locked" immediately
    # instead of waiting briefly for the other writer to finish.
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _dumps(value) -> str:
    return json.dumps(value if value is not None else [])


def insert_jd(record: dict):
    with _connect() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO jds
            (role_id, document_id, role_title, department, mandatory_skills, preferred_technical_skills,
             soft_preferred_skills, min_years_experience, education_requirements, relevant_certifications,
             responsibilities, file_name, extraction_method, extraction_warnings, uploaded_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            record["role_id"], record["document_id"], record.get("role_title", ""),
            record.get("department", ""), _dumps(record.get("mandatory_skills")),
            _dumps(record.get("preferred_technical_skills")), _dumps(record.get("soft_preferred_skills")),
            record.get("min_years_experience", 0), _dumps(record.get("education_requirements")),
            _dumps(record.get("relevant_certifications")), _dumps(record.get("responsibilities")),
            record.get("file_name", ""), record.get("extraction_method", ""),
            _dumps(record.get("extraction_warnings")), record.get("uploaded_at", ""),
        ))


def insert_resume(record: dict):
    with _connect() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO resumes
            (candidate_id, document_id, candidate_name, skills, total_years_experience, experience,
             education, certifications, projects, file_name, extraction_method, extraction_warnings, uploaded_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            record["candidate_id"], record["document_id"], record.get("candidate_name", ""),
            _dumps(record.get("skills")), record.get("total_years_experience", 0),
            _dumps(record.get("experience")), _dumps(record.get("education")),
            _dumps(record.get("certifications")), _dumps(record.get("projects")),
            record.get("file_name", ""), record.get("extraction_method", ""),
            _dumps(record.get("extraction_warnings")), record.get("uploaded_at", ""),
        ))


def insert_score(candidate_id: str, role_id: str, run_id: str, result: dict):
    with _connect() as conn:
        conn.execute("""
            INSERT INTO scores (candidate_id, role_id, run_id, category_scores, evidence, final_score,
                                 hard_gate_failed, hard_gate_reason, scored_at)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            candidate_id, role_id, run_id, json.dumps(result["category_scores"]),
            json.dumps(result.get("evidence", {})), result["final_score"],
            int(result["hard_gate_failed"]), result["hard_gate_reason"], result["scored_at"],
        ))


def get_all_roles() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT document_id, role_id, role_title, department, uploaded_at FROM jds").fetchall()
        return [dict(r) for r in rows]


def get_jd_by_role_id(role_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jds WHERE role_id = ? ORDER BY uploaded_at DESC LIMIT 1", (role_id,)).fetchone()
        if not row:
            return None
        return _row_to_jd_dict(row)


def _row_to_jd_dict(row) -> dict:
    d = dict(row)
    for field in ["mandatory_skills", "preferred_technical_skills", "soft_preferred_skills",
                  "education_requirements", "relevant_certifications", "responsibilities", "extraction_warnings"]:
        d[field] = json.loads(d[field]) if d[field] else []
    return d


def get_resumes(candidate_ids: list[str] = None) -> list[dict]:
    with _connect() as conn:
        if candidate_ids:
            placeholders = ",".join("?" for _ in candidate_ids)
            rows = conn.execute(f"SELECT * FROM resumes WHERE candidate_id IN ({placeholders})", candidate_ids).fetchall()
        else:
            rows = conn.execute("SELECT * FROM resumes").fetchall()
        return [_row_to_resume_dict(r) for r in rows]


def _row_to_resume_dict(row) -> dict:
    d = dict(row)
    for field in ["skills", "experience", "education", "certifications", "projects", "extraction_warnings"]:
        d[field] = json.loads(d[field]) if d[field] else []
    return d


def get_scores_by_role(role_id: str) -> list[dict]:
    with _connect() as conn:
        # Only the most recent /analyze run for this role — otherwise re-running analysis
        # (even with an overlapping or different candidate set) would pile every historical
        # score row on top of the previous ones, showing candidates from old runs too.
        latest_run = conn.execute(
            "SELECT run_id FROM scores WHERE role_id = ? ORDER BY scored_at DESC LIMIT 1", (role_id,)
        ).fetchone()
        if not latest_run:
            return []
        run_id = latest_run["run_id"]

        rows = conn.execute("""
            SELECT s.*, r.candidate_name, r.file_name, r.skills, r.total_years_experience,
                   r.experience, r.education, r.certifications, r.projects
            FROM scores s JOIN resumes r ON s.candidate_id = r.candidate_id
            WHERE s.role_id = ? AND s.run_id = ?
            ORDER BY s.final_score DESC
        """, (role_id, run_id)).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["category_scores"] = json.loads(d["category_scores"])
            # Rows scored before the evidence column existed will have NULL here —
            # fall back to an empty dict instead of crashing on json.loads(None).
            d["evidence"] = json.loads(d["evidence"]) if d.get("evidence") else {}
            d["hard_gate_failed"] = bool(d["hard_gate_failed"])
            # Raw extracted resume fields — needed for the candidate comparison table,
            # not just the computed scores.
            for field in ["skills", "experience", "education", "certifications", "projects"]:
                d[field] = json.loads(d[field]) if d[field] else []
            results.append(d)
        return results