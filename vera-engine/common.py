"""
common.py — shared utilities for generating document/candidate/role IDs
and timestamps, per the Phase 1 schema.
"""

import re
import uuid
from datetime import datetime, timezone


def new_id() -> str:
    """Generates a new unique id (used for document_id / candidate_id)."""
    return str(uuid.uuid4())


def slugify(text: str) -> str:
    """
    Turns a role title into a stable role_id slug.
    e.g. "AI Engineer" -> "ai-engineer", "AI/ML Engineer II" -> "ai-ml-engineer-ii"
    """
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "unknown-role"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()