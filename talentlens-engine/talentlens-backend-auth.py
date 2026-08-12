"""
auth.py — minimal HR login layer for TalentLens.

Deliberately simple for the POC:
  - Passwords hashed with stdlib PBKDF2 (no bcrypt/native-dep install needed).
  - Sessions are stateless JWTs (no server-side session store).
  - Registration is open (POST /auth/register). Fine for a POC with a handful
    of HR users; before a real rollout this should move behind an invite/admin
    flow so anyone with API access can't create themselves an HR account.

Swappable later: if you outgrow this, the natural upgrade path is the same
ZITADEL/OIDC setup already used in AuthWall — this module exists so the
frontend has something real to log into today.
"""

import base64
import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone

import jwt

SECRET_KEY = os.environ.get("TALENTLENS_SECRET_KEY", "dev-secret-change-me-before-deploying")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 12

PBKDF2_ITERATIONS = 260_000


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return base64.b64encode(salt + derived).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        raw = base64.b64decode(hashed.encode("utf-8"))
    except Exception:
        return False
    salt, stored_derived = raw[:16], raw[16:]
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return hmac.compare_digest(derived, stored_derived)


def create_access_token(user_id: str, email: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "iat": now,
        "exp": now + timedelta(hours=TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Raises jwt.PyJWTError (expired, invalid signature, malformed, etc.) on failure."""
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
