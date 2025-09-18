# app/core/token_utils.py

import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Tuple

def generate_refresh_token_string(length: int = 64) -> str:
    # URL-safe opaque token (high entropy)
    return secrets.token_urlsafe(length)

def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()

def refresh_token_expiry(days: int = 30) -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=days)
