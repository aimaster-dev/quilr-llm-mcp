"""Tenant API key fingerprints.

Raw tenant API keys never reach the database or the logs. Everything downstream of authentication
identifies a tenant by the SHA-256 fingerprint of its key (64 lowercase hex characters).
"""

from __future__ import annotations

import hashlib
import re

_FINGERPRINT_RE = re.compile(r"[0-9a-f]{64}")


def fingerprint_api_key(api_key: str) -> str:
    """Return the SHA-256 hex digest of a tenant API key."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def is_fingerprint(value: str) -> bool:
    """True if `value` has the shape of a key fingerprint (and therefore is not a raw key)."""
    return _FINGERPRINT_RE.fullmatch(value) is not None
