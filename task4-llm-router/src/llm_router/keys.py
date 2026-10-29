"""Tenant API keys: fingerprints and the allow-list.

Raw tenant API keys never reach the database or the logs. Everything downstream of authentication
identifies a tenant by the SHA-256 fingerprint of its key (64 lowercase hex characters). The
allow-list itself is held as fingerprints too, so it can be configured without the raw keys.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Iterable

_FINGERPRINT_RE = re.compile(r"[0-9a-f]{64}")
_MAX_KEY_LENGTH = 512


def fingerprint_api_key(api_key: str) -> str:
    """Return the SHA-256 hex digest of a tenant API key."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def is_fingerprint(value: str) -> bool:
    """True if `value` has the shape of a key fingerprint (and therefore is not a raw key)."""
    return _FINGERPRINT_RE.fullmatch(value) is not None


class TenantKeyring:
    """The set of tenant keys allowed to use the gateway, stored as fingerprints."""

    def __init__(self, fingerprints: Iterable[str]) -> None:
        normalised = {fp.strip().lower() for fp in fingerprints if fp.strip()}
        invalid = [fp for fp in normalised if not is_fingerprint(fp)]
        if invalid:
            raise ValueError(
                f"{len(invalid)} tenant key fingerprint(s) are not 64-character SHA-256 hex digests"
            )
        self._fingerprints = tuple(sorted(normalised))

    def __len__(self) -> int:
        return len(self._fingerprints)

    def authenticate(self, api_key: str) -> str | None:
        """Return the tenant fingerprint for a presented key, or ``None`` if it is not allowed.

        Every stored fingerprint is compared in constant time and the loop never exits early, so the
        response time does not depend on which (or whether a) fingerprint matched.
        """
        candidate = fingerprint_api_key(api_key)
        match = None
        for fingerprint in self._fingerprints:
            if hmac.compare_digest(fingerprint, candidate):
                match = fingerprint
        return match

    def authenticate_header(self, authorization: str | None) -> str | None:
        """Authenticate an ``Authorization: Bearer <key>`` header value."""
        if not authorization:
            return None
        scheme, _, token = authorization.partition(" ")
        token = token.strip()
        if scheme.lower() != "bearer" or not token or len(token) > _MAX_KEY_LENGTH:
            return None
        return self.authenticate(token)
