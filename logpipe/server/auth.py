"""The admin password that guards the operator UI and ``/api``.

One password, kept as a scrypt hash in the control store (so it survives restarts with the
ClickHouse store). A login hands out a signed session cookie; the signing key includes the hash,
so changing the password signs everyone out. Agents never use this: they send API keys.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time

from ..settings import get_settings

COOKIE = "logpipe_session"
_N, _R, _P = 2**14, 8, 1


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P)
    return f"scrypt${_N}${_R}${_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, digest = stored.split("$")
        got = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got.hex(), digest)


class Auth:
    def __init__(self, control):
        self.control = control
        self._lock = threading.Lock()
        self._failures: dict[str, list[float]] = {}

    @property
    def _doc(self) -> dict:
        return self.control.get("auth", "admin") or {}

    @property
    def configured(self) -> bool:
        return bool(self._doc.get("hash"))

    def set_password(self, password: str) -> None:
        if len(password) < (n := get_settings().server.min_password_length):
            raise ValueError(f"the password needs at least {n} characters")
        self.control.put("auth", "admin", {"hash": hash_password(password),
                                           "secret": secrets.token_hex(32), "changed_at": time.time()})

    def ensure(self, from_env: str | None) -> str | None:
        """At startup: ``from_env`` (LOGPIPE_ADMIN_PASSWORD) always wins, so a forgotten password
        is reset by setting it; with neither a stored nor an env password, one is generated and
        returned so the caller can print it once."""
        if from_env:
            if not (self.configured and verify_password(from_env, self._doc["hash"])):
                self.set_password(from_env)
            return None
        if self.configured:
            return None
        generated = secrets.token_urlsafe(12)
        self.set_password(generated)
        return generated

    def blocked_for(self, client: str) -> float:
        """Seconds until ``client`` may try again (0 when it may)."""
        now = time.monotonic()
        with self._lock:
            cfg = get_settings().server
            window = cfg.login_failure_window_minutes * 60
            recent = [t for t in self._failures.get(client, []) if now - t < window]
            self._failures[client] = recent
            return window - (now - recent[0]) if len(recent) >= cfg.login_max_failures else 0

    def login(self, client: str, password: str) -> str | None:
        """A session token, or None for a wrong password (counted against ``client``)."""
        doc = self._doc
        if doc.get("hash") and verify_password(password, doc["hash"]):
            with self._lock:
                self._failures.pop(client, None)
            return self._sign(doc, int(time.time()) + get_settings().server.session_seconds)
        with self._lock:
            self._failures.setdefault(client, []).append(time.monotonic())
        return None

    @staticmethod
    def _sig(doc: dict, expires: int) -> str:
        key = (doc["secret"] + doc["hash"]).encode()
        return hmac.new(key, str(expires).encode(), hashlib.sha256).hexdigest()

    def _sign(self, doc: dict, expires: int) -> str:
        return f"{expires}.{self._sig(doc, expires)}"

    def valid(self, token: str | None) -> bool:
        doc = self._doc
        if not token or not doc.get("hash"):
            return False
        expires, _, sig = token.partition(".")
        if not expires.isdigit() or int(expires) < time.time():
            return False
        return hmac.compare_digest(sig, self._sig(doc, int(expires)))
