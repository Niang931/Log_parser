"""The /mcp endpoint: tools an AI assistant uses to read logs, triage incidents and pre-review templates.

Tools call the app's own /api routes with the caller's mcp-scoped key, so every validation those
routes do applies, and the middleware's allowlist below is the single limit on what a key may do.
"""
from __future__ import annotations

import re

_ID = r"[^/]+"
MCP_ROUTES = [
    ("GET", r"/api/logs(/histogram|/facets)?"),
    ("GET", r"/api/query/schema"),
    ("POST", r"/api/query"),
    ("GET", rf"/api/incidents(/{_ID})?"),
    ("PUT", rf"/api/incidents/{_ID}"),
    ("GET", r"/api/alerts"),
    ("GET", r"/api/templates"),
    ("GET", rf"/api/review(/{_ID})?"),
    ("POST", r"/api/review/generate"),
    ("POST", rf"/api/review/{_ID}/decide-all"),
    ("PUT", rf"/api/drafts/{_ID}"),
    ("POST", rf"/api/drafts/{_ID}/(check|merge)"),
    ("GET", r"/api/(overview|stats|system|system/alerts|agents|drift|warnings)"),
    ("GET", r"/api/queries"),
    ("POST", r"/api/queries"),
]
_ALLOWED = [(method, re.compile(pattern)) for method, pattern in MCP_ROUTES]


def mcp_may_call(method: str, path: str) -> bool:
    return any(method == m and pattern.fullmatch(path) for m, pattern in _ALLOWED)
