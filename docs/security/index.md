# Security

Log lines are attacker-controlled text: a device, a request header or a user name can put any
string into them. logpipe treats every line, and everything a model says about a line, as
untrusted. The pages in this section describe each attack class; the table says where it could
reach logpipe and what stops it.

| Threat | Where it could hit | Defence | Test |
|---|---|---|---|
| [SQL injection](attacks/sql.md) | Query page, Ask, MCP `run_query` | QuerySpec → builder only (no LLM SQL); `sql_guard` blocks writes and table functions; read-only DB users | `test_ask_*`, `test_server*` |
| [Prompt injection](attacks/prompt.md) | key naming, alias checks, Ask synthesis, MCP clients reading lines | output validation, citation check, opposite-state veto, length caps, proposals-only MCP | `test_prompt_injection.py` |
| [ReDoS](attacks/redos.md) | masks, parsing rules | `safe_masks` (length cap, no catastrophic patterns); `regex("…")` in rules refuses nested quantifiers | validate and rule tests |
| [Key / field-name injection](attacks/keys.md) | extracted field names reaching SQL, JSON paths, the UI | `clean_keys`, `KEY_RE` | validate tests |
| [DOM injection](attacks/domi.md) | log text and LLM text in the UI | React escapes all text; the UI never sets raw HTML (no `dangerouslySetInnerHTML`) | — |

## Access model

| Who | Credential | Can |
|---|---|---|
| Operator | admin password (scrypt, lockout after 5 failures / 15 min) | everything in the UI and `/api` |
| Agent | API key, scope `intake` (stored as sha256) | `/v1/logs`, `/intake/v1/*` only |
| AI assistant | API key, scope `mcp` | the `MCP_ROUTES` allowlist: reads, notes, draft proposals, `ask` |

CI runs pip-audit, npm audit, gitleaks and Trivy on every PR ([CI](../ci.md#security-job)).
