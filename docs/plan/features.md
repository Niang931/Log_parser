# MCP: features

## Features (approved, build in this order)
1. **Scoped API keys**: `scope` is `intake` or `mcp`, and keys without a scope count as `intake`. Intake routes reject `mcp` keys. The web Keys page gets a scope select and a scope badge.
2. **Key-gated `/api` allowlist and `/mcp` mount.**
3. **Log tools.**
4. **Incident tools**, including note-only writes.
5. **Template review tools**, with no apply or discard.
6. **Health tools.**
7. **Drift and warnings.**
8. **Saved queries.**
9. **Prompts**: `investigate_incident(id)` and `review_batch(id)`.
10. **Attribution**: `by: <key name>` on incident notes, drafts and saved queries, shown in the web UI.
11. **README section**: how to create a key, plus the `claude mcp add` command.

