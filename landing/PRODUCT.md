# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users
SRE and platform engineers who own noisy service logs (nginx, postgres, Java services) and need to parse, watch and explain them without writing regex or shipping logs to a vendor.

## Product Purpose
logpipe is an open-source log pipeline. It parses known lines on a frozen Drain tree in under a millisecond, learns unknown formats in the background (group by shape, LLM names the fields, publish and backfill), flags anomalies per service with LogBERT, and answers "why did it break?" in plain language citing the exact log lines. The landing page exists to get visitors to the GitHub repo or to run `uv run logpipe showcase`.

## Positioning
Self-learning: unknown lines are kept as PENDING and turned into templates automatically, never dropped. Root-cause answers are read-only and cite the lines that prove them.

## Capabilities and Constraints
- Queue: in-memory or Kafka (started in Docker when needed).
- Store: memory, ClickHouse or MongoDB.
- LLM: offline rules by default (no network), a signed-in web chat (e.g. Gemini), or a hosted API.
- MCP endpoint at `/mcp` for Claude Code and other MCP clients.
- Go agent (`logpipe-agent`) ships logs from hosts.

## Brand Commitments
- Name: logpipe (lowercase).
- Landing page keeps the product UI's identity: near-black ground, indigo sidebar surface, violet action colour, Manrope + JetBrains Mono.

## Evidence on Hand
- Real product screenshots in `public/screenshots/` (Log Explorer, Templates, incident, root cause, ranking, Ask). Show them uncropped at native aspect ratio.
- Third-party logos in `public/assets/icons/` with credits in `icons-credits.json`.
- No customers, testimonials, pricing or benchmarks exist; do not invent them.

## Product Principles
- Prove with the real UI, not with claims.
- Never drop data; show what happens to unknown lines.
- Offline by default; every query is read-only.
