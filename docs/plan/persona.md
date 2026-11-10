# MCP: persona

## Context
Incident triage is slow: when something breaks you click through the web UI (incidents → lines → other services → metrics) to piece together what happened. Goal: an MCP endpoint so Claude (in Claude Code, for you and teammates) can pull that evidence itself and answer with a cited timeline/root cause, answer ad-hoc log questions, and pre-review new template batches. Humans keep the final say: the AI cannot apply batches or change incident status.

## Persona (approved)
- **Who**: you and teammate operators of a logpipe deployment. They are technical, use Claude Code, and today work through the web UI and hand-written SQL.
- **Problem**: triage means manually connecting incidents, log lines, other services' errors and pipeline health. Template review is a tedious batch chore. The cost is slow MTTR and a growing review backlog.
- **Solution**: a Streamable-HTTP MCP at `/mcp` on the existing FastAPI server, authed by per-person scoped API keys. The client AI does the reasoning. MCP exposes data plus a narrow set of "proposal" writes, each attributed to the key that made it.
- **Out of scope**: applying or discarding batches, changing incident status, admin of settings/rules/keys, storing RCA reports on the server, Slack/Trello posting, autonomous bots, pagination.

