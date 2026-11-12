# Getting started

logpipe runs in four ways. Pick by what you want to see:

| You want to… | Run | Needs |
|---|---|---|
| See the whole pipeline work, live, in 10 seconds | `uv run logpipe demo` | uv |
| Watch the product end to end: agent, review, alerts, Ask | `uv run logpipe showcase` | uv, Go (builds the agent) |
| Ship your own hosts' logs and use the operator UI | `docker compose --profile platform up --build` | Docker |
| Parse and score a directory of log files in batch | `make calibrate train run` | uv |

Not sure? `uv run logpipe` (same as `logpipe start`) asks a few questions, starts the
services it needs (Kafka, ClickHouse) and runs the right command. It also prints that command,
so next time you can run it directly.

1. [Install](install.md): uv, Docker, `.env`, optional extras (Go, Node, a browser for web LLMs).
2. [Run it](run.md): each mode step by step, with what you should see.
3. [Commands](commands.md): every `logpipe` subcommand and `make` target.
