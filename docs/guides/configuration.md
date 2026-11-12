# Configuration

All settings live in `logpipe/settings.py`: one `pydantic-settings` class per environment
prefix, read once through `get_settings()`. `.env` is loaded by `make` (and `uv run --env-file
.env`). Override any field by **prefix + field name in upper case**, e.g.
`LOGPIPE_RCA_BEFORE_MINUTES=10`. Settings a person changes in the UI are separate:
`RuntimeSettings`, stored in the control plane.

## Environment

=== "App & LLM (no prefix)"

    | Variable | Default | |
    |---|---|---|
    | `LOGS_PATH` | `logs` | batch input directory |
    | `REGISTRY_PATH` | `artifacts/registry.json` | |
    | `MODEL_PATH` | `artifacts/logbert` | batch LogBERT model |
    | `STORE` | `memory` | `memory` · `clickhouse` · `mongodb` |
    | `QUEUE` | `memory` | `memory` · `kafka` |
    | `LLM_BACKEND` | `offline` | `offline` · `hf` · `llm:<provider>[:<model>]` · `llm:web:<site>` |
    | `WEB_BROWSER` | `camoufox` | `camoufox` · `chromium` |
    | `HF_BASE` / `HF_ADAPTER` | DeepSeek-R1-Distill-Llama-8B / `MThien213/deepparse-iot-r1-8b` | `hf` backend |

=== "Stores & queue"

    | Prefix | Fields (defaults) |
    |---|---|
    | `CLICKHOUSE_` | `HOST` 127.0.0.1, `PORT` 8123, `USER` default, `PASSWORD`, `DB` default, `READONLY_USER`, `READONLY_PASSWORD` |
    | `MONGO_` | `URI` mongodb://127.0.0.1:27017, `DB` logpipe, `READONLY_URI` |
    | `KAFKA_` | `BOOTSTRAP` 127.0.0.1:9092, `TOPIC` logpipe.unresolved, `GROUP` logpipe-workers, `DLQ_TOPIC` (`<topic>.dlq`), `KEY_BY` shape \| device |

=== "Server (`LOGPIPE_`)"

    | Field | Default | |
    |---|---|---|
    | `LOG_FORMAT` | text | `json` for one object per line in `server.log` |
    | `BOOTSTRAP_KEY` | — | an API key accepted at startup (compose demo) |
    | `ADMIN_PASSWORD` | — | always replaces the stored password |
    | `SESSION_DAYS` | 7 | |
    | `LOGIN_MAX_FAILURES` / `LOGIN_FAILURE_WINDOW_MINUTES` | 5 / 15 | lockout per address |
    | `MAX_BODY_MB` | 64 | decompressed intake batch |
    | `MIN_TRAIN_WINDOWS` / `MIN_TRAIN_STEPS` | 8 / 1500 | LogBERT waits for data |
    | `MIN_DRAFT_LINES` | 50 | |
    | `INCIDENT_GAP_MINUTES` | 5 | |
    | `MAX_QUERY_ROWS` / `MAX_BUILDER_ROWS` | 500 / 500 | |
    | `DRIFT_KEEP` / `DRIFT_MIN_LIVE` | 2000 / 200 | |

=== "Ask (`LOGPIPE_ASK_`)"

    | Field | Default | |
    |---|---|---|
    | `MODEL` | sentence-transformers/all-MiniLM-L6-v2 | text → template |
    | `PLAN_TTL_MINUTES` | 15 | a waiting plan expires |
    | `MAX_ROUNDS` | 7 | discovery, evidence, one widening, four follow-ups |
    | `DEFAULT_MINUTES` / `MAX_SPAN_DAYS` | 60 / 90 | |
    | `MAX_ROWS` / `ROW_CAP` / `AGG_CAP` | 500 / 2000 / 20000 | |
    | `BUDGET_TOKENS` | 6000 | evidence bundle sent to the LLM |
    | `SHUTDOWN_MIN_SCORE` | 0.45 | |
    | `LLM_TIMEOUT_SECONDS` | 240 | |

=== "Root cause (`LOGPIPE_RCA_`)"

    | Field | Default | |
    |---|---|---|
    | `BEFORE_MINUTES` / `AFTER_MINUTES` | 30 / 2 | window around the anchor |
    | `WIDE_BEFORE_MINUTES` / `WIDE_AFTER_MINUTES` | 120 / 5 | widened once |
    | `BASELINE_HOURS` | 24 | |
    | `SILENCE_MIN_RATE` / `SILENCE_MINUTES` | 2.0 / 3 | |
    | `ERROR_WEIGHT` / `WARN_WEIGHT` / `OTHER_WEIGHT` | 0.8 / 0.5 / 0.1 | anomaly factor by level |
    | `SHUTDOWN_WEIGHT` / `RELATED_WEIGHT` | 0.9 / 0.5 | |
    | `PROXIMITY_SECONDS` | 600 | `exp(−Δt / this)` |
    | `PRECURSOR_SECONDS` | 120 | |
    | `STRONG` / `MIN_STRONG` | 0.15 / 3 | when to widen |
    | `MAX_CANDIDATES` / `MAX_LINES` / `LINE_CHARS` | 15 / 40 / 300 | |
    | `FOLLOW_UPS` / `MOVES_PER_ROUND` / `MOVE_MAX_HOURS` | 4 / 3 / 24 | |
    | `HINT_MIN_RATIO` / `HINT_MIN_SHARE` / `MAX_HINTS` | 3.0 / 0.2 / 5 | value hints |

## In the UI (`RuntimeSettings`)

| Setting | Default | Range |
|---|---|---|
| `sample_cap`: lines sampled per source for drafts | 5000 | 100–200,000 |
| `retrain_days`: model age before an automatic retrain | 7 | 0.01–3650 |
| `draft_lines`: lines before a new source is drafted | 500 | 50–200,000 |
| `draft_after_minutes`: or after this long with ≥ 50 lines | 60 | 1–100,000 |
| `ask_require_approval`: every Ask read waits for a person | true | — |

## What stays a constant

Regex tables (level rules, the shutdown pattern), prompts, column maps and the allowed Mongo
stages are part of the code's contract, not settings. Change them in code, with a test.
