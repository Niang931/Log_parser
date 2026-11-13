# Datadog in/out loop: persona

## Context
This is for a hackathon with a Datadog track. The deadline is 2–3 days, and success means a live demo against your Datadog org (API + APP keys are available).
`feat/datadog` only sends logpipe's **own health** to Datadog so far: DogStatsD stage metrics, a JSON-line log file the Agent tails, and LLM Observability spans (`logpipe/observability.py`), plus the `--datadog` flag, the agent compose profile, `docs/datadog/*.json`, and `make demo-datadog`.
**The branch is broken right now**: `observability.py` imports `DatadogSettings` from `logpipe/settings.py`, but that class was never committed, so `import logpipe.observability` raises ImportError (verified).
The goal is to make logpipe useful to Datadog customers: it reads their DD logs, finds anomalies, and writes the anomalies, the AI root-cause analysis (RCA) and grok parsers back into Datadog.

## 1. Persona (approved)
- **Who**: an SRE or on-call engineer who already pays for Datadog Logs, with many services and millions of unstructured lines.
- **Problem**: new failure patterns don't trip any threshold monitor until users complain, and grok parsers are written by hand and fall behind. The cost is longer outages and engineers spending time on parsers.
- **Solution**: logpipe sits next to Datadog. It pulls their logs, learns templates on its own, LogBERT flags unusual sequences, and every incident and its AI RCA appears in Datadog as Events. Learned templates are exported as DD grok pipelines.
- **Out of scope**: replacing DD storage or UI, multi-org support, production-grade retries and backoff, a UI settings page for DD, APM tracing, Go agent → DD forwarding, provisioning dashboards through the API, Kafka DSM/ClickHouse DBM.

