# Datadog in/out loop: features

## Features (all approved, built in this order)
| # | Feature | Solves |
|---|---|---|
| F0 | Fix the branch: add `DatadogSettings` | nothing works without it |
| F1 | DD Logs Search API → `ServerCore.ingest()` (poller) | gets their logs in, no re-instrumentation |
| F2 | New logpipe incident → DD Event, with deep links | detection shows up where the SRE looks |
| F3 | Auto-run RCA (no approval) → follow-up DD Event | faster triage |
| F4 | Approved templates → DD Logs Pipeline (grok parser) | removes hand-written grok |
| F5 | DD monitor webhook → logpipe incident + auto RCA → DD Event | closes the loop for DD-originated alerts |
| F6 | `logpipe showcase --datadog` + README section | the judges see it end to end |

