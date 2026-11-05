# Persona

## Context
logpipe today only ingests synthetic IoT fleets or replayed files (`logpipe run --logs`, `logpipe demo`).
To be a real product it needs a host-side collector that ships live logs, a backend that takes them
in, and an operator UI. The existing stdlib demo (`logpipe/demo/server.py` + `index.html`) stays as
the developer view at `/demo`; the new React UI is the operator view.

## Persona
**Who:** Ops/SREs running Linux servers, and IoT gateway operators (ARM, constrained). Comfortable
with YAML, systemd and docker; not interested in ML internals.
**Problem:** logs are scattered across hosts and formats; multiline stack traces get split into
noise; nobody knows which host ships what. Today logpipe can't even receive live logs.
**Cost of not solving:** template/anomaly detection only works on replayed files, so it never
catches anything in production.
**Solution (user-specified parts marked *):** a Go agent* with Datadog-style YAML* (agent.yaml +
conf.d), CLI* to track files / edit config, multiline*, docker image*; a FastAPI `logpipe server`
feeding the existing `Pipeline`; a React UI* (intuitive, operator-facing) with a fleet view and
remote config push*.
**Out of scope:** UI login/users (behind proxy), TLS/mTLS management, HA, Prometheus exporter,
deb/rpm packages, Kubernetes DaemonSet, porting the demo page to React.
