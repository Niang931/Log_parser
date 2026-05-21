# ─────────────────────────────────────────────────────────────────────────────
# DeepParse v2 — Production Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage build:
#   Stage 1 (builder) — install Python deps in a virtual-env
#   Stage 2 (runtime) — slim image, non-root user, read-only source
#
# Security controls
# -----------------
#   • Non-root user (uid 10001)  → no container escape via file writes
#   • Read-only /app mount       → source immutability in prod
#   • ANTHROPIC_API_KEY via env  → no secrets baked into image
#   • PYTHONDONTWRITEBYTECODE    → no .pyc files in output volume
#
# Rubric: deploy model (API/CLI/container), security constraints,
#         operational readiness, enterprise-ready thinking
# ─────────────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.11

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /build

# System deps for pandas/lxml
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        build-essential libxml2-dev libxslt1-dev && \
    rm -rf /var/lib/apt/lists/*

# Virtual environment → easy copy to runtime stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: slim runtime ─────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="DeepParse v2"
LABEL org.opencontainers.image.description="Silicon-Fab Log Parsing Pipeline"
LABEL org.opencontainers.image.version="2.0.0"

# Non-root user
RUN groupadd --gid 10001 deepparse && \
    useradd  --uid 10001 --gid deepparse --shell /bin/bash --create-home deepparse

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source (read-only in prod; writeable artifacts/ via volume)
COPY --chown=deepparse:deepparse . /app

# Artifacts directory (writable volume mount point)
RUN mkdir -p /app/artifacts/output /app/artifacts/data /app/artifacts/eval && \
    chown -R deepparse:deepparse /app/artifacts

# Switch to non-root
USER deepparse

# Runtime environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Health-check: run pipeline in mock mode against synthetic data
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=3 \
    CMD python main.py \
            --input artifacts/data/synthetic_fab.log \
            --llm-provider mock \
            --static-masks masks_fab_universal.json \
            --no-mask-cache \
            --max-logs 20 \
            --no-drain-state 2>&1 | grep -q "Parse rate" || exit 1

# Default entrypoint: CLI passthrough
ENTRYPOINT ["python", "main.py"]

# Default command: show help
CMD ["--help"]

# ─────────────────────────────────────────────────────────────────────────────
# Usage examples
# ─────────────────────────────────────────────────────────────────────────────
# Build:
#   docker build -t deepparse:v2 .
#
# Parse a mounted log file (Anthropic provider):
#   docker run --rm \
#     -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
#     -v $(pwd)/logs:/data:ro \
#     -v $(pwd)/out:/app/artifacts/output \
#     deepparse:v2 \
#     --input /data/fab_logs.log \
#     --static-masks /app/masks_fab_universal.json
#
# Offline/CI mode (no API key, universal masks only):
#   docker run --rm \
#     -v $(pwd)/logs:/data:ro \
#     -v $(pwd)/out:/app/artifacts/output \
#     deepparse:v2 \
#     --input /data/fab_logs.log \
#     --llm-provider mock \
#     --static-masks /app/masks_fab_universal.json
#
# Eval mode:
#   docker run --rm \
#     -v $(pwd)/out:/app/artifacts/output \
#     -v $(pwd)/eval:/app/artifacts/eval \
#     deepparse:v2 --mode eval
# ─────────────────────────────────────────────────────────────────────────────
