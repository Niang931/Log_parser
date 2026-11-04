# logpipe server: agent intake, the operator UI and the learning lifecycle.
FROM node:26-alpine AS web
WORKDIR /src/web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-fund --no-audit
COPY web/ ./
RUN npm run build

FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never \
    PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY DeepParse DeepParse
RUN uv sync --frozen --no-dev --no-install-project
COPY logpipe logpipe
COPY --from=web /src/logpipe/server/static logpipe/server/static
RUN uv sync --frozen --no-dev
VOLUME /app/artifacts
EXPOSE 8080
CMD ["logpipe", "server", "--host", "0.0.0.0", "--port", "8080", "--store", "clickhouse"]
