FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace
RUN pip install --upgrade pip && pip install -e ".[test,lint]"

ENTRYPOINT ["deepparse"]
