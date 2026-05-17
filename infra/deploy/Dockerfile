# Production image for the RAG API.
#
# Two-stage build:
#  - Stage 1 installs deps into a venv (heavy: torch, sentence-transformers).
#  - Stage 2 copies only the venv + source into a slim runtime image.
#
# The image does NOT bundle Ollama or any embedding model — those run as a
# separate container in docker-compose so embedding weights can be cached on
# a host volume independent of API rebuilds.

FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:${PATH}" \
    COLLECTIONS_DIR=/data/collections

# Non-root user for the API process.
RUN groupadd --system --gid 1001 prudentia \
    && useradd --system --uid 1001 --gid prudentia --home /app prudentia

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY src ./src
COPY config.yaml ./
RUN mkdir -p /data/collections /app/logs && chown -R prudentia:prudentia /app /data
USER prudentia

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
