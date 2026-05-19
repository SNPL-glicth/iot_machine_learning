# ════════════════════════════════════════════════════════════════════
# ZENIN ML Service — Multi-stage Dockerfile (Workspace Root Context)
# Build:
#   docker build -t zenin-ml:local . --no-cache
# Run:
#   docker run --rm -p 8002:8002 zenin-ml:local
# ════════════════════════════════════════════════════════════════════

# ─── STAGE 1: builder ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    unixodbc-dev \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY iot_machine_learning/requirements.txt ./requirements.txt
COPY iot_ingest_services/requirements.txt ./requirements-ingest.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       -r requirements.txt \
       -r requirements-ingest.txt

# ─── STAGE 2: runtime ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime deps: unixodbc for pyodbc, libgomp1 for numpy/openmp
RUN apt-get update && apt-get install -y --no-install-recommends \
    unixodbc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages + entrypoints from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Non-root user
RUN groupadd -r zenin && useradd -r -g zenin zenin

# Writable directories for compliance export and batch audit logs
RUN mkdir -p /var/lib/zenin/compliance /app/logs \
    && chown -R zenin:zenin /var/lib/zenin /app/logs

WORKDIR /app

# Copy source code (both sibling modules)
COPY iot_machine_learning/ ./iot_machine_learning/
COPY iot_ingest_services/ ./iot_ingest_services/

RUN chown -R zenin:zenin /app

USER zenin

ENV PYTHONPATH=/app
ENV ML_API_WORKERS=4

EXPOSE 8002

CMD uvicorn iot_machine_learning.ml_service.main:app \
    --host 0.0.0.0 --port 8002 --workers ${ML_API_WORKERS:-4}
