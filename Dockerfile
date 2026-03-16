# ==============================================================
#  Multi-stage Dockerfile for the Time Series ML Pipeline
# ==============================================================

# ---------- Stage 1: builder ------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- Stage 2: runtime ------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="Gabriel Demetrios Lafis"
LABEL description="Cassandra Time Series ML Pipeline"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY main.py .

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "main.py"]
