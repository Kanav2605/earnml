# ─────────────────────────────────────────────────────────────────
# Stage 1: Builder — install Python deps into a clean layer
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for scientific Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ git curl \
    libhdf5-dev libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into a prefix so we can copy cleanly to stage 2
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ─────────────────────────────────────────────────────────────────
# Stage 2: Runtime — slim final image
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project source
COPY api/       ./api/
COPY ml/        ./ml/
COPY monitoring/ ./monitoring/

# Create model storage directory
RUN mkdir -p /app/ml/models/saved /app/logs /app/mlruns

# Non-root user (security best practice — JPM would care about this)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=http://mlflow:5000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
