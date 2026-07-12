# ============================================================
# CIPHER — Fraud Detection System
# Dockerfile (Module 10 — Production-Hardened)
# ============================================================
#
# Build order is layer-cache-optimised:
#   1. System packages  (changes almost never)
#   2. requirements.txt (changes rarely)
#   3. pip install      (cached unless requirements changed)
#   4. Source code      (changes frequently — cache hit above)
#
# Usage:
#   docker build -t cipher-app .
#   docker run -p 8501:8501 cipher-app
# ============================================================

FROM python:3.11-slim

# ---------------------------------------------------------------------------
# Build-time metadata
# ---------------------------------------------------------------------------
LABEL maintainer="CIPHER Team" \
      version="1.0.0" \
      description="CIPHER Industry-Grade Fraud Detection System"

# ---------------------------------------------------------------------------
# Environment — prevents .pyc files and ensures stdout/stderr are unbuffered
# ---------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------------------------------------------------------------------------
# System dependencies
#   libgomp1  — OpenMP runtime required by LightGBM's multithreaded inference
#   curl      — used by docker-compose health checks
# ---------------------------------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Create a non-root user/group for security hardening
# ---------------------------------------------------------------------------
RUN groupadd --system cipher \
    && useradd --system --gid cipher --no-create-home --shell /sbin/nologin cipher

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------------------------
# Layer 1 — Python dependencies (cached unless requirements.txt changes)
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Layer 2 — Application source (invalidates cache on code changes only)
# ---------------------------------------------------------------------------
COPY src/ ./src/
COPY app.py .
COPY config/ ./config/

# ---------------------------------------------------------------------------
# Create runtime directories that may be bind-mounted by docker-compose.
# Pre-creating them ensures the cipher user owns them even before the mount.
# ---------------------------------------------------------------------------
RUN mkdir -p data models plots/shap logs reports \
    && chown -R cipher:cipher /app

# ---------------------------------------------------------------------------
# Drop to non-root user
# ---------------------------------------------------------------------------
USER cipher

# ---------------------------------------------------------------------------
# Expose Streamlit port
# ---------------------------------------------------------------------------
EXPOSE 8501

# ---------------------------------------------------------------------------
# Health check — Streamlit's built-in health endpoint
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ---------------------------------------------------------------------------
# Entrypoint — start Streamlit dashboard
# ---------------------------------------------------------------------------
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
