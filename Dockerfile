# Multi-stage production container for CloudInsight AI
FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install minimal build tools and curl for health checking
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root system user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/Datasets /app/Models /app/Reports /app/ReportTemplates && \
    chown -R appuser:appuser /app

# Install Python dependencies
COPY requirements-lock.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-lock.txt

# Copy application source
COPY --chown=appuser:appuser . /app

# Switch to non-root user
USER appuser

# Expose Streamlit default port
EXPOSE 8501

# Container healthcheck probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python scripts/healthcheck.py || exit 1

# Launch production server
ENTRYPOINT ["streamlit", "run", "App.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
