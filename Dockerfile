# Multi-stage production Dockerfile voor CryptoSmartTrader V2
# Enterprise-grade container met security hardening en health checks

# Stage 1: Builder
FROM python:3.11.10-slim-bookworm AS builder

# Metadata
LABEL org.opencontainers.image.title="CryptoSmartTrader V2"
LABEL org.opencontainers.image.description="Enterprise cryptocurrency trading intelligence system"
LABEL org.opencontainers.image.version="2.0.0"
LABEL org.opencontainers.image.vendor="CryptoSmartTrader"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Stage 2: Runtime
FROM python:3.11.10-slim-bookworm AS runtime

# Security: Create non-root user
RUN groupadd --gid 1000 trader && \
    useradd --uid 1000 --gid trader --shell /bin/bash --create-home trader

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get autoclean

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    TRADING_MODE=paper

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=trader:trader /app/.venv /app/.venv

# Copy application code
COPY --chown=trader:trader . .

# Create necessary directories
RUN mkdir -p logs cache reports config && \
    chown -R trader:trader logs cache reports config

# Security: Drop capabilities and set limits
USER trader

# Expose ports
EXPOSE 5000 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Use tini for signal handling
ENTRYPOINT ["tini", "--"]

# Default command
CMD ["python", "-m", "uvicorn", "src.cryptosmarttrader.api.main:app", "--host", "0.0.0.0", "--port", "5000"]