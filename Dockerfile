# Multi-stage production Dockerfile for CryptoSmartTrader V2
# FASE E - Enterprise deployment with security hardening

# Build stage
FROM python:3.13.6-slim-bookworm AS builder

# Security: Create non-root user for build
RUN groupadd --gid 1000 builder && \
    useradd --uid 1000 --gid builder --shell /bin/bash --create-home builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN pip install --no-cache-dir uv==0.1.6

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Production stage
FROM python:3.13.6-slim-bookworm AS production

# Security labels
LABEL maintainer="CryptoSmartTrader V2 Team"
LABEL version="2.0.0"
LABEL description="Enterprise cryptocurrency trading intelligence system"
LABEL security.scan="required"

# Create non-root user for production
RUN groupadd --gid 1000 trader && \
    useradd --uid 1000 --gid trader --shell /bin/bash --create-home trader

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create application directories with proper permissions
RUN mkdir -p /app /app/logs /app/data /app/models /app/exports \
    && chown -R trader:trader /app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=trader:trader /app/.venv /app/.venv

# Copy application code
COPY --chown=trader:trader . .

# Set Python path to use virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Security: Drop capabilities and run as non-root
USER trader

# Environment configuration
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CRYPTOSMARTTRADER_ENV=production
ENV CRYPTOSMARTTRADER_LOG_LEVEL=INFO

# Expose ports
EXPOSE 5000 8001 8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Create entrypoint script
COPY --chown=trader:trader docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Default command
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["dashboard"]