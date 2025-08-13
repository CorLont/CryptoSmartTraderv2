# Multi-stage production Dockerfile for CryptoSmartTrader V2
# Enterprise-grade container with security hardening and health checks

# Build stage
FROM python:3.11.10-slim-bookworm AS builder

# Security: Create non-root user early
RUN groupadd --gid 1000 trader && \
    useradd --uid 1000 --gid trader --shell /bin/bash --create-home trader

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies to virtual environment
RUN uv sync --frozen --no-dev

# Production stage
FROM python:3.11.10-slim-bookworm AS production

# Metadata
LABEL org.opencontainers.image.title="CryptoSmartTrader V2"
LABEL org.opencontainers.image.description="Enterprise cryptocurrency trading intelligence system"
LABEL org.opencontainers.image.version="2.0.0"
LABEL org.opencontainers.image.vendor="CryptoSmartTrader"

# Security: Create non-root user
RUN groupadd --gid 1000 trader && \
    useradd --uid 1000 --gid trader --shell /bin/bash --create-home trader

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install UV in production stage
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/trader/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=trader:trader /app/.venv /app/.venv

# Copy application code
COPY --chown=trader:trader . .

# Create required directories with proper permissions
RUN mkdir -p /app/logs /app/data /app/models /app/cache /app/exports && \
    chown -R trader:trader /app

# Security: Switch to non-root user
USER trader

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

# Expose ports
EXPOSE 5000 8001 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Default command - multi-service startup
CMD ["sh", "-c", "uv sync --frozen && (uv run python api/health_endpoint.py & uv run python metrics/metrics_server.py & uv run streamlit run app_fixed_all_issues.py --server.port 5000 --server.headless true --server.address 0.0.0.0 & wait)"]