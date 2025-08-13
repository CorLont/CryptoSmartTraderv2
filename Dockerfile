# Enterprise-grade Python runtime with pinned versions
FROM python:3.11.9-slim-bookworm

# Security: Create non-root user
RUN groupadd --gid 1000 trader && \
    useradd --uid 1000 --gid trader --shell /bin/bash --create-home trader

# Install system dependencies with pinned versions for security
RUN apt-get update && apt-get install -y \
    curl=7.88.1-10+deb12u8 \
    git=1:2.39.2-1.1 \
    build-essential=12.9 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies with uv sync (frozen lockfile)
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ src/
COPY api/ api/
COPY configs/ configs/

# Set ownership to non-root user
RUN chown -R trader:trader /app

# Switch to non-root user
USER trader

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose ports
EXPOSE 5000 8000 8001

# Default command
CMD ["uv", "run", "streamlit", "run", "app_fixed_all_issues.py", "--server.port", "5000", "--server.headless", "true", "--server.address", "0.0.0.0"]