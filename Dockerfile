# Kolada MCP Server Dockerfile
# Multi-stage build for optimal image size

FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install . --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY src/ src/

# Install the application
RUN pip install -e .

# Pre-download the sentence transformer model
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_OFFLINE=0

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('KBLab/sentence-bert-swedish-cased')"

# Set offline mode after downloading
ENV TRANSFORMERS_OFFLINE=1

# Production stage
FROM python:3.11-slim AS production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_OFFLINE=1 \
    MCP_TRANSPORT=http \
    PORT=8001 \
    PYTHONPATH=/app/src

WORKDIR /app

# Copy installed packages from builder
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Copy application and model cache
COPY --from=base /app/src /app/src
COPY --from=base /app/.cache /app/.cache

# Create non-root user
RUN useradd -m -u 1000 app && \
    chown -R app:app /app

USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:${PORT}/health/live')" || exit 1

EXPOSE 8001

# Run the server
CMD ["python", "-m", "kolada_mcp"]
