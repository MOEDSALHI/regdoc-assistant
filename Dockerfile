# ============================================================
# Stage 1 — builder
# ============================================================
FROM python:3.12.7-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Install uv directly from the official image (no Python overhead)
COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /usr/local/bin/uv

WORKDIR /app

# Copy only dependency files first — Docker cache friendly
# (uv sync only re-runs when pyproject.toml or uv.lock changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Pre-download the cross-encoder model into the venv's HF cache
# This bakes the 120MB model into the image — no network needed at runtime
ENV HF_HOME=/app/.cache/huggingface
RUN uv run python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', max_length=512)"

# Now copy the application code (changes most often, smallest layer)
COPY src/ ./src/
COPY main.py ./

# Install the project itself
RUN uv sync --frozen --no-dev

# ============================================================
# Stage 2 — runtime
# ============================================================
FROM python:3.12.7-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    PATH="/app/.venv/bin:$PATH"

# Create non-root user
RUN groupadd --system app && useradd --system --gid app --no-create-home app

WORKDIR /app

# Copy venv (includes installed packages and pre-downloaded HF cache) from builder
COPY --from=builder --chown=app:app /app /app

USER app

EXPOSE 8000

# Healthcheck — used by docker-compose to wait for readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]