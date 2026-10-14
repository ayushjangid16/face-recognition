# syntax=docker/dockerfile:1
###############################################################################
# Stage 1 — builder
# dlib (pulled in by face_recognition) compiles FROM SOURCE and needs cmake +
# a C++ toolchain. Everything (wheels included) is installed under /install so
# the runtime stage can copy it verbatim without ever running pip or a compiler.
###############################################################################
FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    CMAKE_BUILD_PARALLEL_LEVEL=4

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Dependency layer is cached independently of application code: it only rebuilds
# when requirements.txt changes.
COPY requirements.txt ./
RUN pip install --prefix=/install -r requirements.txt

###############################################################################
# Stage 2 — runtime (slim, non-root, no compilers / no pip cache)
###############################################################################
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=10000 \
    DATA_DIR=/app/data

# Runtime shared libraries only:
#   libglib2.0-0 — required by opencv-python-headless
#   libgomp1     — OpenMP runtime used by dlib
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# All third-party Python packages come from the builder stage (site-packages +
# console scripts such as `uvicorn` land in /usr/local).
COPY --from=builder /install /usr/local

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

WORKDIR /app

# Only what the app actually imports/serves at runtime:
#   app/       -> FastAPI application (app.main:app)
#   routers/   -> page router (imported by app.main)
#   static/, templates/ -> the web UI (served by routers/pages.py)
COPY app ./app
COPY routers ./routers
COPY static ./static
COPY templates ./templates

# The app creates data/gallery.db (SQLite) at startup; give only that
# directory to the unprivileged user. Code stays root-owned/read-only.
RUN mkdir -p /app/data \
    && chown -R appuser:appuser /app/data

USER appuser

# EXPOSE documents the default port only — it does NOT publish the port and
# Render does not read it. Render routes traffic to the value of $PORT
# (dashboard "Port" field, default 10000), which the CMD below reads at
# container start, so changing the port needs no rebuild.
EXPOSE 10000

# Uses the app's real health endpoint (GET /api/health). Render ignores
# HEALTHCHECK and uses its own probe, but this helps during local testing.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os, urllib.request as u; u.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT', '10000'), timeout=4)"

# Single uvicorn worker ON PURPOSE: scan sessions and the gallery cache live in
# process memory (see app/main.py docstring). `exec` makes uvicorn PID 1 so
# SIGTERM from Docker/Render triggers a graceful shutdown.
# --proxy-headers: Render terminates TLS and sets X-Forwarded-*; trusting them
# (only reachable through Render's proxy) gives correct client IPs for the
# per-IP rate limit. No secrets and no .env are baked in — configuration comes
# from runtime environment variables only.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"${PORT:-10000}\" --proxy-headers --forwarded-allow-ips '*'"]
