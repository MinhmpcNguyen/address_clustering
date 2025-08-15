# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.11
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Tạo user không đặc quyền
ARG UID=10001
RUN adduser \
  --disabled-password \
  --gecos "" \
  --home "/home/appuser" \
  --shell "/bin/bash" \
  --uid "${UID}" \
  appuser

# Cài uv (copy binary)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Gói hệ thống: Postgres client/headers + toolchain build (gcc/gfortran) + python headers
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  postgresql libpq-dev \
  python3-dev build-essential gfortran \
  && rm -rf /var/lib/apt/lists/*
RUN pg_config --version

# Dùng shim distutils của setuptools (giải quyết thiếu distutils.*)
ENV SETUPTOOLS_USE_DISTUTILS=local

# Tối ưu cache cho uv
ENV UV_LINK_MODE=copy

# Cài deps theo lockfile trước khi copy source (tận dụng cache)
RUN --mount=type=cache,target=/root/.cache/uv \
  --mount=type=bind,source=uv.lock,target=/app/uv.lock \
  --mount=type=bind,source=pyproject.toml,target=/app/pyproject.toml \
  uv sync --locked --no-install-project

# Copy source
COPY . /app

# Đồng bộ môi trường ảo theo lock sau khi có source
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --locked

# Quyền thư mục và user
RUN mkdir -p /app/shared && chown -R appuser:appuser /app /home/appuser
USER appuser

# PYTHONPATH & PATH
ENV PYTHONPATH=app
ENV PATH="/app/.venv/bin:${PATH}"

# Expose port
EXPOSE 8000

# Chạy app
CMD ["uv", "run", "uvicorn", "api.app_main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]