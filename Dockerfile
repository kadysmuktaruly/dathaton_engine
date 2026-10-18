FROM python:3.11-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.11.15 /uv /uvx /bin/
ENV PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv PATH="/opt/venv/bin:$PATH"
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev
COPY api ./api
COPY engine ./engine
COPY rules ./rules
COPY tools ./tools
RUN useradd --create-home --uid 1000 engine && mkdir -p /app/runs /app/cache /app/weights /app/hf \
    && chown -R engine:engine /app
USER engine
ENV HF_HOME=/app/hf
EXPOSE 8001
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
