FROM ghcr.io/astral-sh/uv:0.8.0-python3.12-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev || uv sync --no-dev

COPY . .

EXPOSE 8000

CMD ["uv", "run", "python", "metrics_server.py", "--port", "8000", "--refresh-seconds", "30"]
