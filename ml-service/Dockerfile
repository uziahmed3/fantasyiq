FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MODEL_DIR=/models
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential libgomp1 curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt ./
# CPU-only torch wheel: ~200MB instead of ~2.5GB of CUDA the API will never use.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.txt

COPY . .
RUN mkdir -p /models && useradd -m -u 10002 mluser && chown -R mluser:mluser /app /models
USER mluser

EXPOSE 9000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:9000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
