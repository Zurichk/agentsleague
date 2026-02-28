FROM python:3.12-slim

LABEL maintainer="AEP CertMaster"
LABEL description="Dashboard multi-agente de preparación para certificaciones Microsoft"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=5033

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p logs/traces data

EXPOSE 5033

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

CMD gunicorn \
    --worker-class eventlet \
    -w 1 \
    --bind 0.0.0.0:${PORT} \
    --timeout 120 \
    --log-level info \
    --access-logfile - \
    "dashboard.app:app"
