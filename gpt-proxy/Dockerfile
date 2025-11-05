# ===== base =====
FROM python:3.11-slim

# Evitar __pycache__ y buffers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Útil para healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends curl \
  && rm -rf /var/lib/apt/lists/*

# ===== deps =====
# Copiamos primero requirements para aprovechar la cache de capas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ===== app =====
# Copia el resto del código (incluye main.py)
COPY . .

# Usuario no-root por seguridad
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Puerto (Cloud Run usa 8080 por defecto)
EXPOSE 8080

# Healthcheck al endpoint /health (tu app lo expone)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://localhost:${PORT}/health || exit 1

# Arranque con Gunicorn + workers Uvicorn
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", ":8080", "--timeout", "120"]
