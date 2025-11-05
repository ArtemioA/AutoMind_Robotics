# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Marca de versión visible
ENV PORT=8080 PYTHONUNBUFFERED=1
# (opcional) imprime marca al iniciar
ENV STARTUP_MARK="✅ Deploy marker: Artemio v1.7-test"

CMD [ "bash", "-lc", "echo $STARTUP_MARK && gunicorn -k uvicorn.workers.UvicornWorker main:app --bind :$PORT --timeout 120" ]
