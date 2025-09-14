FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Código de la API (incluye app/ y artefactos)
COPY . /app

# Permisos de arranque
RUN chmod +x /app/run.sh

# Puerto por defecto (Railway inyecta $PORT igualmente)
EXPOSE 8001

# Comando de inicio
CMD ["/app/run.sh"]
