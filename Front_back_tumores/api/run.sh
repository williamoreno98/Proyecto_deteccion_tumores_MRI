#!/usr/bin/env bash
set -e
PORT="${PORT:-8001}"
WORKERS="${WORKERS:-1}"
HOST="0.0.0.0"
exec uvicorn app.main:app --host "$HOST" --port "$PORT" --workers "$WORKERS"
