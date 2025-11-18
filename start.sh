#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_MODULE="app.main:app"
APP_DIR="."
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
WORKERS="${WEB_CONCURRENCY:-2}"
LOG_LEVEL="${LOG_LEVEL:-info}"

cd "$ROOT_DIR"

exec uvicorn "$APP_MODULE" \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL" \
    --app-dir "$APP_DIR"
