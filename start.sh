#!/usr/bin/env bash
set -euo pipefail

PORT="7860"
CONDA_ENV="sam2-env"
APP_FILE="gradio_segment.py"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda command not found. Please initialize conda first."
  exit 1
fi

echo "Checking port ${PORT}..."
pids="$(lsof -ti tcp:${PORT} || true)"

if [[ -n "${pids}" ]]; then
  echo "Port ${PORT} is occupied by PID(s): ${pids}"
  echo "Stopping existing process(es)..."
  kill ${pids} || true
  sleep 2

  still_running="$(lsof -ti tcp:${PORT} || true)"
  if [[ -n "${still_running}" ]]; then
    echo "Force killing PID(s): ${still_running}"
    kill -9 ${still_running} || true
    sleep 1
  fi
else
  echo "Port ${PORT} is free."
fi

if [[ ! -f "${APP_FILE}" ]]; then
  echo "ERROR: ${APP_FILE} not found in ${SCRIPT_DIR}"
  exit 1
fi

echo "Starting SAM2 UI with conda env '${CONDA_ENV}'..."
echo "Open: http://127.0.0.1:${PORT}"
exec conda run -n "${CONDA_ENV}" python "${APP_FILE}"
