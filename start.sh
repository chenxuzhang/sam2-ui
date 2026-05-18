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

get_port_pids() {
  if command -v lsof >/dev/null 2>&1; then
    # macOS: lsof is reliable
    lsof -ti "tcp:${PORT}" 2>/dev/null || true
  elif command -v ss >/dev/null 2>&1; then
    # Linux: try ss (prefer socket info)
    ss -tlnp "sport = :${PORT}" 2>/dev/null \
      | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' \
      | sort -u || true
  elif command -v fuser >/dev/null 2>&1; then
    # Linux fallback: fuser (output format: "1234 5678 " with spaces)
    fuser "${PORT}/tcp" 2>/dev/null | tr ' ' '\n' | sed '/^$/d' || true
  elif command -v netstat >/dev/null 2>&1; then
    netstat -tlnp 2>/dev/null | awk -v p=":${PORT} " '$4 ~ p { split($NF,a,"/"); print a[1] }' | sort -u || true
  fi
}

pids="$(get_port_pids)"

if [[ -n "${pids}" ]]; then
  echo "Port ${PORT} is occupied by PID(s): ${pids}"
  echo "Stopping existing process(es)..."
  kill ${pids} || true
  sleep 2

  still_running="$(get_port_pids)"
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

export GRADIO_ROOT_PATH="/proxy/${PORT}"

echo "Starting SAM2 UI with conda env '${CONDA_ENV}'..."
echo "Proxy URL: /proxy/${PORT}"
exec conda run -n "${CONDA_ENV}" python "${APP_FILE}"
