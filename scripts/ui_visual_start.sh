#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${UI_PORT:-8502}"
LOG_DIR="$ROOT_DIR/logs"
PID_FILE="$LOG_DIR/ui_visual.pid"
OUT_LOG="$LOG_DIR/ui_visual.log"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "ui_visual ya está corriendo con PID $(cat "$PID_FILE")"
  exit 0
fi

cd "$ROOT_DIR"
nohup streamlit run ui_visual/app.py --server.port "$PORT" --server.headless true >"$OUT_LOG" 2>&1 &
echo $! > "$PID_FILE"
echo "ui_visual iniciado en puerto $PORT (PID $(cat "$PID_FILE"))"
