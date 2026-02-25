#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT_DIR/logs/ui_visual.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No existe logs/ui_visual.pid"
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "ui_visual detenido (PID $PID)"
else
  echo "PID $PID no está corriendo"
fi
rm -f "$PID_FILE"
