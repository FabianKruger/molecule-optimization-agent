#!/usr/bin/env bash
# Stop local web app processes: seismo-api (backend) and Vite (frontend).
# Prefer process-name matches over port 8000 so a vLLM serve is left alone.

set -eu

stop_pattern() {
  local label=$1
  local pattern=$2
  local pids
  pids=$(pgrep -f "$pattern" 2>/dev/null || true)
  if [ -z "$pids" ]; then
    echo "No ${label} processes found."
    return 0
  fi
  echo "Stopping ${label}: $(echo "$pids" | tr '\n' ' ')"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  sleep 0.5
  pids=$(pgrep -f "$pattern" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "Force-killing ${label}: $(echo "$pids" | tr '\n' ' ')"
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
  fi
}

stop_npm_frontend() {
  local pids pid cwd
  pids=$(pgrep -f 'npm run dev' 2>/dev/null || true)
  if [ -z "$pids" ]; then
    echo "No npm frontend processes found."
    return 0
  fi
  for pid in $pids; do
    cwd=$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -1 || true)
    case "$cwd" in
      */frontend|*/frontend/)
        echo "Stopping npm frontend (PID ${pid}, cwd ${cwd})"
        kill "$pid" 2>/dev/null || true
        sleep 0.3
        if kill -0 "$pid" 2>/dev/null; then
          echo "Force-killing npm frontend (PID ${pid})"
          kill -9 "$pid" 2>/dev/null || true
        fi
        ;;
    esac
  done
}

stop_pattern "pixi seismo-api" "pixi run seismo-api"
stop_pattern "pixi frontend" "pixi run frontend"
stop_pattern "seismo-api" "seismo-api"
stop_pattern "Vite frontend" "frontend/node_modules/.bin/vite"
stop_npm_frontend

echo "Done."
