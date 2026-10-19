#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -f ".venv/bin/python" ]; then
  exec ".venv/bin/python" run.py "$@"
fi
exec "${PYTHON:-python3}" run.py "$@"
