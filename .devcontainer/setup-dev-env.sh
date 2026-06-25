#!/usr/bin/env bash
# Idempotent dev-env setup. Runs as postCreateCommand inside the container (user: node).
# Safe to re-run by hand: `bash .devcontainer/setup-dev-env.sh`
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "==> uv sync   (venv: ${UV_PROJECT_ENVIRONMENT:-.venv})"
# Creates/updates the project virtualenv from pyproject.toml + uv.lock and installs the dev group.
uv sync

echo "==> toolchain versions"
uv run python --version
uv run ruff --version
uv run pytest --version

cat <<EOF

Dev environment ready.
  - Python venv : ${UV_PROJECT_ENVIRONMENT:-.venv}
  - Activate    : source ${UV_PROJECT_ENVIRONMENT:-.venv}/bin/activate
  - Or prefix   : uv run <cmd>

Heavy deps (pokerkit / open_spiel / rlcard / torch / opencv ...) are intentionally NOT
installed yet — add them with \`uv add <pkg>\` when building the engine/ai/bot modules.
EOF
