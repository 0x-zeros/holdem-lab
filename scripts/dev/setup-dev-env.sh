#!/usr/bin/env bash
# Idempotent dev-env setup. Runs as the devcontainer postCreateCommand (user:
# node), and is safe to re-run by hand after dependency changes:
#   scripts/dev/setup-dev-env.sh
#
# The devcontainer IMAGE owns system packages (.devcontainer/Dockerfile). This
# script only installs project-level Python deps into the per-platform venv,
# which is why it lives under scripts/dev/ (it needs the /workspace bind mount)
# rather than in the image.
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[ -n "$ROOT_DIR" ] || ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="$(bash scripts/dev/venv-dir.sh)"
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"

log() { printf '\n==> %s\n' "$*"; }

log "uv sync   (venv: ${VENV_DIR})"
# Creates/updates the per-platform venv from pyproject.toml + uv.lock and installs the dev group.
uv sync

log "venv hygiene (scripts/dev/check-venv.sh)"
bash scripts/dev/check-venv.sh

log "toolchain versions"
scripts/dev/py --version
uv run ruff --version
uv run pytest --version

cat <<EOF

Dev environment ready.
  - Python venv : ${VENV_DIR}
  - Run Python  : scripts/dev/py ...        (or: UV_PROJECT_ENVIRONMENT=${VENV_DIR} uv run <cmd>)
  - Verify      : scripts/dev/verify-dev-env.sh

Heavy deps (pokerkit / open_spiel / rlcard / torch / opencv ...) are intentionally
NOT installed yet — add them with \`uv add <pkg>\` when building the engine/ai/bot modules.
EOF
