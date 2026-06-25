#!/usr/bin/env bash
# Run the local verification suite. Assumes the dev environment has already been
# installed by scripts/dev/setup-dev-env.sh.
#   scripts/dev/verify-dev-env.sh
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[ -n "$ROOT_DIR" ] || ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="$(bash scripts/dev/venv-dir.sh)"
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

log "venv hygiene (scripts/dev/check-venv.sh)"
bash scripts/dev/check-venv.sh

[ -x "${VENV_DIR}/bin/python" ] || die "venv not found at ${VENV_DIR}; run scripts/dev/setup-dev-env.sh first"

# Static checks only make sense once there is Python source (mypy/ruff error on
# an empty tree). The skeleton has none yet (legacy rust/web is excluded), so
# guard on it. Keep these excludes in sync with pyproject's ruff/mypy excludes.
if find . -name '*.py' \
     -not -path '*/.venv*' \
     -not -path './rust/*' -not -path './web/*' -not -path './equity-calculator/*' \
     2>/dev/null | grep -q .; then
  log "ruff format --check"
  uv run ruff format --check .
  log "ruff lint"
  uv run ruff check .
  log "mypy"
  uv run mypy .
else
  log "ruff / mypy skipped (no .py files yet)"
fi

log "pytest"
# exit code 5 == "no tests collected" — acceptable for the current skeleton.
set +e
uv run pytest -q
rc=$?
set -e
if [ "$rc" -ne 0 ] && [ "$rc" -ne 5 ]; then
  die "pytest failed (exit ${rc})"
fi
[ "$rc" -eq 5 ] && log "pytest: no tests collected yet"

log "verify OK"
