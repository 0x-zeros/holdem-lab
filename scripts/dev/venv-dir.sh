#!/usr/bin/env bash
# Resolve THE canonical per-platform Python venv dir for this repo, so tooling
# never silently falls back to the bare (ambiguous) .venv.
#
# Resolution order:
#   1) $UV_PROJECT_ENVIRONMENT   explicit override (what `uv` itself honors)
#   2) $PYTHON_VENV_DIR          devcontainer sets this (see .devcontainer)
#   3) OS default:  Linux -> .venv-docker ;  macOS -> .venv-mac   (at repo root)
#
# Why: the repo is shared by the Linux devcontainer AND the macOS host via a bind
# mount. The bare ".venv" name (every tool's default) collides between them and is
# how a wrong-platform venv keeps getting created. Run Python via scripts/dev/py
# or `uv run` (with UV_PROJECT_ENVIRONMENT); never hardcode a venv path.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ -n "${UV_PROJECT_ENVIRONMENT:-}" ]; then
  printf '%s\n' "${UV_PROJECT_ENVIRONMENT}"
elif [ -n "${PYTHON_VENV_DIR:-}" ]; then
  printf '%s\n' "${PYTHON_VENV_DIR}"
else
  case "$(uname -s)" in
    Darwin) printf '%s\n' "${repo_root}/.venv-mac" ;;
    *)      printf '%s\n' "${repo_root}/.venv-docker" ;;
  esac
fi
