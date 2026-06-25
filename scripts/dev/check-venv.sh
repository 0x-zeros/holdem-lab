#!/usr/bin/env bash
# Guard against the two venv mistakes a shared bind-mounted repo keeps hitting
# (the repo is shared by the Linux devcontainer and the macOS host):
#   1) selecting bare .venv — the universal tool default; ambiguous / collides
#      across the two platforms. Mere existence is tolerated so agents do not
#      block on destructive cleanup; selection is not.
#   2) selecting a venv built for the wrong platform (e.g. a Linux venv on macOS).
# Fails closed only when the active/resolved environment is unsafe. Called by
# verify-dev-env.sh; safe to run standalone.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

abs_existing_parent_path() {
  local path="$1" dir base
  case "$path" in
    /*) ;;
    *) path="${repo_root}/${path}" ;;
  esac
  dir="$(dirname "$path")"
  base="$(basename "$path")"
  if [ -d "$dir" ]; then
    printf '%s/%s\n' "$(cd "$dir" && pwd -P)" "$base"
  else
    printf '%s/%s\n' "$dir" "$base"
  fi
}

venv_dir="$(bash "${script_dir}/venv-dir.sh")"
bare_venv="${repo_root}/.venv"
case "$(uname -s)" in
  Darwin) recommended_venv="${repo_root}/.venv-mac" ;;
  *)      recommended_venv="${repo_root}/.venv-docker" ;;
esac
resolved_venv_dir="$(abs_existing_parent_path "$venv_dir")"
resolved_bare_venv="$(abs_existing_parent_path "$bare_venv")"

# 1) Never select the bare .venv (reject obvious relative forms too, since uv
#    interprets project environments relative to the project root).
case "$venv_dir" in
  .venv|./.venv)
    cat >&2 <<EOF
error: resolved Python venv is ${venv_dir}, which selects the forbidden bare .venv.

  Use the per-platform venv instead:
    container -> .venv-docker   |   macOS host -> .venv-mac

  Run Python via scripts/dev/py, or export:
    UV_PROJECT_ENVIRONMENT="${recommended_venv}"
EOF
    exit 1
    ;;
esac

if [ "$resolved_venv_dir" = "$resolved_bare_venv" ]; then
  cat >&2 <<EOF
error: resolved Python venv is ${bare_venv}, which is forbidden.

  This repo is shared by the Linux devcontainer and the macOS host via a bind
  mount. The bare ".venv" name (every tool's default) collides between them.

  container -> .venv-docker   |   macOS host -> .venv-mac
  Run Python via scripts/dev/py, or export UV_PROJECT_ENVIRONMENT to the
  per-platform path before running uv directly:
    UV_PROJECT_ENVIRONMENT="${recommended_venv}"
EOF
  exit 1
fi

if [ -e "$bare_venv" ] || [ -L "$bare_venv" ]; then
  cat >&2 <<EOF
warning: ${bare_venv} exists but is ignored by repository tooling.
  No cleanup is required. Ensure direct 'uv run' shells export
  UV_PROJECT_ENVIRONMENT to the per-platform venv before use.
EOF
fi

# 2) The resolved venv (if present) must match the current OS platform.
#    Fail CLOSED: it must be executable, must actually run (a wrong-platform
#    binary can keep its +x bit but fail exec), and must report the current OS.
pybin="${venv_dir}/bin/python"
if [ -d "${venv_dir}" ]; then
  if [ ! -x "${pybin}" ]; then
    printf 'error: %s exists but %s is not executable (broken/wrong-platform venv).\n  Recreate: rm -rf %s && scripts/dev/setup-dev-env.sh\n' \
      "${venv_dir}" "${pybin}" "${venv_dir}" >&2
    exit 1
  fi
  if ! vplat="$("${pybin}" -c 'import platform; print(platform.system())' 2>/dev/null)"; then
    printf 'error: %s failed to run (broken/wrong-platform venv).\n  Recreate: rm -rf %s && scripts/dev/setup-dev-env.sh\n' \
      "${pybin}" "${venv_dir}" >&2
    exit 1
  fi
  os="$(uname -s)"
  if [ "${vplat}" != "${os}" ]; then
    printf 'error: %s is a %s venv but this OS is %s.\n  Recreate: rm -rf %s && scripts/dev/setup-dev-env.sh\n' \
      "${venv_dir}" "${vplat}" "${os}" "${venv_dir}" >&2
    exit 1
  fi
fi

printf 'venv check OK (%s)\n' "${venv_dir}"
