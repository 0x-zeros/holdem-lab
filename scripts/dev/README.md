# Development Scripts

Repository-level helpers for local development and the VS Code devcontainer.
The repo is shared by the **Linux devcontainer** and the **macOS host** via a bind
mount, so Python uses a **per-platform venv**: `.venv-docker` (container) /
`.venv-mac` (host). Never use the bare `.venv` and never hardcode a venv path —
resolve it via `venv-dir.sh`, run Python via `py` or `uv run`.

| Script | Purpose |
|--------|---------|
| `setup-dev-env.sh` | Install/update project Python deps into the per-platform venv (`uv sync`). Runs as the devcontainer `postCreateCommand`; safe to re-run by hand. |
| `verify-dev-env.sh` | Local verification: venv hygiene → ruff format/lint → mypy → pytest. Does **not** install deps. |
| `check-venv.sh` | Fail-closed guard against the bare `.venv` and wrong-platform venvs. Called by verify; safe standalone. |
| `venv-dir.sh` | Prints the canonical venv dir (`$UV_PROJECT_ENVIRONMENT` > `$PYTHON_VENV_DIR` > OS default). |
| `py` | Canonical Python entrypoint — runs the resolved venv's `python` so nothing hardcodes a path. |
| `claude-shell.sh` | Host-side helper: `docker exec` into the running devcontainer as `node` (matched by VS Code's `devcontainer.local_folder` label). |

## Why `setup-dev-env.sh` lives here, not in `.devcontainer/`

The devcontainer **image** owns system packages (`.devcontainer/Dockerfile`). This
script installs **project-level** deps and therefore needs the `/workspace` bind
mount, which only exists after the container is created — so `devcontainer.json`
runs it as `postCreateCommand`.

## Common usage

```bash
scripts/dev/setup-dev-env.sh        # install/update deps (also runs on container create)
scripts/dev/verify-dev-env.sh       # run the verification suite
scripts/dev/py -m pytest            # run Python through the per-platform venv
scripts/dev/claude-shell.sh         # (host) drop into the running devcontainer
```
