# holdem-lab dev container

Three-tier hybrid dev environment (see the full plan at
`~/.claude/plans/ai-ai-steam-ai-jaunty-stonebraker.md`). **This directory is tier ① only.**

| Tier | What | Where it runs | Status |
|------|------|---------------|--------|
| ① headless core | `engine` + `ai`: rules, CFR/RL, tests, CI | this dev container (Linux) | **built here** |
| ② GUI / bot harness | `game` (pygame) + "bot vs our own game" | Xvfb + noVNC container | deferred to game/bot phase |
| ③ real Steam bot | drive Poker Legends (Steam 758980) | macOS host (native) | deferred; game has no Linux build |

## Tier ① — what's inside

`ubuntu:24.04` + Node 24 (for the Claude Code / Codex CLIs) + **uv-managed CPython 3.12** +
a native build toolchain (`build-essential cmake clang pkg-config`). The `node` user has
passwordless `sudo`, so later sessions can `sudo apt-get install` CV/OCR/GUI libs without an
image rebuild.

- Project venv: `/workspace/.venv-docker` (host side uses a separate `.venv-mac`).
- `postCreateCommand` runs `scripts/dev/setup-dev-env.sh` → `uv sync`. The dev
  helper scripts live in [`scripts/dev/`](../scripts/dev/README.md).
- Heavy deps (pokerkit / open_spiel / rlcard / torch / opencv) are **not** pre-installed;
  add them with `uv add` when building those modules.

## Usage

Open the folder in VS Code → **Dev Containers: Reopen in Container**, then start a new
Claude Code session **inside** the container. Or build/run manually:

```bash
docker build -t holdem-lab-dev .devcontainer
docker run --rm -it -v "$PWD":/workspace -w /workspace holdem-lab-dev zsh
```

## Tier ② / ③ (later)

- Tier ② will add `xvfb x11vnc websockify novnc fluxbox libgl1 libsdl2* tesseract-ocr` on
  top of this image (a `docker/gui/` compose service, `FROM` the tier ① image) so the pygame
  game + the screen-capture→OCR→click bot pipeline run headless and are viewable at
  `http://localhost:7900`. Pattern borrowed from `web3-tycoon/docker/e2e`.
- Tier ③ runs natively on macOS (Screen Recording + Accessibility permissions); the real
  Poker Legends game has no Linux build, so it cannot live in a container.
