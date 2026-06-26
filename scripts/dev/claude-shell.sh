#!/bin/bash
# Host-side helper: enter this project's running VS Code devcontainer.
# Usage from the repository root:
#   scripts/dev/claude-shell.sh

set -euo pipefail

PROJECT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# Match the VS Code devcontainer precisely by its local-folder label.
CONTAINER=$(docker ps \
  --filter "label=devcontainer.local_folder=${PROJECT_DIR}" \
  --format "{{.Names}}" \
  | head -1)

if [ -z "$CONTAINER" ]; then
  echo "No devcontainer found for project: ${PROJECT_DIR}"
  echo "Open it first in VS Code: Dev Containers: Reopen in Container"
  exit 1
fi

echo "Entering devcontainer: $CONTAINER"

docker exec -it -u node -w /workspace "$CONTAINER" zsh

# codex在.devcontainer里现在用得最舒服的命令：
# 无文件沙盒限制，但仍保留审批：更像全盘权限；高风险动作仍可要求确认。
# codex --sandbox danger-full-access --ask-for-approval on-request

# --- handy variants (copy/run inside the container as needed) ---
#
# Claude Code:
#   claude                                   # normal, with approvals
#   claude --permission-mode auto            # auto-accept workspace edits
#   claude --dangerously-skip-permissions    # only inside an isolated devcontainer
#
# Codex (permission low -> high):
#   codex --sandbox read-only --ask-for-approval on-request
#   codex --sandbox workspace-write --ask-for-approval on-request
#   codex --sandbox workspace-write --add-dir /workspace/.git --ask-for-approval on-request
#   codex --sandbox workspace-write --add-dir /workspace/.git --ask-for-approval never
#   codex --sandbox danger-full-access --ask-for-approval on-request
#   codex --dangerously-bypass-approvals-and-sandbox   # only inside an isolated devcontainer
