# holdem-lab dev container — tier ① "headless core" (engine + ai)
#
# Every download comes from a canonical/official source over HTTPS, and is checksum-verified
# where the publisher ships checksums:
#   - base image   : Docker Hub official library image  (docker.io/library/ubuntu:24.04)
#   - apt packages : Ubuntu official repositories       (incl. Python 3.12 — Ubuntu's default)
#   - Node.js      : official nodejs.org tarball         + SHA256 verify vs SHASUMS256.txt
#   - uv           : official astral-sh GitHub release   + SHA256 verify vs .sha256 sidecar
#   - git-delta    : official dandavison GitHub release  (pinned)
#   - claude/codex : official npm registry               (official Anthropic / OpenAI packages)
# No third-party apt repos (no NodeSource) and no `curl | sh` convenience installers.
FROM docker.io/library/ubuntu:24.04

ARG TZ
ENV TZ="$TZ"
ENV DEBIAN_FRONTEND=noninteractive

ARG CLAUDE_CODE_VERSION=latest
ARG CODEX_VERSION=latest
ARG NODE_MAJOR=24
ARG GIT_DELTA_VERSION=0.18.2

# --- base tools + Ubuntu's Python 3.12 + native build toolchain + CV/OCR runtime ---
# Python 3.12 is Ubuntu 24.04's default python3 (official). build-essential/cmake/clang/
# python3-dev let us compile native wheels later (e.g. OpenSpiel from source on arm64).
# xz-utils is needed to extract the official Node .tar.xz.
# tesseract-ocr/libgl1/fontconfig/fonts-dejavu-core support the bot CV/OCR pipeline.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget gnupg git jq less procps sudo fzf zsh man-db unzip nano vim locales xz-utils \
    build-essential cmake pkg-config clang libssl-dev \
    python3 python3-venv python3-dev python3-pip python-is-python3 \
    tesseract-ocr libgl1 fontconfig fonts-dejavu-core \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- locale ---
RUN sed -i 's/^# *\(en_US.UTF-8 UTF-8\)/\1/' /etc/locale.gen \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# --- node user at uid 1000 (ubuntu:24.04 ships an 'ubuntu' user there) + passwordless sudo ---
# passwordless sudo lets later dev sessions `apt-get install` CV/OCR/GUI libs without an image rebuild.
RUN userdel -r ubuntu 2>/dev/null || true \
    && groupdel ubuntu 2>/dev/null || true \
    && groupadd --gid 1000 node \
    && useradd --uid 1000 --gid node --shell /bin/zsh --create-home node \
    && echo "node ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/node \
    && chmod 0440 /etc/sudoers.d/node

# --- Node.js from the OFFICIAL nodejs.org distribution (+ SHA256 verify), installed to /usr/local ---
# Only used to run the Claude Code / Codex CLIs. latest-v24.x tracks the newest 24.x; the tarball
# is verified against SHASUMS256.txt fetched from the same official directory.
RUN ARCH=$(dpkg --print-architecture) && \
    case "$ARCH" in \
      arm64) NODE_ARCH=arm64 ;; \
      amd64) NODE_ARCH=x64 ;; \
      *) echo "Unsupported arch: $ARCH" && exit 1 ;; \
    esac && \
    cd /tmp && \
    wget -q "https://nodejs.org/dist/latest-v${NODE_MAJOR}.x/SHASUMS256.txt" && \
    NODE_TARBALL="$(awk -v a="linux-${NODE_ARCH}.tar.xz" '$2 ~ a {print $2}' SHASUMS256.txt | grep -v headers | head -1)" && \
    wget -q "https://nodejs.org/dist/latest-v${NODE_MAJOR}.x/${NODE_TARBALL}" && \
    grep " ${NODE_TARBALL}\$" SHASUMS256.txt | sha256sum -c - && \
    tar -xJf "${NODE_TARBALL}" -C /usr/local --strip-components=1 && \
    rm -f "${NODE_TARBALL}" SHASUMS256.txt && \
    node --version && npm --version

# --- uv from the OFFICIAL astral-sh GitHub release (+ SHA256 verify vs .sha256 sidecar) ---
RUN ARCH=$(dpkg --print-architecture) && \
    case "$ARCH" in \
      arm64) UV_ARCH=aarch64-unknown-linux-gnu ;; \
      amd64) UV_ARCH=x86_64-unknown-linux-gnu ;; \
      *) echo "Unsupported arch: $ARCH" && exit 1 ;; \
    esac && \
    cd /tmp && \
    wget -q "https://github.com/astral-sh/uv/releases/latest/download/uv-${UV_ARCH}.tar.gz" && \
    wget -q "https://github.com/astral-sh/uv/releases/latest/download/uv-${UV_ARCH}.tar.gz.sha256" && \
    sha256sum -c "uv-${UV_ARCH}.tar.gz.sha256" && \
    tar -xzf "uv-${UV_ARCH}.tar.gz" -C /usr/local/bin --strip-components=1 "uv-${UV_ARCH}/uv" "uv-${UV_ARCH}/uvx" && \
    rm -f "uv-${UV_ARCH}.tar.gz" "uv-${UV_ARCH}.tar.gz.sha256" && \
    uv --version

# uv must use Ubuntu's apt Python — never download a CPython build from the internet.
ENV UV_PYTHON_PREFERENCE=only-system \
    UV_PYTHON_DOWNLOADS=never

# --- npm global dir, command history, workspace + config dirs ---
RUN mkdir -p /usr/local/share/npm-global && chown -R node:node /usr/local/share
ENV DEVCONTAINER=true
RUN mkdir /commandhistory && touch /commandhistory/.bash_history && chown -R node:node /commandhistory
RUN mkdir -p /workspace /home/node/.claude /home/node/.codex && \
    chown -R node:node /workspace /home/node/.claude /home/node/.codex
WORKDIR /workspace

# --- git-delta (nicer diffs) from the OFFICIAL dandavison GitHub release (pinned) ---
RUN ARCH=$(dpkg --print-architecture) && \
    cd /tmp && \
    wget -q "https://github.com/dandavison/delta/releases/download/${GIT_DELTA_VERSION}/git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
    dpkg -i "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
    rm "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb"

# --- non-root user from here on ---
USER node
ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH=$PATH:/usr/local/share/npm-global/bin
ENV SHELL=/bin/zsh \
    EDITOR=nano \
    VISUAL=nano

# --- minimal zsh config (no oh-my-zsh / powerlevel10k / third-party installer) ---
RUN cat > /home/node/.zshrc <<'ZSHRC'
export HISTFILE=/commandhistory/.bash_history
export HISTSIZE=100000
export SAVEHIST=100000
setopt SHARE_HISTORY INC_APPEND_HISTORY HIST_IGNORE_ALL_DUPS
autoload -Uz compinit && compinit -u
PROMPT='%F{cyan}node@holdem-lab%f %F{yellow}%~%f %# '
[ -f /usr/share/doc/fzf/examples/key-bindings.zsh ] && source /usr/share/doc/fzf/examples/key-bindings.zsh
[ -f /usr/share/doc/fzf/examples/completion.zsh ] && source /usr/share/doc/fzf/examples/completion.zsh
ZSHRC

# --- Claude Code (official npm; explicit native platform package so the binary isn't skipped) ---
RUN ARCH=$(dpkg --print-architecture) && \
    case "$ARCH" in \
      arm64) NATIVE_PKG=@anthropic-ai/claude-code-linux-arm64 ;; \
      amd64) NATIVE_PKG=@anthropic-ai/claude-code-linux-x64 ;; \
      *) echo "Unsupported arch: $ARCH" && exit 1 ;; \
    esac && \
    npm install -g --include=optional \
      "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
      "${NATIVE_PKG}" && \
    node /usr/local/share/npm-global/lib/node_modules/@anthropic-ai/claude-code/install.cjs && \
    claude --version

# --- OpenAI Codex CLI (official npm; native binary via npm alias) ---
RUN ARCH=$(dpkg --print-architecture) && \
    case "$ARCH" in \
      arm64) PLATFORM=linux-arm64 ;; \
      amd64) PLATFORM=linux-x64 ;; \
      *) echo "Unsupported arch: $ARCH" && exit 1 ;; \
    esac && \
    npm install -g \
      "@openai/codex@${CODEX_VERSION}" \
      "@openai/codex-${PLATFORM}@npm:@openai/codex@${PLATFORM}" && \
    codex --version
