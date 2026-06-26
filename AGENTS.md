# AGENTS.md

本项目的**规范化主规则文件**（canonical rules）。Codex 直接读取；Claude Code 经 `CLAUDE.md`
顶部的 `@AGENTS.md` import 读取。本文件只承载规则与锚点，不放详情（命令/架构等放 `CLAUDE.md`
或被引用的文档）。

## 依赖与下载来源（Dependency & Download Sources）

**所有外部产物（第三方依赖、命令行工具、预编译二进制、容器基础镜像、模型权重、数据集等）
一律只从官方 / 权威来源获取；严禁从来路不明、非官方或第三方转发的渠道下载。**

- **只允许的来源**：① 项目官网，或其官方文档明确指定的下载地址；② 官方包管理仓库（PyPI、
  npm registry、crates.io、Ubuntu/Debian 官方 apt 源、Docker Hub 官方/已认证镜像）；③ 软件作者
  或官方组织的正式发布（如其官方 GitHub org 的 Releases）。
- **禁止的来源**：非官方第三方镜像站、个人搬运/转发、来路不明的随机链接、`curl … | sh` 之类
  未经核验的一键安装脚本、搜索引擎或 AI 临时给出的非官方下载页。
- **必须校验完整性**：官方若提供校验信息（SHA256 / GPG 签名 / 官方 checksum 文件），下载后
  必须校验通过再使用；尽量 pin 明确版本，避免无法审计的浮动来源。
- **必须使用安全传输**：一律走 HTTPS 或等价的已签名渠道；不通过明文 HTTP 获取可执行内容。
- **存疑即停**：无法确认来源是否官方/权威时，停止并向用户确认，绝不擅自从可疑来源下载。

> 落地示例（`.devcontainer/Dockerfile`）：Node.js 取自 **nodejs.org** 并校验 `SHASUMS256.txt`；
> uv 取自 **astral-sh 官方 GitHub Release** 并校验 `.sha256` sidecar；Python 用 **Ubuntu 官方
> apt**；`claude-code` / `codex` 用**官方 npm registry**。**不**使用 NodeSource 第三方 apt 源，
> **不**使用 `curl | sh` 一键脚本。

## 路线图（Roadmap）

项目完整规划与分阶段实施见 `docs/plan.md`（德扑游戏 + 扑克 AI + Steam CV bot 的 Python
重写）；**当前进度与下一步**记在该文件顶部。
