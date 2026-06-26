# holdem-lab 路线图：德扑游戏 + AI + Steam CV Bot

> 项目规范化路线图（canonical roadmap），从初期规划固化进仓库，供容器内任意 agent
> （Claude / Codex）读取。规则类约束见 `AGENTS.md`。

## 当前进度（截至 2026-06-26 阶段 4 最小 CV/OCR 识别器）

**已完成**
- Dev container（tier ① 无头核心）已 build 并在容器内验证通过：`ubuntu:24.04` + Ubuntu apt
  Python 3.12 + uv + Node 24（全部官方源 + 校验）+ Claude/Codex CLI。
- `scripts/dev/`（`venv-dir` / `check-venv` / `setup-dev-env` / `verify-dev-env` / `py` /
  `claude-shell`）已迁移并适配为 Python-at-root；`postCreate` 跑 `scripts/dev/setup-dev-env.sh`。
- `pyproject.toml`（uv，非包；dev: ruff/pytest/mypy，已排除 legacy）、`AGENTS.md`
  （官方下载源规则）已就位。
- 阶段 0 完成：`rust/` + `web/` 已整体 `git mv` 到 `equity-calculator/`；Pages workflow、
  `.gitignore`、验证脚本路径已跟随更新；根 `pyproject.toml` 已转 uv workspace，成员为
  `common/ engine/ ai/ game/ bot/`。
- `common/` 已落地共享 `GameState` / `Action` / `Card` / `Street` / `PlayerState` / `Pot`
  dataclass，作为 engine / game / ai / bot 的通用接口。
- 阶段 1 engine 基础闭环：`engine/` 已引入 PokerKit（官方 PyPI，锁文件记录哈希），提供
  `PokerKitFacade`、`HoldemEnv`、PokerKit → `GameState` 转换；支持固定 deck 测试入口、
  单座位隐藏观测、no-limit bet/raise-to 金额语义。
- 阶段 1 adapter 边界：已引入 RLCard 与 OpenSpiel（均来自官方 PyPI，锁文件记录哈希），
  `engine/holdem_engine/adapters/{rlcard.py,openspiel.py}` 已提供 `GameState` → 训练/求解观测、
  合法动作 id、动作 id → `Action` 的映射；OpenSpiel 后端验证到 `universal_poker` 可用。
- 阶段 3 最小 AI 入口：`ai/` 已提供 `holdem_ai.decide(state) -> Action` 与确定性启发式
  `HeuristicPolicy`；当前不新增评估器/训练依赖，先让 game/bot 共用同一个决策入口。
- 阶段 2 最小 pygame 牌桌：`game/` 已引入 pygame（官方 PyPI，锁文件记录哈希），提供
  `uv run holdem-game` / `python -m holdem_game` 启动入口；可渲染座位、手牌/暗牌、公共牌、
  筹码、底池与动作按钮，人类座位直连 `HoldemEnv.step()`，AI 座位调用 `holdem_ai.decide()`。
- 阶段 4 bot 骨架：`bot/` 已提供 `Capture` / `Recognizer` / `Automator` / `BotOrchestrator`
  接口；已实现无截图依赖的 in-process `GameState` 适配器，用于先验证
  capture → recognize → `ai.decide()` → automate 的控制闭环。
- 阶段 4 in-process bot 闭环：`game/` 已接入 bot pipeline，`uv run holdem-game --bot-seat 0`
  可让 bot 控制原本人类座位；该路径仍然不加截图/CV 依赖，先验证同一个
  `GameState` / `ai.decide()` / `Action` 控制面。
- 阶段 4 视觉 fixture：`bot/vision/annotations.py` 已定义截图标注 JSON schema；`game/` 已提供
  `uv run holdem-game-capture-fixture --output-dir artifacts/vision-fixtures`，可生成自家 pygame
  牌桌 PNG + JSON 标注，用于后续 CV/OCR 识别准确率评估。
- 阶段 4 视觉评估口径：`bot/vision/recognition.py` 已定义 `RecognizedTable` 中间结果、
  annotation oracle recognizer 与 `evaluate_recognition()`；后续真实 CV/OCR 只要输出同一结构，
  即可按 cards/buttons/chips/seats/table 分组统计准确率。
- 阶段 4 最小 CV/OCR 识别器：已引入 `opencv-python` / `pytesseract`（官方 PyPI，锁文件记录哈希）
  与 `tesseract-ocr` / `libgl1`（Ubuntu 官方 apt 源，已写入 devcontainer Dockerfile）；
  `bot/vision/roi_ocr.py` 可基于 fixture ROI 做 OpenCV 裁剪预处理、Tesseract OCR，并用
  `evaluate_recognition()` 统计识别准确率。
- 规则测试已覆盖：盲注、下注轮推进、全员弃牌终局、heads-up all-in 自动 runout、边池数学、
  摊牌平分、边池派奖、筹码守恒。
- 当前验证：`scripts/dev/verify-dev-env.sh` 通过（CV/OCR runtime、ruff format/check、mypy、pytest，
  47 tests）。
- 历史提交：`ea3dace`（dev container + AGENTS.md）、`086682e`（scripts/dev）、`7eb9f48`、
  `0a79c5c`、`c93b1bd`、`d90410a`、`f6090e8`、`560386d`、`d903102`、`380f477`、
  `d20669b`、`cabe333`。尚未 push。

**下一步：提高自家 pygame fixture 识别准确率**
- 先补牌面模板匹配，替代 card OCR 对花色的误读；再补按钮/数字 OCR 的预处理和评价报告 CLI。
- 暂不把 LLM/VLM 放进主链路；后续作为低置信度兜底或冷启动标注辅助。

**环境备注（容器内）**
- 你在 Linux devcontainer 里，用户 `node`，`UV_PROJECT_ENVIRONMENT=/workspace/.venv-docker`。
- 跑 Python 用 `scripts/dev/py …` 或 `uv run …`；验证用 `scripts/dev/verify-dev-env.sh`。
- 只有 `game`（手玩）和真·Poker Legends bot 在 macOS host 跑，其余都在容器内。

---

## Context（背景）

现有 `holdem-lab` 只是一个德扑 equity 计算器（Rust core + WASM + Tauri + React），
在整体目标里占比很小。本次要在**同一仓库**（保留名字）里，基本**从零**构建三块大功能：

1. 一个可玩的德州扑克游戏；
2. 用 AI 玩这个本地游戏；
3. 用 AI（画面识别 + 自动化）玩 Steam 上的扑克游戏。

现有计算器先**归档到一边**，不约束新设计；日后或可作为加速器（用 PyO3 把 Rust 手牌评估
包给 Python 提速），但当前不依赖它。核心洞察：**引擎产出的 `GameState` 是通用接口**——本地
游戏和 Steam bot 都把状态喂给同一个 `ai.decide(state) -> action`，bot 只是“用 CV 把屏幕翻成
同样的状态”。AI 只写一次，两处复用。

## 已确定的技术选型（来自与用户确认）

- 归档：`rust/` + `web/` → `equity-calculator/` 子目录（不叫 legacy）。
- 主语言：Python；`uv` 管理的单仓工作区。
- 引擎：自研**薄 facade**（掌控我们自己的 `GameState`），底层包 PokerKit（全规则），
  并提供适配器接 OpenSpiel / RLCard。
- 游戏 UI：pygame（在 host 运行）。
- AI：借力 OpenSpiel / RLCard，启发式 → CFR → 自博弈，分阶段。
- Bot：Python（host 运行）；**先拿自家 pygame 游戏练管线**（Xvfb 容器），真目标
  **Poker Legends（Steam 758980）在 macOS 原生**跑。
- 开发环境：hybrid dev container（见下）。

## 目标结构

```
holdem-lab/
├── equity-calculator/      # 归档：原 rust/ + web/（整体搬入）
├── common/                 # 共享类型：GameState / Action / Card / Street
├── engine/                 # 规则权威引擎 + 多智能体环境（Python）
├── ai/                     # 决策引擎（Python）
├── game/                   # pygame 牌桌（host）
├── bot/                    # Steam CV bot（Python, host）
├── .devcontainer/          # hybrid 容器（engine/ai 用）
├── pyproject.toml          # uv workspace
└── README.md
```

## 分阶段实施

### 阶段 0 — 归档 + 脚手架
- `git mv rust/ web/` → `equity-calculator/`（整体搬：Tauri 的 `../../holdem-core` 相对路径
  不变）。更新 `.github/workflows/deploy.yml` 里 `rust/holdem-*` → `equity-calculator/rust/holdem-*`，
  或暂停该 workflow。
- 建 uv workspace：根 `pyproject.toml` + 成员 `common/ engine/ ai/ game/ bot/`。
- 工具：ruff（lint+format）、pytest、mypy（可选）。
- `common/`：`GameState`、`Action`、`Card`、`Street` 等 dataclass——全项目的“通用接口”。
- `.devcontainer/`（见“开发环境”）。

### 阶段 1 — engine/（先有一局能跑的德扑）
- 包 **PokerKit** 实现全规则：盲注、下注轮、合法动作、边池、摊牌、定胜负；对外只暴露
  我们自己的 `GameState` 与 `step(action)`。手牌评估用 PokerKit 自带（或 `eval7`/`treys`/
  `phevaluator` 备选）。
- 环境 API：Gym/PettingZoo 风格 `reset / step / legal_actions / observe`，供 AI 自博弈。
- 适配器：`engine/holdem_engine/adapters/{openspiel.py,rlcard.py}`（GameState ↔ 各库格式）。
- 测试：边池数学、摊牌平分、筹码守恒（property test）。
- 代表文件：`engine/holdem_engine/{state.py,facade.py,env.py}`。

### 阶段 2 — game/（pygame，host 运行）
- 牌桌：座位、手牌/公共牌、筹码/底池、动作按钮、简单动画。
- 座位可设 人/AI；AI 座位调用 `ai.decide()`，同进程直连 engine。
- 兼作 AI 自博弈“观战”可视化。
- 代表文件：`game/holdem_game/{app.py,table_view.py,widgets.py,assets/}`。

### 阶段 3 — ai/（决策引擎，可与阶段 2 并行）
- API：`decide(state: GameState) -> Action`。
- A 启发式：Monte-Carlo equity + 底池赔率 + 位置 + 听牌 outs → 动作 + 下注量
  （快速上线，驱动游戏 AI 座位与首个 bot）。
- B preflop GTO：push/fold 与开池范围（查表或 CFR 解）。
- C CFR/CFR+：经 OpenSpiel 在抽象博弈上求解；或经 RLCard 做 deep-CFR/自博弈。
- 评估：AI vs AI 的 bb/100；OpenSpiel best-response 算 exploitability。
- 代表文件：`ai/holdem_ai/{api.py,heuristic.py,preflop.py,cfr.py,eval.py}`。

### 阶段 4 — bot/（Steam CV bot，host 运行）
- 抽象层：`Capture`、`Recognizer`（像素→GameState）、`Automator`——平台/游戏/后端皆可替换。
  `Capture`/`Automator` 两套后端：**xvfb 容器**（打自家游戏、可 CI、确定性）/ **host 原生**
  （mss + pyautogui 打真实屏幕）；`Recognizer` 与 `ai.decide()` 两边共用同一份代码。
- 识别默认：卡牌模板匹配 + 数字 OCR（Tesseract/PaddleOCR）+ 按钮检测；VLM（复用现有
  Qwen-VL/豆包）做兜底与冷启动。
- 编排：轮到我方 → 读屏 → `ai.decide()` → 操作；安全：急停热键、仅在我方回合操作、
  拟人随机延时。
- 策略：**先对自家 pygame 游戏**跑通全链路（Xvfb 容器内、可控/可 CI），再写
  `bot/holdem_bot/adapters/poker_legends.py` 对接真目标（**Poker Legends 758980**，macOS 原生）。
- 合规：Poker Legends 是 play-money 社交扑克（非真钱），属个人自动化/学习；注意其 ToS 可能
  禁止自动化，仅供本地学习用途。
- 代表文件：`bot/holdem_bot/{capture.py,recognize/,automate.py,orchestrator.py}`。

## 开发环境（三层 hybrid，融合两个参考项目）

参考 quant（无头核心）+ web3-tycoon（Xvfb + noVNC 的 GUI/测试容器），分三层：

- **① 无头核心容器（engine + ai）** —— **骨架两项目通用**（同作者：Claude/Codex 挂载、
  firewall、runArgs、zsh 都一致）。**base 取 web3-tycoon 的 `ubuntu:24.04`**（比 quant 的
  `node:20` 更适合 Python+图形项目，且与 ② 共用 base、apt 包好叠加），**Python 工具链照搬
  quant**：uv + 固定 Python 3.12 + 双 venv；Node 仅为装 Claude/Codex CLI 而保留。负责
  训练/求解/测试/CI，并把 **OpenSpiel 的 C++ 依赖在镜像里一次构建好**。Rust 精简（仅日后做
  PyO3 加速器时再装），加 `cmake build-essential` 等。
- **② GUI / “bot 打自家游戏”测试容器（game + 阶段 4 练管线）** —— 借用 web3-tycoon 的
  **Xvfb + x11vnc + websockify + noVNC + fluxbox** 方案：pygame 渲染到 `DISPLAY=:0`，浏览器开
  `localhost:7900` 即可观看。关键收益：**阶段 4“先拿自家游戏练管线”可整套塞进同一个 Xvfb 容器**
  —— pygame 游戏 + 截图（mss 抓 Xvfb 帧）+ OCR + 注入式点击（Xlib/pyautogui 打同一帧）在容器内
  闭环，确定性、可 CI、免 host 权限。镜像 `FROM` ① 再叠加图形层，加
  `xvfb x11vnc websockify novnc fluxbox libgl1 libsdl2* tesseract-ocr`。（人要“手玩”想更跟手时，
  game 也可直接在 host 原生跑。）
- **③ 真·Steam bot（host 原生，macOS 先行）** —— 目标游戏 **Poker Legends（Steam 758980）
  有 Mac/Win 版、无 Linux 版**，所以**直接在 macOS 开发机原生跑整套**（截图+CV+自动化+游戏同机），
  无需 Windows 机/VM；进不了容器（游戏无 Linux 版，Xvfb 也帮不上）。macOS 需授予**屏幕录制 +
  辅助功能(Accessibility)** 权限；capture 用 Quartz `CGWindowListCreateImage` 或 mss 抓游戏窗口，
  automate 用 pyautogui/pynput。Windows 版日后可作第二目标。`Capture`/`Automator` 的
  **xvfb（练自家游戏）/ macOS-host（打真游戏）双后端**，识别与决策代码两边复用。
- **venv**：沿用 quant 的双 venv（`.venv-docker` / `.venv-mac`），同一 uv 工作区两处各用。
- **GPU**：启发式 + tabular CFR 用 CPU 足够；deep-CFR/RL（C 阶段）再定（Linux+NVIDIA host
  `--gpus all`；macOS MPS 只能 host 侧）。

## 验证

- engine：`pytest`（规则/边池/摊牌/筹码守恒 property test）。
- ai：评估脚本跑 AI vs random / vs 启发式 的 bb/100；CFR 用 OpenSpiel 算 exploitability。
- game：host 上手玩一局 人 vs AI。
- bot：对自家 pygame 游戏跑 读屏 → 决策 → 点击 闭环，统计识别准确率，再迁 Steam。

## 已确认 / 待后续

- **目标游戏已定**：Poker Legends（Steam 758980），有 Mac/Win 版、无 Linux 版 → 阶段 4 在
  **macOS 开发机原生**跑，首个 `adapters/poker_legends.py`。
- 待办（不阻塞阶段 0–3）：进入阶段 4 前采集游戏窗口的牌/按钮/数字布局（模板 + OCR 区域）；
  Windows 版作为可选第二目标。
