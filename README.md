# holdem-lab

`holdem-lab` 是一个 Python 德州扑克实验仓库，目标是把同一套规则状态接口用于三件事：

1. 可手玩的本地德州扑克游戏；
2. 用 AI 玩本地游戏；
3. 用画面识别和自动化去玩 Steam 的 Poker Legends。

核心接口是 `GameState`：`engine` 产出统一状态，本地 `game` 和 Steam `bot` 都把状态交给同一个
`holdem_ai.decide(state) -> Action`。Bot 不重新写策略，只负责把屏幕翻译成同样的状态。

## 当前状态

- Python + `uv` 单仓 workspace 已就绪。
- `common/` 提供 `GameState`、`Action`、`Card`、`Street` 等共享 dataclass。
- `engine/` 已有 PokerKit facade、`HoldemEnv`、边池/摊牌/筹码守恒测试，以及 RLCard/OpenSpiel
  适配器边界。
- `ai/` 已有可解释启发式策略、轻量 rollout equity、profile 评估矩阵。
- `game/` 已有 pygame 牌桌，可一人对 AI 连续试玩。
- `bot/` 已有 Capture/Recognizer/Automator 编排、安全闸门、Poker Legends 离线视频/截图识别原型和
  macOS host dry-run 骨架。

完整路线图与最新进度见 [`docs/plan.md`](docs/plan.md)。

## 架构

```text
holdem-lab/
├── common/              # 全项目共享类型：GameState / Action / Card / Street
├── engine/              # 德扑规则引擎 facade + HoldemEnv + 训练/求解适配器
├── ai/                  # 决策策略：启发式、profile、评估入口，后续 CFR/RL
├── game/                # pygame 本地牌桌，手玩和 AI 自博弈可视化
├── bot/                 # 截图识别、安全闸门、自动化编排、Poker Legends 适配
├── scripts/dev/         # devcontainer/macOS 双 venv 辅助脚本
├── docs/                # 路线图、试玩说明、开发记录
└── pyproject.toml       # uv workspace 根配置
```

数据流：

```text
本地游戏: engine/HoldemEnv -> GameState -> ai.decide() -> engine.step()

Steam bot: 屏幕截图 -> bot recognizer -> GameState -> ai.decide() -> automator
```

## 运行环境

仓库同时支持 Linux dev container 和 macOS host，共用代码但使用不同虚拟环境：

- Linux dev container：`.venv-docker`，用于 engine、ai、bot 离线识别、测试和评估。
- macOS host：`.venv-mac`，用于手玩 pygame 窗口，以及后续真实 Poker Legends 截图/自动化。

不要使用裸 `.venv`。跑 Python 用：

```bash
uv run ...
# 或
scripts/dev/py ...
```

如果在 macOS host 上首次运行：

```bash
export UV_PROJECT_ENVIRONMENT="$(scripts/dev/venv-dir.sh)"
scripts/dev/setup-dev-env.sh
```

依赖和下载来源规则见 [`AGENTS.md`](AGENTS.md)：所有外部产物只从官方/权威来源获取；官方提供校验时
必须校验；禁止第三方镜像和 `curl | sh` 一键脚本。

## 本地游戏

一人对 AI：

```bash
uv run holdem-game-heads-up
```

默认是 no-limit holdem，最小级别为 `5/10`，默认买入为 `100BB`（`1000` 筹码）。

等价命令：

```bash
uv run holdem-game --heads-up
```

默认 3 人桌：

```bash
uv run holdem-game
```

指定桌面参数：

```bash
uv run holdem-game --players 4 --human-seat 2 --starting-stack 500 --small-blind 5 --big-blind 10
```

从 `5/10` 开始按层翻倍：

```bash
uv run holdem-game-heads-up --stake-level 2   # 10/20，默认 2000 筹码
uv run holdem-game-heads-up --stake-level 3   # 20/40，默认 4000 筹码
```

选择 AI profile：

```bash
uv run holdem-game-heads-up --ai-profile tight
uv run holdem-game-heads-up --ai-profile loose
uv run holdem-game-heads-up --ai-profile no_equity
```

可选 profile：

- `current`：当前默认启发式策略。
- `tight`：更紧，下注/继续阈值更高。
- `loose`：更松，更愿意跟注和下注。
- `no_equity`：关闭 rollout equity，用于对比。

说明：`--bot-seat` 不是普通 AI 对手开关，而是测试 bot pipeline 接管座位的入口。正常手玩不要加
`--bot-seat`。

更多操作说明见 [`docs/local-game.md`](docs/local-game.md)。

一手结束后游戏会停在结算面板，显示赢家、你的本手输赢、各座位 payoff 和摊牌牌型；按 `N` 或点击
`Next hand` 才进入下一手。

## AI 评估

两组 profile heads-up 对战：

```bash
uv run holdem-ai-evaluate-heads-up --profile-a current --profile-b tight --hands 100 --seed 1
```

profile 矩阵评估：

```bash
uv run holdem-ai-evaluate-heads-up --matrix current no_equity tight loose --hands 100 --seed 1
```

输出为 JSON，包含 chips、bb/100、胜负、动作频率和策略理由频率。

## Bot / Poker Legends

真目标是 Steam Poker Legends（appid `758980`）。该游戏没有 Linux 版，所以真实游戏窗口截图和自动化
在 macOS host 原生运行。当前默认仍是 fail-closed：只有 screen state、牌面、按钮、筹码和座位信息
都满足安全条件，才会把 `GameState` 交给 AI；否则停在 `no_game_state` / `low_confidence` 等原因。

当前 bot 侧主要入口包括：

```bash
uv run holdem-bot-ingest-video <video> --out <dir>
uv run holdem-bot-select-keyframes <manifest.json> --out <dir> --select ...
uv run holdem-bot-llm-annotate <annotations...> --image-root <dir> --out <dir>
uv run holdem-bot-run-poker-legends-dry-run --image <png> --out <jsonl>
```

具体阶段、产物位置和安全口径见 [`docs/plan.md`](docs/plan.md)。

## Dev Container

在 VS Code 中使用 **Dev Containers: Reopen in Container** 即可进入 Linux 开发环境。

常用命令：

```bash
scripts/dev/setup-dev-env.sh
scripts/dev/verify-dev-env.sh
scripts/dev/py -m pytest
```

当前容器是无头核心环境，不直接弹出 pygame 手玩窗口。可以在容器内跑：

```bash
uv run holdem-game-heads-up --help
uv run holdem-game-capture-fixture --output-dir artifacts/vision-fixtures
uv run holdem-ai-evaluate-heads-up --matrix current tight loose --hands 20 --seed 1
```

手玩 `game` 和真实 Poker Legends bot 在 macOS host 跑。

## 验证

提交前运行：

```bash
scripts/dev/verify-dev-env.sh
```

该脚本会检查 venv 使用、CV/OCR runtime、ruff format/lint、mypy 和 pytest。

## 许可证

MIT
