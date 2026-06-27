# 本地 pygame 试玩

本地游戏用于验证 engine、`GameState` 和 `holdem_ai` 共用决策入口。普通试玩不需要
Poker Legends、截图识别或 Steam 环境。

## 一人对 AI

推荐先打一对一：

```bash
uv run holdem-game-heads-up
```

等价命令：

```bash
uv run holdem-game --heads-up
```

默认你控制 seat 0，对面 seat 1 由 `holdem_ai` 自动决策。

默认牌局是 no-limit holdem，最小级别为小盲 `5` / 大盲 `10`，默认买入为 `100BB`
（即 `1000` 筹码）。

## 多人桌

默认启动是 3 人桌：

```bash
uv run holdem-game
```

也可以指定人数、座位、筹码和盲注：

```bash
uv run holdem-game --players 4 --human-seat 2 --starting-stack 500 --small-blind 5 --big-blind 10
```

除 `--human-seat` 之外的普通座位都会由本地 AI 自动行动。

也可以用 stake level 从 `5/10` 开始按层翻倍：

```bash
uv run holdem-game-heads-up --stake-level 2   # 10/20，默认 2000 筹码
uv run holdem-game-heads-up --stake-level 3   # 20/40，默认 4000 筹码
```

## 游戏节奏

牌桌顶部会显示当前轮到的座位和最大等待倒计时。默认本地 AI 每次行动前等待 `700ms`，
让一人对 AI 试玩更接近正常游戏节奏：

```bash
uv run holdem-game-heads-up --ai-delay-ms 900 --turn-timeout-sec 30
```

`--ai-delay-ms 0` 可恢复测试用的近即时 AI 行动；`--turn-timeout-sec` 目前只控制界面显示，
不会自动替玩家操作。

## AI profile

本地游戏和 AI 评估器共用同一组 profile：

```bash
uv run holdem-game-heads-up --ai-profile tight
uv run holdem-game-heads-up --ai-profile loose
uv run holdem-game-heads-up --ai-profile no_equity
```

可选值：

- `current`：当前默认启发式策略。
- `tight`：更紧，继续和下注阈值更高。
- `loose`：更松，更愿意跟注和下注。
- `no_equity`：关闭 rollout equity，用于对比评估。

## 操作

- 鼠标点击底部动作按钮。
- `F` fold。
- `C` 或空格 call/check。
- `B` 或 `R` bet/raise。
- 有 bet/raise 时可直接输入数字作为总下注额。
- Backspace/Delete 编辑下注额，Up/Down 按 big blind 调整，Enter 提交。
- `P` 暂停/恢复 AI 自动推进。
- 一手结束后会停在结算面板，显示赢家、payoff 和摊牌牌型。
- `N` 或点击 `Next hand` 开下一手。
- Escape 退出。

说明：`GameState.legal_actions` 仍只在需要跟注时暴露 Fold，避免 AI 在可以免费 Check 时学到
无意义弃牌；pygame 人类 UI 会额外保留 Fold 按钮，用来模拟商业牌局的快捷操作。

## bot-seat 说明

`--bot-seat` 不是普通 AI 对手开关；它用于测试 bot 的
Capture/Recognizer/Automator pipeline 接管某个座位。正常手玩时不要加 `--bot-seat`。
