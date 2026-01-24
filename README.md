# holdem-lab

德州扑克概率计算库与游戏引擎，用于分析和验证。

## 功能特性

- **概率计算器**: Monte Carlo 模拟计算手牌胜率
- **手牌评估器**: 7 张牌选 5 张最优组合评估
- **游戏引擎**: 完整的牌局状态机
- **事件日志**: 记录和回放牌局
- **分析工具**: Jupyter notebooks 胜率分析

## 安装

```bash
cd holdem-core
uv pip install -e ".[dev]"
```

安装分析依赖：

```bash
uv pip install -e ".[analysis]"
```

## 分析 Notebook

```bash
uv run jupyter lab analysis/
```

## 快速开始

```python
from holdem_lab import (
    parse_cards,
    EquityRequest, PlayerHand, calculate_equity,
    GameState,
)

# 计算 AA vs KK 胜率
request = EquityRequest(
    players=[
        PlayerHand(hole_cards=tuple(parse_cards("Ah Ad"))),
        PlayerHand(hole_cards=tuple(parse_cards("Kh Kd"))),
    ],
    num_simulations=10000,
)
result = calculate_equity(request)
print(f"AA: {result.players[0].equity:.1%}")
print(f"KK: {result.players[1].equity:.1%}")

# 运行完整牌局
game = GameState(num_players=2, seed=42)
result = game.run_to_showdown()
print(f"赢家: 玩家 {result.winners[0]}")
```

## 运行测试

```bash
cd holdem-core
uv run pytest
```

## 许可证

MIT
