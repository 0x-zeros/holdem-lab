# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
# 安装开发依赖
cd holdem-core
uv pip install -e ".[dev]"

# 运行所有测试
uv run pytest

# 运行单个测试文件
uv run pytest tests/test_evaluator.py

# 运行单个测试用例
uv run pytest tests/test_evaluator.py::TestEvaluateFive::test_royal_flush -v

# 安装分析依赖（Jupyter notebooks）
uv pip install -e ".[analysis]"
```

## 架构概览

德州扑克概率计算库，采用分层架构：

```
cards.py (基础层，无依赖)
    │
    ├── evaluator.py (手牌评估)
    ├── event_log.py (事件日志)
    └── equity.py (概率计算)
            │
            └── game_state.py (状态机，整合所有模块)
```

### 模块依赖关系

- **cards.py**: 基础模块，定义 `Card`, `Deck`, `Rank`, `Suit`，以及解析函数
- **evaluator.py**: 依赖 cards，实现 7 选 5 最优手牌评估
- **equity.py**: 依赖 cards + evaluator，Monte Carlo 概率计算
- **event_log.py**: 依赖 cards，事件记录与回放
- **game_state.py**: 整合所有模块，提供完整牌局状态机

### 关键设计

- 评估器使用枚举 C(7,5)=21 组合选最优，非查表法
- A-2-3-4-5 (wheel) 是有效顺子，顶牌为 5
- `EventLog` 实现了 `__len__`，判断非空时需用 `is not None` 而非 truthy 检查
- 所有 Card 对象是 frozen dataclass，可 hash
