# holdem-lab

德州扑克概率计算库与游戏引擎。

## 功能特性

- **概率计算器**: Monte Carlo 模拟计算手牌胜率
- **手牌评估器**: 7 张牌选 5 张最优组合评估
- **游戏引擎**: 完整的牌局状态机
- **事件日志**: 记录和回放牌局
- **Web 应用**: React + FastAPI 在线计算器
- **桌面应用**: Tauri + Rust 原生客户端

## 项目结构

| 目录 | 说明 |
|------|------|
| `holdem-core/` | Python 核心库 |
| `web/` | Web 应用 (React + FastAPI) |
| `rust/` | Rust 核心库 + Tauri 桌面应用 |
| `analysis/` | Jupyter 分析 notebooks |

## 开发文档

详细的安装、配置、开发和发布说明请参阅 **[开发文档](docs/DEVELOPMENT.md)**。

## 许可证

MIT
