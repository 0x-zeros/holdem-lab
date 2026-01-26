# holdem-lab 开发文档

## 项目架构

```
holdem-lab/
├── holdem-core/          # Python 核心库
│   ├── src/holdem_lab/   # 概率计算、手牌评估、游戏引擎
│   └── tests/
│
├── web/                  # Web 应用
│   ├── backend/          # FastAPI 后端 (Python)
│   └── frontend/         # React 前端 (TypeScript)
│
├── rust/                 # Rust 版本
│   ├── holdem-core/      # Rust 核心库
│   └── holdem-app/       # Tauri 桌面应用
│
├── analysis/             # Jupyter 分析 notebooks
└── design/               # 设计文件
```

---

## 快速开始

### Python 核心库

```bash
cd holdem-core
uv pip install -e ".[dev]"
```

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

### 分析 Notebooks

```bash
uv pip install -e ".[analysis]"
uv run jupyter lab analysis/
```

---

## Web 应用开发

### 启动开发服务

```bash
# 后端 (Python/FastAPI)
cd web/backend
python run.py              # → localhost:8000

# 前端 (React/Vite)
cd web/frontend
npm install
npm run dev                # → localhost:5173
```

### 端口配置

默认端口：
- Frontend: `5173`
- Backend: `8000`

**端口冲突时**，使用环境变量：

```bash
# 方法 1: 修改 .env 文件
# web/frontend/.env
VITE_FRONTEND_PORT=3000
VITE_BACKEND_PORT=9000

# 方法 2: 命令行环境变量
VITE_FRONTEND_PORT=3000 npm run dev
EQUITY_PORT=9000 python run.py
```

### 环境变量参考

| 变量 | 服务 | 默认值 | 说明 |
|------|------|--------|------|
| `VITE_FRONTEND_PORT` | Frontend | 5173 | Vite 开发服务器端口 |
| `VITE_BACKEND_PORT` | Frontend | 8000 | API 代理目标端口 |
| `EQUITY_PORT` | Backend | 8000 | FastAPI 服务端口 |
| `EQUITY_FRONTEND_PORT` | Backend | 5173 | CORS 允许的前端端口 |

### Docker 开发

```bash
cd web

# 默认端口
docker compose -f docker-compose.dev.yml up

# 自定义端口
FRONTEND_PORT=3000 BACKEND_PORT=9000 docker compose -f docker-compose.dev.yml up
```

---

## Tauri 桌面应用

Tauri 应用复用 `web/frontend` 前端，使用 Rust 实现后端。

### 开发模式

```bash
cd rust/holdem-app
npm install
npm run tauri:dev
```

### 构建发布包

```bash
npm run tauri:build
```

生成的安装包位于：
- Windows: `rust/holdem-app/src-tauri/target/release/bundle/msi/`
- macOS: `rust/holdem-app/src-tauri/target/release/bundle/dmg/`

### 运行已编译的程序

```bash
# Windows
./rust/holdem-app/src-tauri/target/release/holdem-app.exe

# macOS/Linux
./rust/holdem-app/src-tauri/target/release/holdem-app
```

---

## 测试

```bash
# Python 核心库
cd holdem-core
uv run pytest

# Web 后端
cd web/backend
uv run pytest

# Rust 核心库
cd rust/holdem-core
cargo test

# 前端类型检查
cd web/frontend
npm run typecheck
```

---

## 发布

### Web 应用 (Docker)

```bash
cd web
docker compose up -d
```

访问地址：
- Frontend: http://localhost (Nginx 反向代理)
- API: http://localhost/api

### Tauri 桌面应用

```bash
cd rust/holdem-app
npm run tauri:build
```

---

## 技术栈

### Python 核心库

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.12+ | 运行时 |
| pytest | 8.x | 测试 |

### Web 后端

| 技术 | 版本 | 用途 |
|------|------|------|
| FastAPI | 0.115.x | Web 框架 |
| Pydantic | 2.10.x | 数据验证 |
| uvicorn | 0.34.x | ASGI 服务器 |

### Web 前端

| 技术 | 版本 | 用途 |
|------|------|------|
| React | 18.x | UI 框架 |
| TypeScript | 5.7.x | 类型安全 |
| Vite | 6.x | 构建工具 |
| Tailwind CSS | 3.x | 样式 |
| Zustand | 5.x | 状态管理 |
| TanStack Query | 5.x | 服务端状态 |

### Rust / Tauri

| 技术 | 版本 | 用途 |
|------|------|------|
| Rust | 1.93+ | 语言 |
| Tauri | 2.x | 桌面框架 |
| serde | 1.0 | 序列化 |
| rand | 0.9 | 随机数 |
