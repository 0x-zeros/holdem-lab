# holdem-lab 开发文档

## 项目架构

```
holdem-lab/
├── rust/                 # Rust 实现
│   ├── holdem-core/      # 核心库 (概率计算、手牌评估)
│   ├── holdem-wasm/      # WASM 绑定 (Web 版)
│   └── holdem-app/       # Tauri 桌面应用
│
└── web/
    └── frontend/         # React 前端 (TypeScript)
```

---

## 快速开始

### Rust 核心库

```bash
cd rust/holdem-core
cargo build
cargo test
```

### Web 前端 (WASM 模式)

```bash
# 1. 构建 WASM
cd rust/holdem-wasm
wasm-pack build --target web --out-dir ../../web/frontend/src/wasm

# 2. 启动前端
cd ../../web/frontend
npm install
npm run dev                # → localhost:5173
```

### Tauri 桌面应用

```bash
cd rust/holdem-app
npm install
npm run tauri:dev          # 开发模式
npm run tauri:build        # 构建发布包
```

---

## 预计算工具

### Preflop Equity 预计算

计算 169 种规范起手牌的翻前胜率，按玩家数生成独立 JSON 文件。

**默认启用并行模式**，自动利用所有 CPU 核心 (~10-17x 加速)。

```bash
cd rust/holdem-core

# 快速测试 (仅2人桌)
cargo run --release --bin precompute -- -s 100000 -p 2

# 完整预计算 (2-10人桌全部)
cargo run --release --bin precompute

# 自定义输出目录
cargo run --release --bin precompute -- -o ./output
```

### 参数说明

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--simulations` | `-s` | 1,000,000 | 每手牌模拟次数 |
| `--players` | `-p` | 2-10 全部 | 只计算指定玩家数 |
| `--output` | `-o` | `../../web/frontend/src/data` | 输出目录 |
| `--help` | `-h` | - | 显示帮助信息 |

### 输出文件

每个玩家数生成独立文件：

```
web/frontend/src/data/
├── preflop-equity-2.json
├── preflop-equity-3.json
├── ...
└── preflop-equity-10.json
```

### 进度输出示例

```
========================================
Preflop Equity Precompute
Simulations per hand: 100,000
Players: 2-10 (all)
Output: ../../web/frontend/src/data/preflop-equity-{N}.json
Mode: Parallel (using all CPU cores)
========================================

[2 players]
Completed in 1m 17s → Saved: ../../web/frontend/src/data/preflop-equity-2.json

[3 players]
Completed in 1m 45s → Saved: ../../web/frontend/src/data/preflop-equity-3.json
...

========================================
Done! Total time: 15m 30s
========================================
```

### JSON 格式 (每个文件)

```json
{
  "AA": 85.2,
  "KK": 82.4,
  "AKs": 67.0,
  ...
}
```

胜率值为百分比，精确到小数点后一位。

---

## 部署模式

| 模式 | 说明 | 后端 |
|------|------|------|
| Web (WASM) | 纯前端静态部署 | 无需后端，浏览器内计算 |
| Desktop (Tauri) | 原生桌面应用 | Rust 本地运行 |

### Web (WASM) 部署

适合静态托管 (GitHub Pages, Vercel, Netlify)：

```bash
cd web/frontend
npm run build
# 部署 dist/ 目录
```

### Tauri 桌面发布

```bash
cd rust/holdem-app
npm run tauri:build
```

生成的安装包位于：
- Windows: `rust/holdem-app/src-tauri/target/release/bundle/msi/`
- macOS: `rust/holdem-app/src-tauri/target/release/bundle/dmg/`

---

## 测试

```bash
# Rust 核心库
cd rust/holdem-core
cargo test

# 前端类型检查
cd web/frontend
npm run typecheck
```

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VITE_FRONTEND_PORT` | 5173 | Vite 开发服务器端口 |
| `BASE_URL` | / | 静态资源基础路径 (GitHub Pages 用) |

---

## 技术栈

### Rust 核心库

| 技术 | 版本 | 用途 |
|------|------|------|
| Rust | 1.75+ | 语言 |
| serde | 1.0 | 序列化 |
| rand | 0.9 | 随机数 |
| wasm-bindgen | 0.2 | WASM 绑定 |

### Web 前端

| 技术 | 版本 | 用途 |
|------|------|------|
| React | 18.x | UI 框架 |
| TypeScript | 5.7.x | 类型安全 |
| Vite | 6.x | 构建工具 |
| Tailwind CSS | 3.x | 样式 |
| Zustand | 5.x | 状态管理 |
| TanStack Query | 5.x | 异步状态 |

### Tauri 桌面

| 技术 | 版本 | 用途 |
|------|------|------|
| Tauri | 2.x | 桌面框架 |

---

## API 客户端

前端使用统一的 `apiClient` 抽象层，自动检测运行环境：

- **Tauri 环境**: 使用 IPC 调用 Rust 命令
- **Web 环境**: 使用 WASM 模块在浏览器内计算

```typescript
import { apiClient } from './api/client'

// 自动选择正确的后端
const result = await apiClient.calculateEquity(request)
```


---

## 预计算时间

****************************************
Preflop Equity Precompute
Simulations per hand: 1,000,000
Players: 2-10 (all)
Output: ../../web/frontend/src/data/preflop-equity-{N}.json
Mode: Parallel (using all CPU cores)
****************************************

[2 players]
Completed in 12m 29s → Saved: ../../web/frontend/src/data/preflop-equity-2.json

[3 players]
Completed in 19m 18s → Saved: ../../web/frontend/src/data/preflop-equity-3.json

[4 players]
Completed in 24m 17s → Saved: ../../web/frontend/src/data/preflop-equity-4.json
Completed in 19m 18s → Saved: ../../web/frontend/src/data/preflop-equity-3.json

[4 players]
Completed in 24m 17s → Saved: ../../web/frontend/src/data/preflop-equity-4.json

[4 players]
Completed in 24m 17s → Saved: ../../web/frontend/src/data/preflop-equity-4.json
[4 players]
Completed in 24m 17s → Saved: ../../web/frontend/src/data/preflop-equity-4.json
Completed in 24m 17s → Saved: ../../web/frontend/src/data/preflop-equity-4.json

[5 players]
Completed in 29m 48s → Saved: ../../web/frontend/src/data/preflop-equity-5.json

[6 players]
Completed in 34m 31s → Saved: ../../web/frontend/src/data/preflop-equity-6.json

[7 players]
Completed in 39m 56s → Saved: ../../web/frontend/src/data/preflop-equity-7.json

[8 players]
Completed in 46m 1s → Saved: ../../web/frontend/src/data/preflop-equity-8.json

[9 players]
Completed in 53m 49s → Saved: ../../web/frontend/src/data/preflop-equity-9.json

[10 players]
Completed in 1h 4m 44s → Saved: ../../web/frontend/src/data/preflop-equity-10.json

****************************************
Done! Total time: 5h 24m 57s
****************************************