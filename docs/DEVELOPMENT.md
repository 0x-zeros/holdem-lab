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

计算 169 种规范起手牌的翻前胜率，生成 JSON 数据文件：

```bash
cd rust/holdem-core

# 快速测试 (10万次模拟，仅2人桌)
cargo run --release --bin precompute -- -s 100000 -p 2

# 完整预计算 (100万次模拟，2-10人桌全部)
cargo run --release --bin precompute -- -o preflop-equity.json

# 指定模拟次数和输出路径
cargo run --release --bin precompute -- -s 500000 -o ../data/equity.json
```

### 参数说明

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--simulations` | `-s` | 1,000,000 | 每手牌模拟次数 |
| `--players` | `-p` | 2-10 全部 | 只计算指定玩家数 |
| `--output` | `-o` | stdout | 输出文件路径 |
| `--help` | `-h` | - | 显示帮助信息 |

### 进度输出示例

```
========================================
Preflop Equity Precompute
Simulations per hand: 1,000,000
Players: 2-10 (all)
========================================

[2 players]
[  1/169] AA   ...  85.2%  (1.2s)
[  2/169] KK   ...  82.1%  (1.1s)
...
[169/169] 32o  ...  31.2%  (1.0s)
Subtotal: 3m 12s

[3 players]
[  1/169] AA   ...  73.1%  (1.3s)
...

========================================
Done! Total time: 28m 48s
Output: preflop-equity.json
========================================
```

### 输出 JSON 格式

```json
{
  "2": { "AA": 85.2, "KK": 82.1, "QQ": 79.9, ... },
  "3": { "AA": 73.1, "KK": 68.5, "QQ": 65.2, ... },
  ...
  "10": { "AA": 31.1, "KK": 26.0, ... }
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
