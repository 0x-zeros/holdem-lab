# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

### Rust 核心库

```bash
cd rust/holdem-core
cargo build
cargo test
```

### Web 前端

```bash
cd web/frontend
npm install
npm run dev          # 开发服务器 → localhost:5173
npm run build        # 构建生产版本
npm run typecheck    # TypeScript 类型检查

# 本地开发需要先构建 WASM
cd ../../rust/holdem-wasm
wasm-pack build --target web --out-dir ../../web/frontend/src/wasm
```

### Tauri 桌面应用

```bash
cd rust/holdem-app
npm install
npm run tauri:dev    # 开发模式
npm run tauri:build  # 构建发布包
```

## 架构概览

德州扑克概率计算工具，采用纯 Rust 实现：

```
rust/holdem-core/        # Rust 核心库
├── card.rs              # Card, Deck, Rank, Suit
├── evaluator.rs         # 手牌评估 (7选5最优)
├── equity.rs            # Monte Carlo 胜率计算
├── draws.rs             # 听牌分析
├── canonize.rs          # 169 规范手牌
└── error.rs             # 错误类型

rust/holdem-wasm/        # WASM 绑定
└── src/lib.rs           # wasm-bindgen 导出

web/frontend/            # React 前端 (TypeScript)
└── src/api/             # API 抽象层 (WASM 或 Tauri)

rust/holdem-app/         # Tauri 桌面应用
└── src-tauri/           # Rust 命令
```

### 关键设计

- 评估器使用枚举 C(7,5)=21 组合选最优，非查表法
- A-2-3-4-5 (wheel) 是有效顺子，顶牌为 5
- 所有 Card 类型是不可变的，可 hash
- Web 版使用 WASM，无需后端服务器
- 桌面版使用 Tauri，直接调用 Rust 代码

### 部署模式

| 模式 | 说明 | 命令 |
|------|------|------|
| Web (WASM) | 纯前端，适合静态托管 | `npm run build` |
| Desktop (Tauri) | 原生应用 | `npm run tauri:build` |

### 运行测试

```bash
# Rust 核心库测试
cd rust/holdem-core && cargo test

# 前端类型检查
cd web/frontend && npm run typecheck
```
