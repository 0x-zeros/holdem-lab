# Equity Calculator Web Application

A web-based Texas Hold'em equity calculator built with React and WebAssembly.

## Architecture

```
web/
└── frontend/         # React frontend
    ├── src/
    │   ├── api/      # API abstraction layer
    │   │   ├── client.ts   # Unified API client
    │   │   ├── wasm.ts     # WASM implementation
    │   │   ├── tauri.ts    # Tauri IPC implementation
    │   │   └── types.ts    # Shared types
    │   ├── components/
    │   ├── store/    # Zustand state
    │   └── wasm/     # Compiled WASM module
    └── ...
```

## Deployment Modes

| Mode | Description | Command |
|------|-------------|---------|
| Web (WASM) | Pure frontend, static hosting | `npm run build` |
| Desktop (Tauri) | Native app with Rust backend | `npm run tauri:build` |

## Quick Start

### Prerequisites

Build the WASM module first:
```bash
cd rust/holdem-wasm
wasm-pack build --target web --out-dir ../../web/frontend/src/wasm
```

### Development

```bash
cd web/frontend
npm install
npm run dev
```

Open http://localhost:5173

### Production Build

```bash
cd web/frontend
npm run build
```

The static files in `dist/` can be deployed to any static hosting service.

## Tech Stack

**Computation:**
- Rust (holdem-core library)
- WebAssembly (browser)
- Tauri (desktop)

**Frontend:**
- React 18
- TypeScript
- Tailwind CSS 4
- Zustand (state)
- TanStack Query (async state)

## Testing

```bash
# Rust core tests
cd rust/holdem-core
cargo test

# Frontend type check
cd web/frontend
npm run typecheck
```
