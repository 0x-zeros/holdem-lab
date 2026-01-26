import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import wasm from 'vite-plugin-wasm'
import topLevelAwait from 'vite-plugin-top-level-await'
import path from 'path'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')

  const FRONTEND_PORT = parseInt(env.VITE_FRONTEND_PORT || '5173')
  const BACKEND_PORT = parseInt(env.VITE_BACKEND_PORT || '8000')
  const USE_WASM = env.VITE_USE_WASM === 'true'

  return {
    plugins: [react(), wasm(), topLevelAwait()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    base: env.BASE_URL || '/',
    server: {
      port: FRONTEND_PORT,
      // Only enable proxy when not in WASM mode
      proxy: USE_WASM
        ? undefined
        : {
            '/api': {
              target: `http://localhost:${BACKEND_PORT}`,
              changeOrigin: true,
            },
          },
    },
    build: {
      target: 'esnext',
    },
    optimizeDeps: {
      exclude: ['holdem-wasm'],
    },
  }
})
