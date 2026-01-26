import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')

  const FRONTEND_PORT = parseInt(env.VITE_FRONTEND_PORT || '5173')
  const BACKEND_PORT = parseInt(env.VITE_BACKEND_PORT || '8000')

  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      port: FRONTEND_PORT,
      proxy: {
        '/api': {
          target: `http://localhost:${BACKEND_PORT}`,
          changeOrigin: true,
        },
      },
    },
  }
})
