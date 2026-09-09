import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), 'VITE_')
  const target = env.VITE_DEV_API_TARGET || 'http://127.0.0.1:8000'

  // Every backend route the app calls. Proxying keeps the browser on a single
  // origin during development, so CORS never comes into play locally.
  const apiRoutes = ['/predict', '/health', '/metrics', '/schema', '/location']

  return {
    plugins: [react()],
    server: {
      proxy: Object.fromEntries(
        apiRoutes.map((route) => [route, { target, changeOrigin: true }]),
      ),
    },
  }
})
