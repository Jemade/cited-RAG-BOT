import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const API_TARGET = process.env.VITE_API_URL || 'http://localhost:8080';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/v1': {
        target: API_TARGET,
        changeOrigin: true,
      },
      '/health': {
        target: API_TARGET,
        changeOrigin: true,
      },
      '/metrics': {
        target: API_TARGET,
        changeOrigin: true,
      }
    }
  }
});
