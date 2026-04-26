import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, '.', '');
    const apiTarget = env.ICU_API_URL || 'http://127.0.0.1:8765';
    return {
      server: {
        port: 3001,
        host: '0.0.0.0',
        strictPort: false,
        proxy: {
          '/api': {
            target: apiTarget,
            changeOrigin: true,
          },
        },
      },
      plugins: [react()],
      define: {
        // Kept for backwards compat with services/groqService.ts (no longer used).
        'process.env.API_KEY': JSON.stringify(env.GROQ_API_KEY || ''),
        'process.env.GROQ_API_KEY': JSON.stringify(env.GROQ_API_KEY || ''),
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      }
    };
});
