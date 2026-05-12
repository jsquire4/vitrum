import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'node:path';

export default defineConfig({
  plugins: [react()],
  test: {
    // happy-dom is faster than jsdom and covers everything the vanilla
    // attachDebugOverlays test needs: createElement / appendChild /
    // removeChild / style + classList. The previous 5 it.skipIf(!hasDom)
    // guards now resolve to true and the tests run.
    environment: 'happy-dom',
  },
  resolve: {
    alias: {
      '@vitrum/core': path.resolve(__dirname, '../core/src/index.ts'),
    },
  },
});
