import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'node:path';

export default defineConfig({
  plugins: [react()],
  test: {
    // Use node environment. DOM-dependent tests in vanilla.test.ts are guarded
    // by it.skipIf(!hasDom) and will be skipped in node env. Add jsdom/happy-dom
    // as a devDep to enable them if needed in the future.
    environment: 'node',
  },
  resolve: {
    alias: {
      '@vitrum/core': path.resolve(__dirname, '../core/src/index.ts'),
    },
  },
});
