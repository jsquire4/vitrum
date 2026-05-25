import path from 'node:path';
import { defineConfig } from 'vitest/config';

const stainedGlassRoot = path.resolve(__dirname, '../stained-glass-extensions');

export default defineConfig({
  resolve: {
    alias: {
      '@vitrum/stained-glass-extensions': path.join(stainedGlassRoot, 'src/index.ts'),
    },
  },
});
