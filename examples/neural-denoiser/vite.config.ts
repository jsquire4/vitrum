import { defineConfig } from 'vite';
import path from 'node:path';

const repoRoot = path.resolve(__dirname, '../..');

export default defineConfig({
  resolve: {
    dedupe: ['three'],
    alias: {
      '@vitrum/core': path.resolve(repoRoot, 'packages/core/src/index.ts'),
      '@vitrum/three-bindings': path.resolve(repoRoot, 'packages/three-bindings/src/index.ts'),
      '@vitrum/walkaround-hybrid': path.resolve(repoRoot, 'packages/walkaround-hybrid/src/index.ts'),
      '@vitrum-examples/shared': path.resolve(repoRoot, 'examples/shared/src/index.ts'),
    },
  },
  server: {
    port: 5176,
  },
});
