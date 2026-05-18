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
    // 5179 — port allocation across examples is hand-managed; cornell-box
    // owns 5174, two-engines 5175, hero-viewer 5176, hero-lighting-designer
    // 5177, hero-product-viz 5178, neural-denoiser 5179. Earlier revisions
    // had neural-denoiser sharing 5176 with hero-viewer, which caused
    // EADDRINUSE on concurrent `dev` runs.
    port: 5179,
  },
});
