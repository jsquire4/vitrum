import { defineConfig } from 'vite';
import path from 'node:path';

const repoRoot = path.resolve(__dirname, '../..');

export default defineConfig({
  resolve: {
    dedupe: ['three', 'three-mesh-bvh', 'three-gpu-pathtracer'],
    alias: {
      '@vitrum/core': path.resolve(repoRoot, 'packages/core/src/index.ts'),
      '@vitrum/pt-webgl': path.resolve(repoRoot, 'packages/pt-webgl/src/index.ts'),
      '@vitrum/three-bindings': path.resolve(repoRoot, 'packages/three-bindings/src/index.ts'),
    },
  },
  server: {
    port: 5174,
  },
});
