import { defineConfig } from 'vite';
import path from 'node:path';

const repoRoot = path.resolve(__dirname, '../..');

export default defineConfig({
  resolve: {
    dedupe: ['three', 'three-mesh-bvh', 'three-gpu-pathtracer'],
    alias: {
      '@vitrum/core': path.resolve(repoRoot, 'packages/core/src/index.ts'),
      '@vitrum/pt-webgl': path.resolve(repoRoot, 'packages/pt-webgl/src/index.ts'),
      '@vitrum/pt-webgpu': path.resolve(repoRoot, 'packages/pt-webgpu/src/index.ts'),
      '@vitrum/three-bindings': path.resolve(repoRoot, 'packages/three-bindings/src/index.ts'),
      '@vitrum/walkaround-hybrid': path.resolve(repoRoot, 'packages/walkaround-hybrid/src/index.ts'),
      '@vitrum-examples/shared': path.resolve(repoRoot, 'examples/shared/src/index.ts'),
    },
  },
  server: {
    port: 5175,
  },
  build: {
    rollupOptions: {
      input: {
        index: path.resolve(__dirname, 'index.html'),
        ptwebgl: path.resolve(__dirname, 'pt-webgl.html'),
        walkaround: path.resolve(__dirname, 'walkaround.html'),
        ptwebgpu: path.resolve(__dirname, 'pt-webgpu.html'),
      },
    },
  },
});
