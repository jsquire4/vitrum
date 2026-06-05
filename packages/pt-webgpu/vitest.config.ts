import path from 'node:path';
import { defineConfig } from 'vitest/config';

/** Vitest resolves the fork via Vite; point at the ESM entry explicitly. */
const pathtracerRoot = path.resolve(__dirname, '../three-gpu-pathtracer');
const coreRoot = path.resolve(__dirname, '../core');
const sharedBvhRoot = path.resolve(__dirname, '../shared-bvh');
const sharedDenoisersRoot = path.resolve(__dirname, '../shared-denoisers');
const sharedSamplersRoot = path.resolve(__dirname, '../shared-samplers');

// Node-side test config — excludes *.gpu.test.ts (those run via the
// vitest.gpu.config.ts in headless Chromium under SwiftShader/Vulkan).
export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/__tests__/**/*.test.ts'],
    exclude: ['src/__tests__/**/*.gpu.test.ts'],
  },
  resolve: {
    alias: {
      '@vitrum/core': path.join(coreRoot, 'src/index.ts'),
      '@vitrum/shared-bvh': path.join(sharedBvhRoot, 'src/index.ts'),
      '@vitrum/shared-denoisers': path.join(sharedDenoisersRoot, 'src/index.ts'),
      '@vitrum/shared-samplers': path.join(sharedSamplersRoot, 'src/index.ts'),
      'three-gpu-pathtracer': path.join(pathtracerRoot, 'src/index.js'),
    },
  },
});
