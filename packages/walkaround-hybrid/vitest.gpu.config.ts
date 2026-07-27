import path from 'node:path';
import { defineConfig } from 'vitest/config';

const coreRoot = path.resolve(__dirname, '../core');
const sharedBvhRoot = path.resolve(__dirname, '../shared-bvh');
const sharedDenoisersRoot = path.resolve(__dirname, '../shared-denoisers');
const sharedSamplersRoot = path.resolve(__dirname, '../shared-samplers');
const stainedGlassRoot = path.resolve(__dirname, '../stained-glass-extensions');
const walkaroundRcRoot = path.resolve(__dirname, '../walkaround-rc');
const repoRoot = path.resolve(__dirname, '../..');

/**
 * Real-WebGPU neural tests. Chromium's SwiftShader adapter executes the same
 * WebGPU validation, pipeline creation, and dispatch paths as a hardware
 * adapter while keeping CI independent of a physical GPU.
 */
export default defineConfig({
  root: repoRoot,
  resolve: {
    alias: {
      '@vitrum/core': path.join(coreRoot, 'src/index.ts'),
      '@vitrum/shared-bvh': path.join(sharedBvhRoot, 'src/index.ts'),
      '@vitrum/shared-denoisers': path.join(sharedDenoisersRoot, 'src/index.ts'),
      '@vitrum/shared-samplers': path.join(sharedSamplersRoot, 'src/index.ts'),
      '@vitrum/stained-glass-extensions': path.join(stainedGlassRoot, 'src/index.ts'),
      '@vitrum/walkaround-rc': path.join(walkaroundRcRoot, 'src/index.ts'),
    },
  },
  test: {
    include: [
      'packages/walkaround-hybrid/__tests__/rcAcceptance.gpu.test.ts',
      'packages/walkaround-hybrid/__tests__/checkerboardDisocclusion.gpu.test.ts',
      'packages/walkaround-hybrid/src/neural/__tests__/**/*.gpu.test.ts',
      'packages/walkaround-hybrid/src/neural/nrc/__tests__/**/*.gpu.test.ts',
    ],
    browser: {
      enabled: true,
      headless: true,
      provider: 'playwright',
      name: 'chromium',
      providerOptions: {
        launch: {
          args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan,UseSkiaRenderer',
            '--use-vulkan=swiftshader',
            '--enable-unsafe-swiftshader',
            '--use-gl=angle',
            '--use-angle=swiftshader',
          ],
        },
      },
    },
  },
});
