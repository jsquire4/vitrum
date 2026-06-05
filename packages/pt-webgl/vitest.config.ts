import path from 'node:path';
import { defineConfig } from 'vitest/config';

/**
 * Resolve the absorbed `three-gpu-pathtracer` workspace package for vitest.
 * The implementation now lives inside vitrum at packages/three-gpu-pathtracer,
 * so worktrees no longer need sibling-repo path discovery.
 */
const pathtracerRoot = path.resolve(__dirname, '../three-gpu-pathtracer');
const coreRoot = path.resolve(__dirname, '../core');
const sceneLightingRoot = path.resolve(__dirname, '../scene-lighting');
const sharedDenoisersRoot = path.resolve(__dirname, '../shared-denoisers');
const sharedSamplersRoot = path.resolve(__dirname, '../shared-samplers');
const stainedGlassRoot = path.resolve(__dirname, '../stained-glass-extensions');
const threeBindingsRoot = path.resolve(__dirname, '../three-bindings');

export default defineConfig({
  test: {
    environment: 'node',
  },
  resolve: {
    alias: {
      '@vitrum/core': path.join(coreRoot, 'src/index.ts'),
      '@vitrum/scene-lighting': path.join(sceneLightingRoot, 'src/index.ts'),
      '@vitrum/shared-denoisers': path.join(sharedDenoisersRoot, 'src/index.ts'),
      '@vitrum/shared-samplers': path.join(sharedSamplersRoot, 'src/index.ts'),
      '@vitrum/stained-glass-extensions': path.join(stainedGlassRoot, 'src/index.ts'),
      '@vitrum/three-bindings': path.join(threeBindingsRoot, 'src/index.ts'),
      // Top-level package import (e.g. `import { WebGLPathTracer } from 'three-gpu-pathtracer'`).
      'three-gpu-pathtracer': path.join(pathtracerRoot, 'src/index.js'),
      // Stable alias for subpath imports — keeps tests off brittle `../../../../../`
      // relative paths that break in worktrees. See materialsTextureSpectral.test.ts.
      '@vitrum-pathtracer': pathtracerRoot,
    },
  },
});
