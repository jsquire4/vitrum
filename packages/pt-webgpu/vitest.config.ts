import path from 'node:path';
import { defineConfig } from 'vitest/config';

/** Vitest resolves the fork via Vite; point at the ESM entry explicitly. */
const pathtracerRoot = path.resolve(__dirname, '../../../three-gpu-pathtracer');

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
      'three-gpu-pathtracer': path.join(pathtracerRoot, 'src/index.js'),
    },
  },
});
