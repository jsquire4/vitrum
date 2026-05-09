import path from 'node:path';
import { defineConfig } from 'vitest/config';

/** Vitest resolves the fork via Vite; point at the ESM entry explicitly. */
const pathtracerRoot = path.resolve(__dirname, '../../../three-gpu-pathtracer');

export default defineConfig({
  test: {
    environment: 'node',
  },
  resolve: {
    alias: {
      'three-gpu-pathtracer': path.join(pathtracerRoot, 'src/index.js'),
    },
  },
});
