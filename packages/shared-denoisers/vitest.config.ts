import { defineConfig } from 'vitest/config';

// Node-side test config — excludes the *.gpu.test.ts files (those run via
// vitest.gpu.config.ts in headless Chromium under SwiftShader for real
// WebGPU coverage).
export default defineConfig({
  test: {
    environment: 'node',
    include: ['__tests__/**/*.test.ts'],
    exclude: ['__tests__/**/*.gpu.test.ts'],
  },
});
