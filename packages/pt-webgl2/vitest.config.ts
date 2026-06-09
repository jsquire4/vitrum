import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'node:url';

// Resolve @vitrum/* from source so the new workspace package's tests run without
// requiring a fresh `npm install` to create node_modules symlinks (which would
// churn the lockfile while concurrent work is in flight).
const r = (p: string) => fileURLToPath(new URL(p, import.meta.url));

export default defineConfig({
  resolve: {
    alias: {
      '@vitrum/core': r('../core/src/index.ts'),
      '@vitrum/shared-bvh': r('../shared-bvh/src/index.ts'),
      '@vitrum/shared-denoisers': r('../shared-denoisers/src/index.ts'),
      '@vitrum/shared-samplers': r('../shared-samplers/src/index.ts'),
    },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
});
