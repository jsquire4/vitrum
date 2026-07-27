import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

const packageRoot = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(packageRoot, '../..');

export default defineConfig({
  root: packageRoot,
  resolve: {
    alias: {
      '@vitrum/core': path.join(repoRoot, 'packages/core/src/index.ts'),
      '@vitrum/shared-bvh': path.join(repoRoot, 'packages/shared-bvh/src/index.ts'),
      '@vitrum/shared-samplers': path.join(repoRoot, 'packages/shared-samplers/src/index.ts'),
      '@vitrum/stained-glass-extensions': path.join(
        repoRoot,
        'packages/stained-glass-extensions/src/index.ts',
      ),
    },
  },
  test: {
    include: ['__tests__/**/*.test.ts'],
    exclude: ['__tests__/**/*.gpu.test.ts'],
  },
});
