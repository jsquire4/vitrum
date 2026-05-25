import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

const repoRoot = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  resolve: {
    alias: {
      '@vitrum/stained-glass-extensions': path.join(
        repoRoot,
        '../stained-glass-extensions/src/index.ts',
      ),
    },
  },
});
