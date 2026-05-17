import fs from 'node:fs';
import path from 'node:path';
import { defineConfig } from 'vitest/config';

/**
 * Resolve the local `three-gpu-pathtracer` fork for vitest.
 *
 * The fork is intentionally NOT a workspace member — it lives as a *sibling*
 * directory to the `vitrum/` repo (per CLAUDE.md). The `package.json` dep
 * `"three-gpu-pathtracer": "file:../../../three-gpu-pathtracer"` resolves
 * relative to `packages/pt-webgl/` and only points at the right place when
 * the package lives at `<parent>/vitrum/packages/pt-webgl/`.
 *
 * In a git worktree at `<parent>/vitrum/.claude/worktrees/<id>/packages/pt-webgl/`,
 * the same relative path lands at `<parent>/vitrum/.claude/worktrees/three-gpu-pathtracer/`
 * which doesn't exist. The npm-install-created symlink at
 * `node_modules/three-gpu-pathtracer` is therefore broken in worktrees.
 *
 * Lookup chain (first hit wins):
 *   1. `VITRUM_PT_FORK_PATH` env var (explicit override for CI / unusual layouts)
 *   2. Three-up sibling of __dirname (main-checkout case)
 *   3. Five-up sibling of __dirname (worktree case: walks past `.claude/worktrees/<id>/`)
 *   4. Walk parents of __dirname looking for a sibling named `three-gpu-pathtracer`
 *
 * If none resolve, the alias points at the stale three-up path; tests that
 * import from the fork should guard with `fs.existsSync(pathtracerRoot)` and
 * skip cleanly (see materialsTextureSpectral.test.ts). Tests that only need
 * the API surface should mock the module with `vi.mock('three-gpu-pathtracer')`
 * (see capabilities.test.ts, factory.test.ts).
 */
function resolvePathtracerRoot(): string {
  const FORK_DIR_NAME = 'three-gpu-pathtracer';
  const SENTINEL = path.join('src', 'index.js');

  const envOverride = process.env['VITRUM_PT_FORK_PATH'];
  if (envOverride && fs.existsSync(path.join(envOverride, SENTINEL))) {
    return envOverride;
  }

  const candidates = [
    // main checkout: pt-webgl → packages → vitrum → <parent>
    path.resolve(__dirname, '../../..', FORK_DIR_NAME),
    // worktree: pt-webgl → packages → <agent-id> → worktrees → .claude → vitrum → <parent>
    path.resolve(__dirname, '../../../../../..', FORK_DIR_NAME),
  ];
  for (const c of candidates) {
    if (fs.existsSync(path.join(c, SENTINEL))) return c;
  }

  // Walk parents looking for a sibling fork directory. Stops at filesystem root.
  let dir = __dirname;
  for (let i = 0; i < 12; i++) {
    const parent = path.dirname(dir);
    if (parent === dir) break;
    const candidate = path.join(parent, FORK_DIR_NAME);
    if (fs.existsSync(path.join(candidate, SENTINEL))) return candidate;
    dir = parent;
  }

  // Not found. Return the canonical main-checkout location so the alias
  // still has a stable value; tests must guard with existsSync.
  return candidates[0]!;
}

const pathtracerRoot = resolvePathtracerRoot();

export default defineConfig({
  test: {
    environment: 'node',
  },
  resolve: {
    alias: {
      // Top-level package import (e.g. `import { WebGLPathTracer } from 'three-gpu-pathtracer'`).
      'three-gpu-pathtracer': path.join(pathtracerRoot, 'src/index.js'),
      // Stable alias for subpath imports — keeps tests off brittle `../../../../../`
      // relative paths that break in worktrees. See materialsTextureSpectral.test.ts.
      '@vitrum-fork/three-gpu-pathtracer': pathtracerRoot,
    },
  },
});
