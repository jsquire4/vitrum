#!/usr/bin/env node
/**
 * Runs three-gpu-pathtracer fork shader string regression checks from the vitrum repo.
 * Requires sibling checkout at ../three-gpu-pathtracer (override with VITRUM_FORK_DIR).
 */
import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { existsSync } from 'node:fs';

const root = resolve(fileURLToPath(import.meta.url), '..', '..');
const forkDir = resolve(process.env.VITRUM_FORK_DIR ?? resolve(root, '..', 'three-gpu-pathtracer'));
const script = resolve(forkDir, 'scripts/shader-smoke-check.js');
if (!existsSync(script)) {
  console.error(`Fork shader smoke script missing: ${script}`);
  console.error('Set VITRUM_FORK_DIR to your three-gpu-pathtracer checkout.');
  process.exit(2);
}
const r = spawnSync(process.execPath, [script], { cwd: forkDir, stdio: 'inherit' });
process.exit(r.status ?? 1);
