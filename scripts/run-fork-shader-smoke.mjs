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
const expectedForkBranch = process.env.VITRUM_EXPECTED_FORK_BRANCH ?? 'phase4-normalmap-shadow-rays';
if (!existsSync(script)) {
  console.error(`Fork shader smoke script missing: ${script}`);
  console.error('Set VITRUM_FORK_DIR to your three-gpu-pathtracer checkout.');
  process.exit(2);
}
const branch = spawnSync('git', ['rev-parse', '--abbrev-ref', 'HEAD'], {
  cwd: forkDir,
  encoding: 'utf8',
});
if (branch.status === 0) {
  const currentBranch = branch.stdout.trim();
  if (currentBranch !== expectedForkBranch) {
    console.error(
      `Fork checkout branch mismatch: expected "${expectedForkBranch}", got "${currentBranch}".`,
    );
    console.error('Switch the fork checkout or override VITRUM_EXPECTED_FORK_BRANCH.');
    process.exit(2);
  }
}
const r = spawnSync(process.execPath, [script], { cwd: forkDir, stdio: 'inherit' });
process.exit(r.status ?? 1);
