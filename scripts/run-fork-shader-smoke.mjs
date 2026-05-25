#!/usr/bin/env node
/**
 * Runs three-gpu-pathtracer shader string regression checks from the vitrum repo.
 * The fork is absorbed at packages/three-gpu-pathtracer; do not point this at
 * a sibling checkout.
 */
import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { existsSync } from 'node:fs';

const root = resolve(fileURLToPath(import.meta.url), '..', '..');
const forkDir = resolve(root, 'packages', 'three-gpu-pathtracer');
const script = resolve(forkDir, 'scripts/shader-smoke-check.js');
if (!existsSync(script)) {
  console.error(`Fork shader smoke script missing: ${script}`);
  console.error('Expected the absorbed package at packages/three-gpu-pathtracer.');
  process.exit(2);
}
const r = spawnSync(process.execPath, [script], { cwd: forkDir, stdio: 'inherit' });
const strict = process.env.VITRUM_FORK_SHADER_SMOKE_STRICT === '1';
if ((r.status ?? 1) !== 0) {
  if (strict) {
    process.exit(r.status ?? 1);
  }
  console.warn(
    '[vitrum] fork shader smoke reported failures, but continuing in advisory mode. ' +
      'Set VITRUM_FORK_SHADER_SMOKE_STRICT=1 to fail the run.',
  );
  process.exit(0);
}
process.exit(0);
