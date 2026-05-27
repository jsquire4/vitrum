/**
 * PR-D6 one-shot: perf JSON + reference PNGs on a hybrid-capable GPU host.
 *
 * From repo root (native Windows Chrome recommended from WSL):
 *   node tools/benchmark-runner/run-gpu-host-windows.mjs run-pr-hybrid-gpu-full.mjs
 *
 * Or directly on a host with ≥16 storage buffers / stage:
 *   VITRUM_PR_REQUIRE_GPU=1 VITRUM_PR_START_SERVER=1 node tools/benchmark-runner/run-pr-hybrid-gpu-full.mjs
 */

import { spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const node = process.execPath;

function run(script, extraEnv = {}) {
  const env = { ...process.env, ...extraEnv };
  const r = spawnSync(node, [resolve(here, script)], {
    stdio: 'inherit',
    env,
    cwd: resolve(here, '..', '..'),
  });
  if (r.status !== 0) {
    process.exit(r.status ?? 1);
  }
}

console.log('[pr-hybrid-gpu-full] benchmark:pr-hybrid (perf → PR-hybrid/perf/latest.json)');
run('run-pr-hybrid-bench.mjs', {
  VITRUM_PR_REQUIRE_GPU: '1',
  VITRUM_PR_START_SERVER: process.env.VITRUM_PR_START_SERVER ?? '1',
});

console.log('[pr-hybrid-gpu-full] benchmark:pr-hybrid-refs (PNG dirs)');
run('run-pr-hybrid-ref-capture.mjs', {
  VITRUM_PR_REF_START_SERVER: '0',
});

console.log('[pr-hybrid-gpu-full] done — see tools/reference-renders/PR-hybrid/');
