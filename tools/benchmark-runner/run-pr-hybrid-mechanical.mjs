/**
 * Mechanical PR-6 checks (no Playwright / GPU).
 */

import { access } from 'node:fs/promises';
import { constants as fsConstants } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { runCommandWithTimeout } from './runCommandWithTimeout.mjs';
import { PR_HYBRID_BENCHMARK_SCENARIOS, WG0_PT_WEBGPU_SCENARIOS } from './scenario-presets.mjs';
import { getRepoRoot } from './repoRoot.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = getRepoRoot(import.meta.url);
const baselineDir = resolve(repoRoot, 'tools/reference-renders/baseline');

const required = [
  'PR-hybrid-200k-static',
  'PR-hybrid-tlas-10-inst',
  'PR-hybrid-material-churn',
  'PR-hybrid-emitter-churn',
];
const presetIds = new Set(PR_HYBRID_BENCHMARK_SCENARIOS.map((s) => s.scenarioId));
let failures = 0;
for (const id of required) {
  if (!presetIds.has(id)) {
    failures += 1;
    console.error(`[pr-mechanical] missing preset ${id}`);
  }
}

for (const scenarioId of WG0_PT_WEBGPU_SCENARIOS) {
  const baselinePng = resolve(baselineDir, `${scenarioId}.png`);
  try {
    await access(baselinePng, fsConstants.F_OK);
    console.log(`[pr-mechanical] PASS WG-0 baseline ${scenarioId}.png`);
  } catch {
    failures += 1;
    console.error(`[pr-mechanical] missing WG-0 baseline ${baselinePng}`);
  }
}

const run = await runCommandWithTimeout(
  'npx vitest run examples/two-engines-one-scene/__tests__/benchmarkScenes.test.ts',
  { cwd: repoRoot, timeoutMs: 120_000 },
);
if (run.code !== 0) {
  failures += 1;
  console.error(run.stderr || run.stdout);
} else {
  console.log('[pr-mechanical] PASS benchmark scene triangle budgets');
}

process.exit(failures > 0 ? 1 : 0);
