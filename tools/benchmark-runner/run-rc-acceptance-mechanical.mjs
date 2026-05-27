/**
 * Mechanical RC acceptance — metrics + vitest using committed 64×64 fixture PNGs.
 * Does not require a hybrid-capable GPU.
 */

import { spawnSync } from 'node:child_process';
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { getRepoRoot } from './repoRoot.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = getRepoRoot(import.meta.url);
const acceptanceDir = resolve(here, 'results', 'acceptance');

function latestMetricsPath() {
  const files = readdirSync(acceptanceDir)
    .filter((f) => f.startsWith('rc-acceptance-metrics-') && f.endsWith('.json'))
    .sort()
    .reverse();
  if (files.length === 0) return null;
  return resolve(acceptanceDir, files[0]);
}

function runNode(script, env = {}) {
  const r = spawnSync('node', [resolve(here, script)], {
    cwd: here,
    env: { ...process.env, ...env },
    stdio: 'inherit',
    encoding: 'utf8',
  });
  if (r.status !== 0) process.exit(r.status ?? 1);
}

function main() {
  runNode('write-rc-mechanical-fixtures.mjs');
  runNode('run-rc-acceptance.mjs', {
    VITRUM_RC_SKIP_CAPTURE: '1',
    VITRUM_RC_START_SERVER: '0',
  });

  const metricsPath = process.env.VITRUM_RC_ACCEPTANCE_METRICS ?? latestMetricsPath();
  if (metricsPath == null) {
    console.error('[rc-acceptance-mechanical] no metrics JSON');
    process.exit(1);
  }

  const metrics = JSON.parse(readFileSync(metricsPath, 'utf8'));
  if (typeof metrics.rcDeltaMean !== 'number' || metrics.rcDeltaMean <= 0.005) {
    console.error(
      `[rc-acceptance-mechanical] rcDeltaMean=${metrics.rcDeltaMean} (expected > 0.005)`,
    );
    process.exit(1);
  }

  const test = spawnSync(
    'npm',
    ['test', '--workspace', '@vitrum/walkaround-hybrid', '--', 'rcAcceptance.gpu'],
    {
      cwd: repoRoot,
      env: {
        ...process.env,
        VITRUM_RC_ACCEPTANCE: '1',
        VITRUM_RC_ACCEPTANCE_METRICS: metricsPath,
      },
      stdio: 'inherit',
      encoding: 'utf8',
    },
  );
  process.exit(test.status ?? 1);
}

main();
