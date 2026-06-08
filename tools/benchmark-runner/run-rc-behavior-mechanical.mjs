/**
 * Mechanical RC behavior acceptance — deterministic fixture metrics + vitest.
 *
 * This does not capture GPU frames. It writes/uses the committed-style 64x64
 * fixture PNG path, asks run-acceptance-metrics.mjs to emit behavior metrics JSON,
 * then runs the env-gated walkaround-rc behavior test against that JSON.
 */

import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { getRepoRoot } from './repoRoot.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = getRepoRoot(import.meta.url);
const acceptanceDir = resolve(here, 'results', 'acceptance');
const behaviorMetricsPath = resolve(acceptanceDir, 'rc-behavior-mechanical-metrics.json');
const acceptanceMetricsPath = resolve(acceptanceDir, 'rc-acceptance-mechanical-metrics.json');
const offPngPath = resolve(
  repoRoot,
  'tools/reference-renders/W8-rc-mechanical-off/cornell-walkaround-rc-off.png',
);
const onPngPath = resolve(
  repoRoot,
  'tools/reference-renders/W8-rc-mechanical-on/cornell-walkaround-rc-on.png',
);

function runNode(script, env = {}) {
  const result = spawnSync('node', [resolve(here, script)], {
    cwd: here,
    env: { ...process.env, ...env },
    stdio: 'inherit',
    encoding: 'utf8',
  });
  if (result.status !== 0) process.exit(result.status ?? 1);
}

function assertBehaviorMetrics(path) {
  const metrics = JSON.parse(readFileSync(path, 'utf8'));
  if (
    typeof metrics.indirectEnergyDelta !== 'number' ||
    typeof metrics.nanPixelCount !== 'number'
  ) {
    console.error(
      `[rc-behavior-mechanical] invalid metrics at ${path}; ` +
        'expected numeric indirectEnergyDelta and nanPixelCount',
    );
    process.exit(1);
  }
  if (metrics.indirectEnergyDelta <= 0.001 || metrics.nanPixelCount !== 0) {
    console.error(
      `[rc-behavior-mechanical] indirectEnergyDelta=${metrics.indirectEnergyDelta}, ` +
        `nanPixelCount=${metrics.nanPixelCount}`,
    );
    process.exit(1);
  }
}

function main() {
  runNode('write-rc-mechanical-fixtures.mjs');
  runNode('run-acceptance-metrics.mjs', {
    VITRUM_RC_OFF_PNG: offPngPath,
    VITRUM_RC_ON_PNG: onPngPath,
    VITRUM_RC_ACCEPTANCE_OUT: acceptanceMetricsPath,
    VITRUM_RC_BEHAVIOR_OUT: behaviorMetricsPath,
  });
  assertBehaviorMetrics(behaviorMetricsPath);

  const test = spawnSync(
    'npm',
    ['test', '--workspace', '@vitrum/walkaround-rc', '--', 'rcBehavior.gpu'],
    {
      cwd: repoRoot,
      env: {
        ...process.env,
        VITRUM_RC_BEHAVIOR_ACCEPTANCE: '1',
        VITRUM_RC_BEHAVIOR_METRICS: behaviorMetricsPath,
      },
      stdio: 'inherit',
      encoding: 'utf8',
    },
  );
  process.exit(test.status ?? 1);
}

main();
