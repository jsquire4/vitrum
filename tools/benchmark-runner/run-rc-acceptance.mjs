/**
 * RC acceptance metrics for committed/offline PNGs.
 *
 * GPU capture used to live here through the retired two-engines example. The
 * runner is now deliberately metrics-only; refresh PNGs with the native
 * validation harness, then point VITRUM_RC_OFF_PNG / VITRUM_RC_ON_PNG here.
 */

import { access, mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { getRepoRoot } from './repoRoot.mjs';
import { runCommandWithTimeout } from './runCommandWithTimeout.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = getRepoRoot(import.meta.url);
const resultsDir = resolve(here, 'results', 'acceptance');

const offPng = process.env.VITRUM_RC_OFF_PNG
  ? resolve(repoRoot, process.env.VITRUM_RC_OFF_PNG)
  : resolve(repoRoot, 'tools/reference-renders/W8-rc-off/cornell-walkaround-rc-off.png');
const onPng = process.env.VITRUM_RC_ON_PNG
  ? resolve(repoRoot, process.env.VITRUM_RC_ON_PNG)
  : resolve(repoRoot, 'tools/reference-renders/W8-rc-on/cornell-walkaround-rc-on.png');

async function assertReadablePng(path, label) {
  try {
    await access(path);
  } catch {
    throw new Error(
      `${label} missing at ${path}. Generate fixtures with ` +
        '`npm run write-rc-mechanical-fixtures --workspace @vitrum/benchmark-runner` ' +
        'or refresh the GPU validation PNGs before running this gate.',
    );
  }
}

async function main() {
  await assertReadablePng(offPng, 'RC off PNG');
  await assertReadablePng(onPng, 'RC on PNG');

  await mkdir(resultsDir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const rcAcceptanceOut = resolve(resultsDir, `rc-acceptance-metrics-${stamp}.json`);
  const rcBehaviorOut = resolve(resultsDir, `rc-behavior-metrics-${stamp}.json`);

  const metrics = await runCommandWithTimeout('node ./run-acceptance-metrics.mjs', {
    cwd: here,
    env: {
      ...process.env,
      VITRUM_RC_OFF_PNG: offPng,
      VITRUM_RC_ON_PNG: onPng,
      VITRUM_RC_ACCEPTANCE_OUT: rcAcceptanceOut,
      VITRUM_RC_BEHAVIOR_OUT: rcBehaviorOut,
      VITRUM_PIPELINE_CREATES_BEFORE: '0',
      VITRUM_PIPELINE_CREATES_AFTER: '0',
    },
    timeoutMs: 60_000,
  });
  if (metrics.code !== 0) {
    throw new Error(
      `run-acceptance-metrics failed (code=${metrics.code}): ${metrics.stderr || metrics.stdout}`,
    );
  }

  const manifest = {
    capturedAt: new Date().toISOString(),
    offPng,
    onPng,
    rcAcceptanceMetrics: rcAcceptanceOut,
    rcBehaviorMetrics: rcBehaviorOut,
    stdout: metrics.stdout.trim(),
    note: 'Metrics-only runner; GPU capture is external validation.',
  };
  const manifestPath = resolve(resultsDir, `rc-acceptance-manifest-${stamp}.json`);
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');

  console.log(`VITRUM_RC_ACCEPTANCE_METRICS=${rcAcceptanceOut}`);
  console.log(`VITRUM_RC_BEHAVIOR_METRICS=${rcBehaviorOut}`);
  console.log(`[rc-acceptance] manifest ${manifestPath}`);
  console.log(metrics.stdout);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
