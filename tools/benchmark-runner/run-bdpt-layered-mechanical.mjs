/**
 * Mechanical BDPT vs layered-BSDF harness — metrics + gated vitest, no GPU.
 * Preserves GPU captures in bdpt-layered-mechanical/ when files are >50KB.
 */

import { spawnSync } from 'node:child_process';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PNG } from 'pngjs';
import { getRepoRoot } from './repoRoot.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = getRepoRoot(import.meta.url);
const fixtureDir = resolve(repoRoot, 'tools/reference-renders/bdpt-layered-mechanical');
const resultsDir = resolve(here, 'results', 'acceptance');

const layeredPng = resolve(fixtureDir, 'cornell-layered.png');
const bdptPng = resolve(fixtureDir, 'cornell-layered-bdpt.png');

function meanAbsRgbDelta(a, b) {
  if (a.width !== b.width || a.height !== b.height) {
    throw new Error(`PNG size mismatch ${a.width}x${a.height} vs ${b.width}x${b.height}`);
  }
  let accum = 0;
  let count = 0;
  for (let y = 0; y < a.height; y += 1) {
    for (let x = 0; x < a.width; x += 1) {
      const i = (y * a.width + x) * 4;
      const dr = Math.abs(((a.data[i] ?? 0) - (b.data[i] ?? 0)) / 255);
      const dg = Math.abs(((a.data[i + 1] ?? 0) - (b.data[i + 1] ?? 0)) / 255);
      const db = Math.abs(((a.data[i + 2] ?? 0) - (b.data[i + 2] ?? 0)) / 255);
      accum += (dr + dg + db) / 3;
      count += 1;
    }
  }
  return count > 0 ? accum / count : 0;
}

function runNode(script) {
  const r = spawnSync('node', [resolve(here, script)], {
    cwd: here,
    env: process.env,
    stdio: 'inherit',
    encoding: 'utf8',
  });
  if (r.status !== 0) process.exit(r.status ?? 1);
}

async function ensureFixturePngs() {
  if (process.env.VITRUM_BDPT_REGENERATE_FIXTURES === '1') {
    runNode('write-bdpt-layered-mechanical-fixtures.mjs');
    return;
  }
  try {
    const [layeredSt, bdptSt] = await Promise.all([stat(layeredPng), stat(bdptPng)]);
    if (layeredSt.size > 50_000 && bdptSt.size > 50_000) return;
  } catch {
    /* missing — generate stubs */
  }
  runNode('write-bdpt-layered-mechanical-fixtures.mjs');
}

async function main() {
  await ensureFixturePngs();

  const layered = PNG.sync.read(await readFile(layeredPng));
  const bdpt = PNG.sync.read(await readFile(bdptPng));
  const bdptDeltaMean = meanAbsRgbDelta(layered, bdpt);

  await mkdir(resultsDir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const metricsPath =
    process.env.VITRUM_BDPT_LAYERED_METRICS_OUT ??
    resolve(resultsDir, `bdpt-layered-metrics-${stamp}.json`);

  const metrics = {
    schemaVersion: 'bdpt-layered-mechanical-2026-05-27',
    bdptDeltaMean,
    layeredPng: 'tools/reference-renders/bdpt-layered-mechanical/cornell-layered.png',
    bdptPng: 'tools/reference-renders/bdpt-layered-mechanical/cornell-layered-bdpt.png',
    note: 'Replace stubs via npm run benchmark:bdpt-layered-refs on a GPU host.',
  };
  await writeFile(metricsPath, `${JSON.stringify(metrics, null, 2)}\n`, 'utf8');
  console.log(`VITRUM_BDPT_LAYERED_METRICS=${metricsPath}`);
  console.log(`bdptDeltaMean=${bdptDeltaMean.toFixed(6)}`);

  if (bdptDeltaMean <= 0.005) {
    throw new Error(`bdptDeltaMean=${bdptDeltaMean} (expected > 0.005)`);
  }

  const test = spawnSync(
    'npm',
    ['test', '--workspace', '@vitrum/pt-webgl', '--', 'bdptLayeredAcceptance.gpu'],
    {
      cwd: repoRoot,
      env: {
        ...process.env,
        VITRUM_BDPT_LAYERED_ACCEPTANCE: '1',
        VITRUM_BDPT_LAYERED_METRICS: metricsPath,
      },
      stdio: 'inherit',
      encoding: 'utf8',
    },
  );
  process.exit(test.status ?? 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
