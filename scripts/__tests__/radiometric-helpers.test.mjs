import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const helpersPath = join(repoRoot, 'tools', 'radiometric-ab', 'helpers.mjs');
const walkaroundHarnessPath = join(repoRoot, 'tools', 'radiometric-ab', 'walkaround-ab.mjs');
const walkaroundRunnerPath = join(repoRoot, 'tools', 'radiometric-ab', 'run-walkaround-ab.mjs');

test('radiometric varianceROI fails closed when capture count is too low', async () => {
  const varianceROI = await loadVarianceROI();
  assert.throws(
    () => varianceROI([], 1, 0, 0, 0, 0),
    /varianceROI requires at least 2 images, got 0/,
  );
  assert.throws(
    () => varianceROI([new Float32Array(4)], 1, 0, 0, 0, 0),
    /varianceROI requires at least 2 images, got 1/,
  );
});

test('walkaround A/B harness exposes high-quality artifact controls', async () => {
  const harness = await readFile(walkaroundHarnessPath, 'utf8');
  assert.match(harness, /VITRUM_WALKAROUND_AB_WIDTH/);
  assert.match(harness, /VITRUM_WALKAROUND_AB_HEIGHT/);
  assert.match(harness, /VITRUM_WALKAROUND_AB_SPP/);
  assert.match(harness, /VITRUM_WALKAROUND_AB_OUTPUT_PATH/);
  assert.match(harness, /qualityProfile/);
  assert.match(harness, /renderConfig/);
});

test('walkaround A/B wrapper preserves custom output/status paths', async () => {
  const runner = await readFile(walkaroundRunnerPath, 'utf8');
  assert.match(runner, /VITRUM_WALKAROUND_AB_OUTPUT_PATH/);
  assert.match(runner, /VITRUM_WALKAROUND_AB_STATUS_PATH/);
  assert.match(runner, /repoRelative\(resultPath\)/);
  assert.match(runner, /preservedResultFile: resultFile/);
  assert.match(runner, /resultFile/);
});

test('walkaround A/B wrapper exposes the glossy 64-SPP promotion lane', async () => {
  const runner = await readFile(walkaroundRunnerPath, 'utf8');
  assert.match(runner, /--glossy-spp64/);
  assert.match(runner, /walkaround-ab-glossy-spp64-status\.json/);
  assert.match(runner, /walkaround-ab-glossy-spp64\.json/);
  assert.match(runner, /glossy-spp64/);
  assert.match(runner, /VITRUM_WALKAROUND_AB_CASES = selectedCases/);
  assert.match(runner, /VITRUM_WALKAROUND_AB_SPP: renderConfig\.spp/);
});

async function loadVarianceROI() {
  const source = await readFile(helpersPath, 'utf8');
  const match = source.match(/export function varianceROI\(([^)]*)\) \{([\s\S]*?)\n\}/);
  assert.ok(match, 'varianceROI export should be present in helpers.mjs');
  return Function(`return function varianceROI(${match[1]}) {${match[2]}\n};`)();
}
