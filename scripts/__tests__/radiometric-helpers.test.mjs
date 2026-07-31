import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const helpersPath = join(repoRoot, 'tools', 'radiometric-ab', 'helpers.mjs');
const walkaroundHarnessPath = join(repoRoot, 'tools', 'radiometric-ab', 'walkaround-ab.mjs');
const walkaroundRunnerPath = join(repoRoot, 'tools', 'radiometric-ab', 'run-walkaround-ab.mjs');
const walkaroundRunValidationPath = join(
  repoRoot,
  'tools',
  'radiometric-ab',
  'walkaroundRunValidation.mjs',
);
const ptRunnerPath = join(repoRoot, 'tools', 'radiometric-ab', 'run-pt-ab.mjs');
const ptBdptHarnessPath = join(repoRoot, 'tools', 'radiometric-ab', 'ab-bdpt.mjs');
const radiometricProofsPath = join(repoRoot, 'tools', 'radiometric-ab', 'proofs.mjs');
const radiometricCheckerPath = join(repoRoot, 'tools', 'radiometric-ab', 'check-results.mjs');
// Bounded multi-vertex BDPT validation lives in ptWebgpuValidation.ts; its
// estimator loop is composed in the path-trace kernel.
const ptWebgpuValidationPath = join(
  repoRoot,
  'packages',
  'pt-webgpu',
  'src',
  'ptWebgpuValidation.ts',
);
const ptWebgpuKernelPath = join(
  repoRoot,
  'packages',
  'pt-webgpu',
  'src',
  'wgsl',
  'pathTrace',
  'kernel.wgsl.ts',
);

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

test('walkaround radiometric regions scale to the actual capture dimensions', async () => {
  const { regionLuminance, resolveWalkaroundRegions, validateWalkaroundPixelBuffer } =
    await import('../../tools/radiometric-ab/walkaroundRegions.mjs');
  const half = resolveWalkaroundRegions(64, 64);
  assert.deepEqual(half.glassCenter, { x0: 24, y0: 24, x1: 40, y1: 40 });
  assert.deepEqual(half.glossyBackWall, { x0: 16, y0: 16, x1: 48, y1: 48 });

  const wide = resolveWalkaroundRegions(256, 96);
  assert.deepEqual(wide.sunReceiver, { x0: 60, y0: 31, x1: 196, y1: 65 });
  const pixels = new Float32Array(256 * 96 * 4).fill(1);
  assert.equal(regionLuminance(pixels, 256, 96, wide.sunReceiver), 1);
  assert.throws(
    () => regionLuminance(pixels, 128, 96, wide.sunReceiver),
    /pixel buffer length must be exactly/,
  );
  assert.doesNotThrow(() => validateWalkaroundPixelBuffer(pixels, 256, 96));
});

test('walkaround capture validation rejects wrong lengths and non-finite values outside the ROI', async () => {
  const { regionLuminance, validateWalkaroundPixelBuffer } =
    await import('../../tools/radiometric-ab/walkaroundRegions.mjs');
  const width = 4;
  const height = 4;
  const expectedLength = width * height * 4;
  assert.throws(
    () => validateWalkaroundPixelBuffer(new Float32Array(expectedLength - 1), width, height),
    /pixel buffer length must be exactly 64/,
  );

  const roi = { x0: 1, y0: 1, x1: 4, y1: 4 };
  for (const [component, invalid] of [
    ['r', Number.NaN],
    ['g', Number.POSITIVE_INFINITY],
    ['b', Number.NEGATIVE_INFINITY],
    ['a', Number.NaN],
  ]) {
    const pixels = new Float32Array(expectedLength).fill(1);
    const componentIndex = { r: 0, g: 1, b: 2, a: 3 }[component];
    pixels[componentIndex] = invalid;
    // The invalid pixel is deliberately outside the selected region. The
    // complete-frame guard, not local ROI arithmetic, must reject it.
    assert.equal(regionLuminance(pixels, width, height, roi), 1);
    assert.throws(
      () => validateWalkaroundPixelBuffer(pixels, width, height, 'fixture capture'),
      new RegExp(`non-finite ${component} component at \\(0, 0\\)`),
    );
  }
});

test('walkaround A/B wrapper preserves custom output/status paths', async () => {
  const runner = await readFile(walkaroundRunnerPath, 'utf8');
  assert.match(runner, /VITRUM_WALKAROUND_AB_OUTPUT_PATH/);
  assert.match(runner, /VITRUM_WALKAROUND_AB_STATUS_PATH/);
  assert.match(runner, /repoRelative\(resultPath\)/);
  assert.match(runner, /preservedResultFile: resultFile/);
  assert.match(runner, /resultFile/);
});

test('walkaround A/B wrapper exposes the glossy 64-SPP regression lane', async () => {
  const runner = await readFile(walkaroundRunnerPath, 'utf8');
  assert.match(runner, /--glossy-spp64/);
  assert.match(runner, /walkaround-ab-glossy-spp64-status\.json/);
  assert.match(runner, /walkaround-ab-glossy-spp64\.json/);
  assert.match(runner, /glossy-spp64/);
  assert.match(runner, /VITRUM_WALKAROUND_AB_CASES = selectedCases/);
  assert.match(runner, /VITRUM_WALKAROUND_AB_SPP: renderConfig\.spp/);
});

test('walkaround A/B wrapper exposes the all-cases 64-SPP regression lane', async () => {
  const runner = await readFile(walkaroundRunnerPath, 'utf8');
  assert.match(runner, /--all-spp64/);
  assert.match(runner, /walkaround-ab-all-spp64-status\.json/);
  assert.match(runner, /walkaround-ab-all-spp64\.json/);
  assert.match(runner, /all-spp64/);
  assert.match(runner, /glossySpp64 && allSpp64/);
});

test('walkaround A/B wrapper fails closed on missing or incomplete result artifacts', async () => {
  const runner = await readFile(walkaroundRunnerPath, 'utf8');
  const validation = await readFile(walkaroundRunValidationPath, 'utf8');
  const combined = `${runner}\n${validation}`;
  assert.match(runner, /function readWalkaroundResultsForStatus/);
  assert.match(runner, /function failClosedResultStatus/);
  assert.match(combined, /walkaround-ab-missing-result/);
  assert.match(combined, /walkaround-ab-invalid-result/);
  assert.match(combined, /walkaround-ab-incomplete-result/);
  assert.match(combined, /walkaround-ab-stale-result/);
  assert.match(combined, /walkaround-ab-nonpass-result/);
  assert.match(combined, /walkaround-ab-invalid-verdict/);
  assert.match(validation, /typeof row\.verdict !== 'string'/);
  assert.match(runner, /verdict: 'FAIL'/);
});

test('pt radiometric wrapper fails closed on missing or incomplete result artifacts', async () => {
  const runner = await readFile(ptRunnerPath, 'utf8');
  assert.match(runner, /function resultArtifactProblem/);
  assert.match(runner, /pt-radiometric-ab-missing-result/);
  assert.match(runner, /pt-radiometric-ab-invalid-result/);
  assert.match(runner, /pt-radiometric-ab-nonpass-result/);
  assert.match(runner, /pt-radiometric-ab-stale-result/);
  assert.match(runner, /expectedProvenance/);
  assert.match(runner, /payload\.verdict === 'PASS'/);
  assert.match(runner, /artifactProblem \?\? hostBoundary/);
});

test('pt BDPT regression harness covers the complete bounded multi-vertex mode', async () => {
  const harness = await readFile(ptBdptHarnessPath, 'utf8');
  const proofs = await readFile(radiometricProofsPath, 'utf8');
  const checker = await readFile(radiometricCheckerPath, 'utf8');
  const ptWebgpuValidation = await readFile(ptWebgpuValidationPath, 'utf8');
  const ptWebgpuKernel = await readFile(ptWebgpuKernelPath, 'utf8');

  assert.match(harness, /CONTROL_MAX_LIGHT_BOUNCES = \[1, 2, 3, 8\]/);
  assert.match(harness, /supportedDepths: CONTROL_MAX_LIGHT_BOUNCES/);
  assert.doesNotMatch(harness, /experimentalMultiVertex/);
  assert.doesNotMatch(harness, /researchFindings/);

  assert.match(proofs, /export const PT_LOCAL_ACCEPTANCE_PROOFS/);
  assert.match(proofs, /bdptConnectionMisFull\.test\.ts/);
  assert.match(proofs, /bdptEstimatorOwnership\.test\.ts/);
  assert.match(proofs, /bdptDeltaTransport\.test\.ts/);
  assert.match(proofs, /BDPT_DEFAULT_LIGHT_BOUNCES = 2/);
  assert.match(proofs, /BDPT_MAX_LIGHT_BOUNCES =\\n {2}PT_WEBGPU_BDPT_SUPPORT\.maxLightVertices/);

  assert.match(checker, /function checkPtSourcePins/);
  assert.match(checker, /named test-source file \$\{path\} has no test declaration/);
  assert.match(checker, /It does not execute those tests/);
  assert.match(checker, /missing stable contract needle/);
  assert.doesNotMatch(checker, /checkBdptMultiVertexResearch/);

  assert.match(ptWebgpuValidation, /BDPT_DEFAULT_LIGHT_BOUNCES = 2/);
  assert.match(
    ptWebgpuValidation,
    /BDPT_MAX_LIGHT_BOUNCES =\s+PT_WEBGPU_BDPT_SUPPORT\.maxLightVertices/,
  );
  assert.doesNotMatch(ptWebgpuValidation, /promotionReady: false/);
  assert.doesNotMatch(ptWebgpuValidation, /research path/);

  assert.match(ptWebgpuKernel, /for \(var lvi = 0u; lvi < maxLv; lvi\+\+\) \{/);
  assert.match(ptWebgpuKernel, /radiance = radiance \+ evaluateBdptConnection\(/);
});

async function loadVarianceROI() {
  const source = await readFile(helpersPath, 'utf8');
  const match = source.match(/export function varianceROI\(([^)]*)\) \{([\s\S]*?)\n\}/);
  assert.ok(match, 'varianceROI export should be present in helpers.mjs');
  return Function(`return function varianceROI(${match[1]}) {${match[2]}\n};`)();
}
