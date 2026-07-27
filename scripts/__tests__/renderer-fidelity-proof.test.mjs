import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const checkerPath = join(repoRoot, 'tools', 'renderer-fidelity-proof', 'check-proofs.mjs');
const ptWebgl2BrowserStatusPath = join(repoRoot, 'tools', 'gltf-browser-proof', 'pt-webgl2-real-status.json');
const ptWebgl2CanvasFirstStatusPath = join(repoRoot, 'tools', 'gltf-browser-proof', 'pt-webgl2-real-canvas-first-status.json');
const ptWebgl2BrowserManifestPath = join(repoRoot, 'tools', 'reference-renders', 'gltf-real-browser-pt-webgl2', 'manifest.json');

test('renderer fidelity checker grades live source contracts without a host-capture veto', async () => {
  const checker = await readFile(checkerPath, 'utf8');

  assert.match(checker, /Host-specific captures are regression artifacts, not maturity/);
  assert.match(checker, /EXPECTED_ROWS/);
  assert.match(checker, /SOURCE_PROOFS/);
  assert.match(checker, /matrix contains an unfinished maturity grade/);
  assert.match(checker, /matrix makes host capture a maturity prerequisite/);
  assert.match(checker, /one-to-eight planar geometric-normal/);
  assert.match(checker, /bdptOwnsFiniteLightFamily/);
  assert.doesNotMatch(checker, /pt-webgl2 must not be marked supported while/);
});

test('renderer fidelity canonical canvas-first HOST-BLOCKED proof covers textured, Draco, and meshopt assets', async () => {
  const manifest = JSON.parse(await readFile(ptWebgl2BrowserManifestPath, 'utf8'));
  const canvasFirstStatus = JSON.parse(await readFile(ptWebgl2CanvasFirstStatusPath, 'utf8'));

  assert.equal(manifest.kind, 'vitrum-browser-gltf-pt-webgl2-goldens');
  assert.equal(manifest.backend, 'pt-webgl2');
  assert.equal(manifest.browserHarness, 'tools/gltf-browser-proof/capture-pt-webgl2-real.mjs');
  assert.deepEqual(manifest.resolution, [64, 64]);
  assert.equal(manifest.samplesPerPixel, 1);
  assert.deepEqual(
    manifest.assets.map((asset) => [asset.assetId, asset.kind, asset.goldenPath]),
    [
      ['box-textured-glb', 'textured-glb', 'tools/reference-renders/gltf-real-browser-pt-webgl2/pt-webgl2-gltf-real-box-textured.png'],
      ['cesium-milk-truck-draco', 'draco', 'tools/reference-renders/gltf-real-browser-pt-webgl2/pt-webgl2-gltf-real-draco.png'],
      ['meshopt-cube-real', 'meshopt', 'tools/reference-renders/gltf-real-browser-pt-webgl2/pt-webgl2-gltf-real-meshopt.png'],
    ],
  );

  const captureMode = 'canvas-first';
  const hostBlockClass = 'browser-canvas-readback-timeout';
  assert.equal(canvasFirstStatus.verdict, 'HOST-BLOCKED');
  assert.equal(canvasFirstStatus.backend, 'pt-webgl2');
  assert.equal(canvasFirstStatus.captureMode, captureMode);
  assert.deepEqual(canvasFirstStatus.hostBlockClasses, [hostBlockClass]);
  assert.equal(canvasFirstStatus.assets.length, manifest.assets.length);
  for (const asset of manifest.assets) {
    const row = canvasFirstStatus.assets.find((candidate) => candidate.assetId === asset.assetId);
    assert.ok(row, `${captureMode} should include ${asset.assetId}`);
    assert.equal(row.verdict, 'HOST-BLOCKED');
    assert.equal(row.captureMode, captureMode);
    assert.equal(row.hostBlockClass, hostBlockClass);
    assert.equal(row.telemetry.assetId, asset.assetId);
    assert.equal(row.telemetry.backend, 'pt-webgl2');
    assert.equal(row.telemetry.realAssetReady, true);
    for (const ext of asset.requiredExtensions ?? []) {
      assert.ok(row.telemetry.extensionsUsed.includes(ext), `${asset.assetId} should report ${ext}`);
    }
    for (const hook of asset.requiredHooks ?? []) {
      assert.equal(row.telemetry.browserDecodeHooks[hook], true, `${asset.assetId} should report ${hook} hook`);
    }
  }
});

test('renderer fidelity engine-first diagnostic fails closed on an uninformative capture', async () => {
  const status = JSON.parse(await readFile(ptWebgl2BrowserStatusPath, 'utf8'));

  assert.equal(status.verdict, 'FAIL');
  assert.equal(status.backend, 'pt-webgl2');
  assert.equal(status.captureMode, 'engine-first');
  assert.equal(status.assetCount, status.assets.length);
  assert.ok(status.assets.length > 0);
  for (const row of status.assets) {
    assert.equal(row.verdict, 'FAIL');
    assert.ok(
      row.captureAttempts.some((attempt) => attempt.status === 'succeeded'),
      `${row.assetId} should preserve the successful readback that produced the rejected frame`,
    );
    assert.match(row.error, /capture is visually uninformative/);
    assert.ok(
      row.structure.lumaRange < row.structure.thresholds.minLumaRange ||
        row.structure.uniqueColorCount < row.structure.thresholds.minUniqueColorCount ||
        row.structure.nonDominantFraction < row.structure.thresholds.minNonDominantFraction,
      `${row.assetId} should fail at least one visual-structure threshold`,
    );
  }
});
