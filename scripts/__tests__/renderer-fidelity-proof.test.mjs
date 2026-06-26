import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const checkerPath = join(repoRoot, 'tools', 'renderer-fidelity-proof', 'check-proofs.mjs');
const promotionStatusPath = join(repoRoot, 'tools', 'renderer-fidelity-proof', 'promotion-status.json');
const ptWebgl2BrowserStatusPath = join(repoRoot, 'tools', 'gltf-browser-proof', 'pt-webgl2-real-status.json');
const ptWebgl2CanvasFirstStatusPath = join(repoRoot, 'tools', 'gltf-browser-proof', 'pt-webgl2-real-canvas-first-status.json');
const ptWebgl2BrowserManifestPath = join(repoRoot, 'tools', 'reference-renders', 'gltf-real-browser-pt-webgl2', 'manifest.json');
const queuePath = join(repoRoot, 'tools', 'road-to-100', 'validation-queue.json');
const validationQueueCheckerPath = join(repoRoot, 'tools', 'road-to-100', 'check-validation-queue.mjs');

test('renderer fidelity promotion status stays partial while pt-webgl2 browser proof is host-blocked', async () => {
  const status = JSON.parse(await readFile(promotionStatusPath, 'utf8'));

  assert.equal(status.harness, 'renderer-fidelity-promotion-proof');
  assert.equal(status.verdict, 'PASS-PARTIAL');
  assert.equal(status.ptWebgl2.browserPromotionReady, false);
  assert.equal(status.ptWebgl2.browserStatus, 'HOST-BLOCKED');
  assert.deepEqual(status.ptWebgl2.hostBlockClasses, [
    'browser-canvas-readback-timeout',
    'engine-readback-timeout',
  ]);
  assert.equal(status.ptWebgl2.browserReadbackPreflight.statusByCaptureMode['engine-first'], 'PASS');
  assert.equal(status.ptWebgl2.browserReadbackPreflight.statusByCaptureMode['canvas-first'], 'PASS');
  assert.equal(status.ptWebgl2.browserReadbackPreflight.webgl2, true);
  assert.equal(status.ptWebgl2.browserReadbackPreflight.unsignedByteReadback, 'PASS');
  assert.equal(status.ptWebgl2.browserReadbackPreflight.floatReadback, 'PASS');
  assert.equal(status.ptWebgl2.browserReadbackPreflight.dataUrl, 'PASS');
  assert.match(
    status.ptWebgl2.browserReadbackPreflight.scope,
    /real glTF page browser capture remains HOST-BLOCKED/,
  );
  assert.equal(status.ptWebgl2.nonPromotionGradeCount, 11);
  assert.equal(status.ptWebgl2.materialFurnaceSourceOracleGroupCount, 4);
  assert.match(status.ptWebgl2.requiredEvidence, /browser\/real-adapter reference A\/B/);

  assert.equal(status.ptWebgpuFullTier.bdptBoundary.safeDefault, 'endpoint-only');
  assert.equal(status.ptWebgpuFullTier.bdptBoundary.multiVertexDefaultReady, false);
  assert.equal(
    status.ptWebgpuFullTier.bdptBoundary.multiVertexCurrentEstimator,
    'additive-sidecar-not-weighted-against-eye-path',
  );
  assert.equal(
    status.ptWebgpuFullTier.bdptBoundary.multiVertexBlocker,
    'not-weighted-against-regular-eye-path-strategy',
  );
});

test('renderer fidelity checker fail-closes browser promotion and source-only pt-webgl2 proofs', async () => {
  const checker = await readFile(checkerPath, 'utf8');
  const validationQueueChecker = await readFile(validationQueueCheckerPath, 'utf8');

  assert.match(checker, /must stay PASS-PARTIAL until pt-webgl2 browser promotion evidence lands/);
  assert.match(checker, /ptWebgl2\.browserPromotionReady must remain false while browser capture is blocked/);
  assert.match(checker, /ptWebgl2 browser status artifacts must agree before promotion accounting/);
  assert.match(checker, /browserReadbackPreflight\.statusByCaptureMode must match browser proof probes/);
  assert.match(checker, /browser proof must preserve passing WebGL2\/readPixels\/toDataURL preflight diagnostics/);
  assert.match(checker, /ptWebgl2\.requiredEvidence must name browser\/real-adapter reference A\/B/);
  assert.match(checker, /pt-webgl2 must not be marked supported while/);
  assert.match(checker, /PT_WEBGL2_MATERIAL_FURNACE_PROOFS/);
  assert.match(validationQueueChecker, /REQUIRED_RENDERER_FIDELITY_SOURCE_STATUS_PATHS/);
  assert.match(validationQueueChecker, /vitrum\.renderer-fidelity\.promotion-provenance\.v1/);
  assert.match(validationQueueChecker, /rendererPromotionProvenance/);
  assert.match(validationQueueChecker, /sourceStatusSha256/);
  assert.match(validationQueueChecker, /assertFileSha256/);
  assert.match(validationQueueChecker, /promotion provenance checkerSha256/);
  assert.match(validationQueueChecker, /promotion provenance sourceStatusSha256/);
  assert.match(checker, /PT_WEBGL2_EXPECTED_BROWSER_CAPTURE_MODES/);
  assert.match(checker, /assertPtWebgl2BrowserHostBlockedStatus/);
  assert.match(checker, /must contain one row per manifest asset/);
  assert.match(checker, /telemetry must prove realAssetReady=true/);
});

test('renderer fidelity browser HOST-BLOCKED proof still covers textured, Draco, and meshopt assets', async () => {
  const manifest = JSON.parse(await readFile(ptWebgl2BrowserManifestPath, 'utf8'));
  const engineFirstStatus = JSON.parse(await readFile(ptWebgl2BrowserStatusPath, 'utf8'));
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

  for (const [status, captureMode, hostBlockClass] of [
    [engineFirstStatus, 'engine-first', 'engine-readback-timeout'],
    [canvasFirstStatus, 'canvas-first', 'browser-canvas-readback-timeout'],
  ]) {
    assert.equal(status.verdict, 'HOST-BLOCKED');
    assert.equal(status.backend, 'pt-webgl2');
    assert.equal(status.captureMode, captureMode);
    assert.deepEqual(status.hostBlockClasses, [hostBlockClass]);
    assert.equal(status.assets.length, manifest.assets.length);
    for (const asset of manifest.assets) {
      const row = status.assets.find((candidate) => candidate.assetId === asset.assetId);
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
  }
});

test('Road renderer fidelity row cites the promotion guard artifact', async () => {
  const queue = JSON.parse(await readFile(queuePath, 'utf8'));
  const row = queue.validationQueue.find((candidate) => candidate.id === 'VQ-RENDERER-FIDELITY-PROOF');
  assert.ok(row, 'VQ-RENDERER-FIDELITY-PROOF row should exist');
  assert.equal(row.status, 'partial-proof-green');

  const promotionArtifact = row.proofArtifacts.find((candidate) =>
    candidate.path === 'tools/renderer-fidelity-proof/promotion-status.json'
  );
  assert.ok(promotionArtifact, 'renderer fidelity row should cite promotion-status.json');
  assert.deepEqual(promotionArtifact.json, {
    verdict: 'PASS-PARTIAL',
    'ptWebgpuFullTier.supportedRowCount': 10,
    'ptWebgpuFullTier.bdptBoundary.safeDefault': 'endpoint-only',
    'ptWebgpuFullTier.bdptBoundary.multiVertexDefaultReady': false,
    'ptWebgpuFullTier.bdptBoundary.multiVertexCurrentEstimator': 'additive-sidecar-not-weighted-against-eye-path',
    'ptWebgpuFullTier.bdptBoundary.multiVertexBlocker': 'not-weighted-against-regular-eye-path-strategy',
    'ptWebgpuFullTier.bdptBoundary.multiVertexSafeAlternative': 'omit bdptOptions.maxLightBounces or set maxLightBounces:1',
    'ptWebgl2.browserPromotionReady': false,
    'ptWebgl2.browserStatus': 'HOST-BLOCKED',
    'ptWebgl2.hostBlockClasses.0': 'browser-canvas-readback-timeout',
    'ptWebgl2.hostBlockClasses.1': 'engine-readback-timeout',
    'ptWebgl2.browserReadbackPreflight.statusByCaptureMode.engine-first': 'PASS',
    'ptWebgl2.browserReadbackPreflight.statusByCaptureMode.canvas-first': 'PASS',
    'ptWebgl2.browserReadbackPreflight.webgl2': true,
    'ptWebgl2.browserReadbackPreflight.unsignedByteReadback': 'PASS',
    'ptWebgl2.browserReadbackPreflight.floatReadback': 'PASS',
    'ptWebgl2.browserReadbackPreflight.dataUrl': 'PASS',
    'ptWebgl2.nonPromotionGradeCount': 11,
    'ptWebgl2.materialFurnaceSourceOracleGroupCount': 4,
  });

  assert.match(row.remaining, /Browser\/real-adapter reference A\/B/);
  assert.match(row.remaining, /browser\/WebGL2 capture is HOST-BLOCKED/);
});
