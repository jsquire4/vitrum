import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const checkerPath = join(repoRoot, 'tools', 'renderer-fidelity-proof', 'check-proofs.mjs');
const promotionStatusPath = join(repoRoot, 'tools', 'renderer-fidelity-proof', 'promotion-status.json');
const queuePath = join(repoRoot, 'tools', 'road-to-100', 'validation-queue.json');

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

  assert.match(checker, /must stay PASS-PARTIAL until pt-webgl2 browser promotion evidence lands/);
  assert.match(checker, /ptWebgl2\.browserPromotionReady must remain false while browser capture is blocked/);
  assert.match(checker, /ptWebgl2 browser status artifacts must agree before promotion accounting/);
  assert.match(checker, /browserReadbackPreflight\.statusByCaptureMode must match browser proof probes/);
  assert.match(checker, /browser proof must preserve passing WebGL2\/readPixels\/toDataURL preflight diagnostics/);
  assert.match(checker, /ptWebgl2\.requiredEvidence must name browser\/real-adapter reference A\/B/);
  assert.match(checker, /pt-webgl2 must not be marked supported while/);
  assert.match(checker, /PT_WEBGL2_MATERIAL_FURNACE_PROOFS/);
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
