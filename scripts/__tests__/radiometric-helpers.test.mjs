import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const helpersPath = join(repoRoot, 'tools', 'radiometric-ab', 'helpers.mjs');
const walkaroundHarnessPath = join(repoRoot, 'tools', 'radiometric-ab', 'walkaround-ab.mjs');
const walkaroundRunnerPath = join(repoRoot, 'tools', 'radiometric-ab', 'run-walkaround-ab.mjs');
const ptRunnerPath = join(repoRoot, 'tools', 'radiometric-ab', 'run-pt-ab.mjs');
const ptBdptHarnessPath = join(repoRoot, 'tools', 'radiometric-ab', 'ab-bdpt.mjs');
const radiometricProofsPath = join(repoRoot, 'tools', 'radiometric-ab', 'proofs.mjs');
const radiometricCheckerPath = join(repoRoot, 'tools', 'radiometric-ab', 'check-results.mjs');
const radiometricProvenancePath = join(repoRoot, 'tools', 'radiometric-ab', 'resultProvenance.mjs');
const validationQueueCheckerPath = join(repoRoot, 'tools', 'road-to-100', 'check-validation-queue.mjs');
const ptWebgpuIndexPath = join(repoRoot, 'packages', 'pt-webgpu', 'src', 'index.ts');
const ptWebgpuKernelPath = join(repoRoot, 'packages', 'pt-webgpu', 'src', 'wgsl', 'pathTrace', 'kernel.wgsl.ts');

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

test('walkaround A/B wrapper exposes the all-cases 64-SPP proof lane', async () => {
  const runner = await readFile(walkaroundRunnerPath, 'utf8');
  assert.match(runner, /--all-spp64/);
  assert.match(runner, /walkaround-ab-all-spp64-status\.json/);
  assert.match(runner, /walkaround-ab-all-spp64\.json/);
  assert.match(runner, /all-spp64/);
  assert.match(runner, /glossySpp64 && allSpp64/);
});

test('walkaround A/B wrapper refreshes promotion status after default proof recaptures', async () => {
  const runner = await readFile(walkaroundRunnerPath, 'utf8');
  assert.match(runner, /walkaround-ab-promotion-status\.json/);
  assert.match(runner, /function buildPromotionStatus/);
  assert.match(runner, /function maybeWritePromotionStatus/);
  assert.match(runner, /usingDefaultProofPaths/);
  assert.match(runner, /sourceStatuses: DEFAULT_SOURCE_STATUS_PATHS/);
  assert.match(runner, /glassProfiles/);
  assert.match(runner, /glossyProfiles/);
  assert.match(runner, /HOST-BLOCKED' \|\| status\.verdict === 'FAIL'/);
});

test('walkaround A/B wrapper fails closed on missing or incomplete result artifacts', async () => {
  const runner = await readFile(walkaroundRunnerPath, 'utf8');
  assert.match(runner, /function readWalkaroundResultsForStatus/);
  assert.match(runner, /function failClosedResultStatus/);
  assert.match(runner, /walkaround-ab-missing-result/);
  assert.match(runner, /walkaround-ab-invalid-result/);
  assert.match(runner, /walkaround-ab-incomplete-result/);
  assert.match(runner, /typeof row\.verdict !== 'string'/);
  assert.match(runner, /verdict: 'FAIL'/);
});


test('Road checker pins radiometric promotion provenance blocks', async () => {
  const validationQueueChecker = await readFile(validationQueueCheckerPath, 'utf8');

  assert.match(validationQueueChecker, /REQUIRED_PT_RADIOMETRIC_PROMOTION_SOURCE_STATUS_PATHS/);
  assert.match(validationQueueChecker, /vitrum\.pt-radiometric-ab\.promotion-provenance\.v1/);
  assert.match(validationQueueChecker, /ptRadiometricPromotionProvenance/);
  assert.match(validationQueueChecker, /REQUIRED_WALKAROUND_PROMOTION_SOURCE_STATUS_PATHS/);
  assert.match(validationQueueChecker, /REQUIRED_WALKAROUND_PROMOTION_SOURCE_RESULT_PATHS/);
  assert.match(validationQueueChecker, /vitrum\.walkaround-ab\.promotion-provenance\.v1/);
  assert.match(validationQueueChecker, /walkaroundPromotionProvenance/);
  assert.match(validationQueueChecker, /assertSha256DigestRows/);
  assert.match(validationQueueChecker, /assertFileSha256/);
  assert.match(validationQueueChecker, /VQ-RADIOMETRIC-PT promotion provenance wrapperSha256/);
  assert.match(validationQueueChecker, /VQ-RADIOMETRIC-PT promotion provenance sourceStatusSha256/);
  assert.match(validationQueueChecker, /function assertPtRadiometricHostStatusCapture/);
  assert.match(validationQueueChecker, /VQ-RADIOMETRIC-PT host status must pin the native WebGPU PT A\/B command/);
  assert.match(validationQueueChecker, /VQ-RADIOMETRIC-PT host status provenance resultSha256/);
  assert.match(validationQueueChecker, /vitrum\.pt-radiometric-ab\.status-provenance\.v1/);
  assert.match(validationQueueChecker, /VQ-WALKAROUND-RADIOMETRIC-AB promotion provenance wrapperSha256/);
  assert.match(validationQueueChecker, /VQ-WALKAROUND-RADIOMETRIC-AB promotion provenance sourceStatusSha256/);
  assert.match(validationQueueChecker, /VQ-WALKAROUND-RADIOMETRIC-AB promotion provenance sourceResultSha256/);
  assert.match(validationQueueChecker, /function assertWalkaroundStatusCapture/);
  assert.match(validationQueueChecker, /walkaround status capture/);
  assert.match(validationQueueChecker, /walkaround-ab-glossy-spp64-status\.json/);
  assert.match(validationQueueChecker, /selectedCases=\$\{expected\.selectedCases\}/);
  assert.match(validationQueueChecker, /browser-real-adapter next steps/);
});
test('pt radiometric wrapper refreshes promotion status after complete recapture', async () => {
  const runner = await readFile(ptRunnerPath, 'utf8');
  assert.match(runner, /pt-promotion-status\.json/);
  assert.match(runner, /function buildPromotionStatus/);
  assert.match(runner, /function maybeWritePromotionStatus/);
  assert.match(runner, /hostStatus\.verdict !== 'PASS'/);
  assert.match(runner, /selectedAllCases/);
  assert.match(runner, /safeDefaultProofs/);
  assert.match(runner, /researchFindings/);
  assert.match(runner, /sourceStatuses: SOURCE_STATUS_PATHS/);
  assert.match(runner, /ptRadiometricStatusProvenance/);
  assert.match(runner, /ptRadiometricPromotionProvenance/);
  assert.match(runner, /status\.provenance = await ptRadiometricStatusProvenance/);
  assert.match(runner, /provenance: await ptRadiometricPromotionProvenance/);
});

test('pt radiometric wrapper fails closed on missing or incomplete result artifacts', async () => {
  const runner = await readFile(ptRunnerPath, 'utf8');
  assert.match(runner, /function passResultArtifactProblem/);
  assert.match(runner, /pt-radiometric-ab-missing-result/);
  assert.match(runner, /pt-radiometric-ab-invalid-result/);
  assert.match(runner, /pt-radiometric-ab-incomplete-result/);
  assert.match(runner, /typeof payload\.verdict !== 'string'/);
  assert.match(runner, /artifactProblem \?\? hostBoundary/);
});

test('pt BDPT radiometric proof keeps multi-vertex mode research-only', async () => {
  const harness = await readFile(ptBdptHarnessPath, 'utf8');
  const proofs = await readFile(radiometricProofsPath, 'utf8');
  const checker = await readFile(radiometricCheckerPath, 'utf8');
  const provenance = await readFile(radiometricProvenancePath, 'utf8');
  const ptWebgpuIndex = await readFile(ptWebgpuIndexPath, 'utf8');
  const ptWebgpuKernel = await readFile(ptWebgpuKernelPath, 'utf8');

  assert.match(harness, /CONTROL_MAX_LIGHT_BOUNCES = \[1, 2, 3\]/);
  assert.match(harness, /experimentalMultiVertex: true/);
  assert.match(harness, /currentEstimator: "additive-sidecar-not-weighted-against-eye-path"/);
  assert.match(harness, /requiredEstimator: "multi-vertex-light-subpath-strategies-weighted-against-regular-eye-path-strategy"/);
  assert.match(harness, /researchFindings:\s*\{\s*bdptMultiVertex: multiVertexResearchFinding/);

  assert.match(proofs, /export const BDPT_MULTIVERTEX_RESEARCH_PROOF/);
  assert.match(proofs, /warningCode: "pt-webgpu\.bdpt-multivertex-research-mode"/);
  assert.match(proofs, /shaderSourcePath: "packages\/pt-webgpu\/src\/wgsl\/pathTrace\/kernel\.wgsl\.ts"/);
  assert.match(proofs, /"for \(var lvi = 1u; lvi < maxLv; lvi\+\+\)"/);
  assert.match(proofs, /"radiance = radiance \+ evaluateBdptConnection\("/);
  assert.match(proofs, /minFindingGlobalRelErr: 0\.10/);

  assert.match(checker, /function checkBdptMultiVertexResearch/);
  assert.match(checker, /result must carry controls\.multiVertexPromotion metadata/);
  assert.match(checker, /result must carry researchFindings\.bdptMultiVertex metadata/);
  assert.match(checker, /firstFindingGlobalRelErr differs from control run/);
  assert.match(checker, /source warning missing/);
  assert.match(checker, /shader source missing/);
  assert.match(checker, /promotionReady: false/);
  assert.match(checker, /ptRadiometricStatusProvenance/);
  assert.match(checker, /ptRadiometricPromotionProvenance/);
  assert.match(checker, /pt-radiometric-ab: status provenance differs from current wrapper\/result identity/);
  assert.match(checker, /pt-radiometric-promotion: provenance differs from current source artifact identity/);

  assert.match(provenance, /PT_RADIOMETRIC_STATUS_PROVENANCE_SCHEMA/);
  assert.match(provenance, /PT_RADIOMETRIC_PROMOTION_PROVENANCE_SCHEMA/);
  assert.match(provenance, /ptRadiometricStatusProvenance/);
  assert.match(provenance, /ptRadiometricPromotionProvenance/);

  assert.match(ptWebgpuIndex, /bdptOptions\.maxLightBounces > 1 activates the multi-vertex BDPT research path/);
  assert.match(ptWebgpuIndex, /promotionReady: false/);
  assert.match(ptWebgpuIndex, /currentEstimator: 'additive-sidecar-not-weighted-against-eye-path'/);

  assert.match(ptWebgpuKernel, /for \(var lvi = 1u; lvi < maxLv; lvi\+\+\) \{/);
  assert.match(ptWebgpuKernel, /radiance = radiance \+ evaluateBdptConnection\(/);
});

async function loadVarianceROI() {
  const source = await readFile(helpersPath, 'utf8');
  const match = source.match(/export function varianceROI\(([^)]*)\) \{([\s\S]*?)\n\}/);
  assert.ok(match, 'varianceROI export should be present in helpers.mjs');
  return Function(`return function varianceROI(${match[1]}) {${match[2]}\n};`)();
}
