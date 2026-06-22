#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
// Verifies the Road-to-100 work queue is explicit, machine-readable, and not
// silently turning validation/provisioning/future-contract tails into code gaps.

const QUEUE_PATH = "tools/road-to-100/validation-queue.json";
const PACKAGE_PATH = "package.json";
const EXECUTION_PLAN_PATH = "plan/gap-closure-execution-plan.md";
const LEDGER_PATH = "plan/road-to-100-gap-ledger-2026-06-11.md";
const PROMISE_LEDGER_PATH = "packages/core/src/engine/promiseLedger.ts";
const LEARNED_CHECKPOINT_MANIFEST_PATH = "tools/neural-denoiser-training/checkpoints/manifest.json";

const ALLOWED_STATUSES = new Set([
  "committed-proof-green",
  "partial-proof-green",
  "host-blocked",
  "evidence-needed",
  "provisioning-needed",
  "decision-needed",
  "future-contract",
]);

const REQUIRED_VALIDATION_IDS = [
  "VQ-PT-WEBGPU-RUNTIME-GOLDENS",
  "VQ-WALKAROUND-BEHAVIORAL-MATRIX",
  "VQ-MUTATION-MATRIX",
  "VQ-GLTF-REAL-WEBGPU",
  "VQ-GLTF-BROWSER-PTWEBGL2",
  "VQ-RADIOMETRIC-PT",
  "VQ-WALKAROUND-RADIOMETRIC-AB",
  "VQ-RENDERER-FIDELITY-PROOF",
  "VQ-ADJOINT-SCOPED-PATH-REPLAY",
  "VQ-LEARNED-SYSTEMS",
  "VQ-GLTF-MATERIAL-TOPOLOGY",
];

const REQUIRED_FUTURE_IDS = [
  "FC-DISPLACEMENT-MICROTESSELLATION",
  "FC-TRANSPARENT-GI-TRANSPORT",
  "FC-WALKAROUND-SPECIALTY-MATERIAL-TRANSPORT",
  "FC-NATIVE-POINT-LINE",
  "FC-ARBITRARY-UV-ARRAYS",
  "FC-NATIVE-INSTANCED-SKINNED-MORPHED",
  "FC-ADJOINT-FULL-PATH-PARITY",
];

const REQUIRED_FUTURE_BLOCKER_NEEDLES = new Map([
  ["FC-DISPLACEMENT-MICROTESSELLATION", ["tessellation", "BVH"]],
  ["FC-TRANSPARENT-GI-TRANSPORT", ["reservoir", "DDGI/RC"]],
  ["FC-WALKAROUND-SPECIALTY-MATERIAL-TRANSPORT", ["spectral", "PT backends"]],
  ["FC-NATIVE-POINT-LINE", ["core point/line", "backend fidelity"]],
  ["FC-ARBITRARY-UV-ARRAYS", ["TextureRef", "shader descriptor"]],
  ["FC-NATIVE-INSTANCED-SKINNED-MORPHED", ["instanced-skinned", "TLAS/BLAS"]],
  ["FC-ADJOINT-FULL-PATH-PARITY", ["differentiable transport", "scoped direct-light replay"]],
]);

const REQUIRED_MUTATION_KINDS = [
  "material",
  "environment",
  "emitter",
  "transform",
  "topology",
  "instanced-count",
  "add-primitive",
  "remove-primitive",
];

const REQUIRED_RENDERER_FIDELITY_ARTIFACT_PATHS = [
  "tools/renderer-fidelity-proof/check-proofs.mjs",
  "plan/renderer-fidelity-matrix.md",
  "plan/fidelity-promotion-playbook.md",
  "README.md",
  "plan/library-architecture.md",
  "HARDWARE-VALIDATION-NEEDS.md",
  "plan/gap-closure-execution-plan.md",
  "tools/gltf-browser-proof/pt-webgl2-real-status.json",
  "tools/reference-renders/gltf-real-browser-pt-webgl2/manifest.json",
  "tools/behavioral-gate/behavioral-gate-dzn-spectral-status.json",
  "tools/behavioral-gate/behavioral-gate-dzn-light-status.json",
  "tools/behavioral-gate/behavioral-gate-dzn-caustic-status.json",
  "tools/behavioral-gate/behavioral-gate-dzn-bdpt-status.json",
  "tools/reference-renders/baseline/ptwgpu-spectral-hero.png",
  "tools/reference-renders/baseline/ptwgpu-thinfilm-angle.png",
  "tools/reference-renders/baseline/ptwgpu-cauchy-dispersion.png",
  "tools/reference-renders/baseline/ptwgpu-layered-front.png",
  "tools/reference-renders/baseline/ptwgpu-sss-mixed-panels.png",
  "tools/reference-renders/baseline/cornell-manylights.png",
  "tools/reference-renders/baseline/ptwgpu-parity-material-fields.png",
  "tools/reference-renders/baseline/mnee-glass-slab.png",
  "tools/reference-renders/baseline/cornell-bdpt-on.png",
  "packages/pt-webgl2/src/glsl/shader/bsdf/__tests__/b9Multiscatter.test.ts",
  "packages/pt-webgl2/src/glsl/shader/bsdf/ggx_functions.glsl.js",
  "packages/pt-webgl2/src/glsl/shader/bsdf/bsdf_functions.glsl.js",
  "packages/pt-webgl2/src/glsl/composeTraceGlsl.test.ts",
  "packages/pt-webgl2/src/scene/materialsTexture.test.ts",
  "packages/pt-webgl2/src/glsl/render/get_surface_record_function.glsl.js",
  "packages/pt-webgl2/src/glsl/render/attenuate_hit_function.glsl.js",
  "packages/pt-webgl2/src/capabilities.ts",
  "packages/pt-webgl2/src/scene/equirectHdrInfo.ts",
  "packages/pt-webgl2/src/scene/equirectHdrInfo.test.ts",
  "packages/pt-webgl2/src/scene/meshAreaLights.test.ts",
  "packages/pt-webgl2/src/scene/meshAreaMis.test.ts",
  "packages/pt-webgl2/src/glsl/shader/sampling/light_sampling_functions.glsl.js",
];

const REQUIRED_WALKAROUND_AB_ARTIFACT_PATHS = [
  "tools/radiometric-ab/check-results.mjs",
  "tools/radiometric-ab/proofs.mjs",
  "tools/radiometric-ab/README.md",
  "tools/radiometric-ab/walkaround-ab.mjs",
  "tools/radiometric-ab/walkaround-ab-host-status.json",
  "tools/radiometric-ab/walkaround-ab-results.json",
  "packages/walkaround-hybrid/src/HybridEngineOptions.ts",
  "packages/walkaround-hybrid/src/shaders/ggxBrdf.wgsl.ts",
  "packages/walkaround-hybrid/src/shaders/shadingTerms.wgsl.ts",
  "packages/walkaround-hybrid/src/shaders/shade.wgsl.ts",
  "packages/walkaround-hybrid/src/shaders/risGi.wgsl.ts",
  "packages/walkaround-hybrid/src/shaders/__tests__/b1GlossyMetalGi.test.ts",
];

/** @param {string} path */
function repoUrl(path) {
  return new URL(`../../${path}`, import.meta.url);
}

/** @param {string} message @returns {never} */
function fail(message) {
  throw new Error(`[road-to-100-validation-status] ${message}`);
}

/** @param {string} path */
async function readText(path) {
  return await Deno.readTextFile(repoUrl(path));
}

/** @param {string} path */
async function readJson(path) {
  return JSON.parse(await readText(path));
}

/** @param {string} path */
async function assertFile(path) {
  const stat = await Deno.stat(repoUrl(path));
  if (!stat.isFile) fail(`${path} is missing or not a file`);
  if (stat.size <= 0) fail(`${path} is empty`);
}

/**
 * @param {unknown} value
 * @param {string} label
 */
function assertNonEmptyString(value, label) {
  if (typeof value !== "string" || value.length === 0) fail(`${label} must be a non-empty string`);
}

/**
 * @param {readonly unknown[]} rows
 * @param {readonly string[]} requiredIds
 * @param {string} label
 */
function assertRequiredIds(rows, requiredIds, label) {
  const ids = new Set(rows.map((row) => row && typeof row === "object" ? String(row.id ?? "") : ""));
  for (const id of requiredIds) {
    if (!ids.has(id)) fail(`${label} missing ${id}`);
  }
  if (ids.size !== rows.length) fail(`${label} contains duplicate or invalid ids`);
}

/**
 * @param {string} command
 * @param {Record<string, string>} scripts
 */
function assertCommandScriptsExist(command, scripts) {
  for (const part of command.split(/\s*&&\s*/)) {
    const match = part.trim().match(/^npm run ([^\s]+)/);
    if (!match) fail(`command must be an npm run script chain: ${command}`);
    const script = match[1];
    if (scripts[script] == null) fail(`command references missing package script "${script}"`);
  }
}

/**
 * @param {unknown} object
 * @param {string} path
 */
function getJsonPath(object, path) {
  let cur = object;
  for (const part of path.split(".")) {
    if (cur == null || typeof cur !== "object" || !(part in cur)) return undefined;
    cur = cur[part];
  }
  return cur;
}

/**
 * @param {unknown} artifact
 * @param {string} ownerId
 */
async function assertArtifact(artifact, ownerId) {
  if (artifact == null || typeof artifact !== "object") fail(`${ownerId}: artifact must be an object`);
  const path = artifact.path;
  assertNonEmptyString(path, `${ownerId}: artifact.path`);
  await assertFile(path);

  if (artifact.type === "png" || String(path).endsWith(".png")) {
    const bytes = await Deno.readFile(repoUrl(path));
    if (bytes[0] !== 0x89 || bytes[1] !== 0x50 || bytes[2] !== 0x4e || bytes[3] !== 0x47) {
      fail(`${ownerId}: ${path} is not a PNG`);
    }
  }

  if (artifact.json != null) {
    if (typeof artifact.json !== "object" || Array.isArray(artifact.json)) {
      fail(`${ownerId}: ${path} json expectations must be an object`);
    }
    const json = await readJson(path);
    for (const [jsonPath, expected] of Object.entries(artifact.json)) {
      const actual = getJsonPath(json, jsonPath);
      if (actual !== expected) {
        fail(`${ownerId}: ${path} ${jsonPath} expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
      }
    }
  }
}

/**
 * @param {unknown} status
 * @param {"pt" | "wh"} prefix
 * @param {string} path
 */
function assertMutationStatusCoverage(status, prefix, path) {
  if (status == null || typeof status !== "object") fail(`${path} must be a status object`);
  if (status.verdict !== "PASS") fail(`${path} must pin verdict PASS`);
  if (status.exitStatus !== 0) fail(`${path} must pin exitStatus 0`);
  if (status.goldenVariant !== "dzn-full") fail(`${path} must pin dzn-full goldenVariant`);
  if (status.summary?.totalConfigs !== REQUIRED_MUTATION_KINDS.length) {
    fail(`${path} must contain ${REQUIRED_MUTATION_KINDS.length} mutation configs`);
  }
  if (status.summary?.failures !== 0) fail(`${path} must pin zero failures`);
  if (!Array.isArray(status.configs)) fail(`${path} configs must be an array`);

  for (const mutationKind of REQUIRED_MUTATION_KINDS) {
    const label = `${prefix}/mutation-${mutationKind}`;
    const config = status.configs.find((item) => item?.label === label);
    if (config == null) fail(`${path} missing mutation config ${label}`);
    if (config.verdict !== "PASS") fail(`${path} ${label} must pin verdict PASS`);
    if (config.rawStatus !== "OK") fail(`${path} ${label} must pin rawStatus OK`);
    if (config.mutationKind !== mutationKind) fail(`${path} ${label} must pin mutationKind ${mutationKind}`);
    if (config.goldenStatus !== "ok") fail(`${path} ${label} must pin goldenStatus ok`);
    if (config.goldenVariant !== "dzn-full") fail(`${path} ${label} must pin goldenVariant dzn-full`);
    if (typeof config.mutationMeanAbs !== "number" || config.mutationMeanAbs < 2) {
      fail(`${path} ${label} must retain an observable mutationMeanAbs >= 2`);
    }
    if (typeof config.mutationMaxAbs !== "number" || config.mutationMaxAbs < 8) {
      fail(`${path} ${label} must retain an observable mutationMaxAbs >= 8`);
    }
  }
}

const [queue, packageJson, executionPlan, ledger, promiseLedger, road] = await Promise.all([
  readJson(QUEUE_PATH),
  readJson(PACKAGE_PATH),
  readText(EXECUTION_PLAN_PATH),
  readText(LEDGER_PATH),
  readText(PROMISE_LEDGER_PATH),
  readText("plan/road-to-100.md"),
]);
const inverseSessionSource = await readText("packages/pt-webgpu/src/inverse/inverseSession.ts");
const inverseSessionTestSource = await readText("packages/pt-webgpu/src/__tests__/inverseSession.test.ts");
const learnedCheckpointManifest = await readJson(LEARNED_CHECKPOINT_MANIFEST_PATH);

if (queue.schema !== "vitrum.road-to-100.validation-queue.v1") fail("queue schema mismatch");
if (typeof queue.currentAsOf !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(queue.currentAsOf)) {
  fail("queue currentAsOf must be YYYY-MM-DD");
}
if (queue.currentAsOf < "2026-06-22") fail("queue currentAsOf predates the latest code-closure wave");

if (!Array.isArray(queue.implementationQueue)) fail("implementationQueue must be an array");
if (queue.implementationQueue.length !== 0) {
  fail("implementationQueue must stay empty unless a source-verified code bug is promoted");
}
if (!executionPlan.includes("implementation queue is empty")) {
  fail("gap execution plan must retain the implementation-queue-empty source verdict");
}
if (!ledger.includes("not the active implementation queue")) {
  fail("compact ledger must keep items_to_fix out of the active implementation queue");
}

if (!Array.isArray(queue.validationQueue)) fail("validationQueue must be an array");
if (!Array.isArray(queue.futureContractRows)) fail("futureContractRows must be an array");
assertRequiredIds(queue.validationQueue, REQUIRED_VALIDATION_IDS, "validationQueue");
assertRequiredIds(queue.futureContractRows, REQUIRED_FUTURE_IDS, "futureContractRows");

const scripts = packageJson.scripts ?? {};
if (typeof scripts !== "object") fail("package.json scripts must be an object");

for (const row of queue.validationQueue) {
  if (row == null || typeof row !== "object") fail("validation row must be an object");
  assertNonEmptyString(row.id, "validation row id");
  assertNonEmptyString(row.title, `${row.id}: title`);
  assertNonEmptyString(row.kind, `${row.id}: kind`);
  assertNonEmptyString(row.status, `${row.id}: status`);
  assertNonEmptyString(row.remaining, `${row.id}: remaining`);
  if (!ALLOWED_STATUSES.has(row.status)) fail(`${row.id}: invalid status ${row.status}`);
  if (row.status === "future-contract") fail(`${row.id}: future-contract rows belong in futureContractRows`);
  if (row.command != null) {
    assertNonEmptyString(row.command, `${row.id}: command`);
    assertCommandScriptsExist(row.command, scripts);
  } else if (row.status !== "evidence-needed" && row.status !== "decision-needed") {
    fail(`${row.id}: non-decision validation rows need a command`);
  }
  if (row.promotionCommand != null) {
    assertNonEmptyString(row.promotionCommand, `${row.id}: promotionCommand`);
    assertCommandScriptsExist(row.promotionCommand, scripts);
  }
  if (!Array.isArray(row.proofArtifacts)) fail(`${row.id}: proofArtifacts must be an array`);
  for (const artifact of row.proofArtifacts) await assertArtifact(artifact, row.id);
}

const mutationRow = queue.validationQueue.find((row) => row.id === "VQ-MUTATION-MATRIX");
if (mutationRow == null) fail("validationQueue missing VQ-MUTATION-MATRIX");
if (!String(mutationRow.remaining).includes("observable before/after pixel deltas")) {
  fail("VQ-MUTATION-MATRIX remaining text must keep the observable mutation delta proof explicit");
}
if (!String(mutationRow.remaining).includes("committed dzn-full post-mutation goldens")) {
  fail("VQ-MUTATION-MATRIX remaining text must keep committed dzn-full goldens explicit");
}
const mutationArtifactPaths = new Set(mutationRow.proofArtifacts.map((artifact) => artifact?.path));
for (const mutationKind of REQUIRED_MUTATION_KINDS) {
  for (const prefix of ["pt", "wh"]) {
    const path = `tools/reference-renders/mutation-behavioral/${prefix}-mutation-${mutationKind}.dzn-full.png`;
    if (!mutationArtifactPaths.has(path)) fail(`VQ-MUTATION-MATRIX must cite ${path}`);
  }
}
assertMutationStatusCoverage(
  await readJson("tools/behavioral-gate/behavioral-gate-dzn-pt-mutation-status.json"),
  "pt",
  "tools/behavioral-gate/behavioral-gate-dzn-pt-mutation-status.json",
);
assertMutationStatusCoverage(
  await readJson("tools/behavioral-gate/behavioral-gate-dzn-wh-mutation-status.json"),
  "wh",
  "tools/behavioral-gate/behavioral-gate-dzn-wh-mutation-status.json",
);

const rendererFidelityRow = queue.validationQueue.find((row) => row.id === "VQ-RENDERER-FIDELITY-PROOF");
if (rendererFidelityRow == null) fail("validationQueue missing VQ-RENDERER-FIDELITY-PROOF");
if (!String(rendererFidelityRow.remaining).includes("pt-webgl2 non-promotion grades")) {
  fail("VQ-RENDERER-FIDELITY-PROOF remaining text must keep pt-webgl2 non-promotion explicit");
}
if (!String(rendererFidelityRow.remaining).includes("browser/WebGL2 capture is HOST-BLOCKED")) {
  fail("VQ-RENDERER-FIDELITY-PROOF remaining text must keep the browser/WebGL2 blocker explicit");
}
const rendererFidelityArtifactPaths = new Set(rendererFidelityRow.proofArtifacts.map((artifact) => artifact?.path));
for (const path of REQUIRED_RENDERER_FIDELITY_ARTIFACT_PATHS) {
  if (!rendererFidelityArtifactPaths.has(path)) {
    fail(`VQ-RENDERER-FIDELITY-PROOF proofArtifacts must cite ${path}`);
  }
}

const learnedRow = queue.validationQueue.find((row) => row.id === "VQ-LEARNED-SYSTEMS");
if (learnedRow == null) fail("validationQueue missing VQ-LEARNED-SYSTEMS");
for (const path of [
  "tools/learned-systems/check-status.mjs",
  "tools/neural-denoiser-training/checkpoints/manifest.json",
  "packages/walkaround-hybrid/src/HybridEngineConfig.ts",
  "packages/walkaround-hybrid/src/HybridEngineOptions.ts",
  "packages/walkaround-hybrid/src/HybridEngine.ts",
  "packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts",
  "packages/walkaround-hybrid/src/neural/weights.ts",
  "packages/walkaround-hybrid/src/__tests__/learnedSystemConfig.test.ts",
  "packages/walkaround-hybrid/src/__tests__/capabilitiesPartition.test.ts",
  "packages/walkaround-hybrid/src/__tests__/hybridLiteTier.test.ts",
  "packages/walkaround-hybrid/src/pipeline/__tests__/nrcStructuralGate.test.ts",
  "packages/walkaround-hybrid/src/pipeline/__tests__/nrcDeviceCapability.test.ts",
  "packages/walkaround-hybrid/src/neural/nrc/__tests__/nrcGateBitIdentity.test.ts",
  "packages/walkaround-hybrid/src/pipeline/__tests__/ppgCoordinatorDiagnostics.test.ts",
  "packages/walkaround-hybrid/src/pipeline/__tests__/ppgCompilerGate.test.ts",
  "README.md",
  "plan/library-architecture.md",
  "packages/walkaround-hybrid/README.md",
  "tools/neural-denoiser-training/README.md",
  "HARDWARE-VALIDATION-NEEDS.md",
]) {
  if (!learnedRow.proofArtifacts.some((artifact) => artifact?.path === path)) {
    fail(`VQ-LEARNED-SYSTEMS proofArtifacts must cite ${path}`);
  }
}
if (learnedCheckpointManifest.schema !== "vitrum.neural-denoiser.checkpoints.v1") {
  fail("learned checkpoint manifest schema mismatch");
}
const productionCheckpoint = learnedCheckpointManifest.productionCheckpoint ?? null;
if (productionCheckpoint === null) {
  if (learnedRow.status !== "provisioning-needed") {
    fail("VQ-LEARNED-SYSTEMS must remain provisioning-needed while productionCheckpoint is null");
  }
  if (!String(learnedRow.remaining).includes("Production neural checkpoint")) {
    fail("VQ-LEARNED-SYSTEMS remaining text must keep the production-checkpoint tail explicit");
  }
} else {
  if (typeof productionCheckpoint !== "string") {
    fail("learned checkpoint manifest productionCheckpoint must be null or a string");
  }
  const checkpoints = Array.isArray(learnedCheckpointManifest.checkpoints)
    ? learnedCheckpointManifest.checkpoints
    : [];
  const productionEntry = checkpoints.find((checkpoint) => checkpoint?.name === productionCheckpoint);
  if (productionEntry == null || productionEntry.role !== "production" || productionEntry.productionDefaultEligible !== true) {
    fail(`productionCheckpoint ${productionCheckpoint} must name a productionDefaultEligible production entry`);
  }
  if (learnedRow.status === "provisioning-needed") {
    fail("VQ-LEARNED-SYSTEMS must move off provisioning-needed once productionCheckpoint is populated");
  }
  if (!learnedRow.proofArtifacts.some((artifact) =>
    artifact?.path === "tools/neural-denoiser-training/quality-ab-production.json"
  )) {
    fail("VQ-LEARNED-SYSTEMS must cite quality-ab-production.json when a production checkpoint exists");
  }
}

const adjointRow = queue.validationQueue.find((row) => row.id === "VQ-ADJOINT-SCOPED-PATH-REPLAY");
if (adjointRow == null) fail("validationQueue missing VQ-ADJOINT-SCOPED-PATH-REPLAY");
for (const needle of [
  "path replay",
  "finite-difference",
  "transport",
  "visibility",
  "displacement geometry",
]) {
  if (!String(adjointRow.remaining).includes(needle)) {
    fail(`VQ-ADJOINT-SCOPED-PATH-REPLAY remaining text must include ${needle}`);
  }
}
if (!adjointRow.proofArtifacts.some((artifact) =>
  artifact?.path === "packages/pt-webgpu/src/inverse/inverseSession.ts"
)) {
  fail("VQ-ADJOINT-SCOPED-PATH-REPLAY must cite inverseSession.ts");
}
if (!adjointRow.proofArtifacts.some((artifact) =>
  artifact?.path === "packages/pt-webgpu/src/__tests__/inverseSession.test.ts"
)) {
  fail("VQ-ADJOINT-SCOPED-PATH-REPLAY must cite inverseSession.test.ts");
}
for (const needle of [
  "direct HDRI/procedural-sky environment NEE",
  "env-map intensity",
  "normalScale",
  "bumpScale",
  "clearcoatNormalScale",
  "Dormant alpha/transmission",
  "active alpha visibility",
]) {
  if (!inverseSessionSource.includes(needle)) {
    fail(`inverseSession.ts scoped adjoint contract prose is stale: missing ${needle}`);
  }
}
for (const needle of [
  "path-replay-unsupported-render-regime",
  "path-replay-unsupported-transport",
  "path-replay-unsupported-visibility",
  "path-replay-unsupported-geometry",
  "path-replay-unsupported-light-selection",
  "path-replay-unsupported-environment",
]) {
  if (!inverseSessionSource.includes(needle)) {
    fail(`inverseSession.ts scoped adjoint downgrade taxonomy is stale: missing ${needle}`);
  }
  if (!inverseSessionTestSource.includes(needle)) {
    fail(`inverseSession.test.ts scoped adjoint downgrade taxonomy is stale: missing ${needle}`);
  }
}
for (const needle of [
  "keeps transport params on finite-difference",
  "keeps active cutout alpha coverage params on finite-difference",
  "downgrades scalar displacement controls",
]) {
  if (!inverseSessionTestSource.includes(needle)) {
    fail(`inverseSession.test.ts scoped adjoint finite-difference coverage is stale: missing ${needle}`);
  }
}

const gltfBrowserRow = queue.validationQueue.find((row) => row.id === "VQ-GLTF-BROWSER-PTWEBGL2");
if (gltfBrowserRow == null) fail("validationQueue missing VQ-GLTF-BROWSER-PTWEBGL2");
if (gltfBrowserRow.status !== "host-blocked") {
  fail("VQ-GLTF-BROWSER-PTWEBGL2 must stay host-blocked until browser PNG/golden proof passes");
}
if (gltfBrowserRow.command !== "npm run gltf-browser-proof-check") {
  fail("VQ-GLTF-BROWSER-PTWEBGL2 command must be the current fail-closed browser proof check");
}
if (gltfBrowserRow.promotionCommand !== "npm run gltf-browser-proof-check:required") {
  fail("VQ-GLTF-BROWSER-PTWEBGL2 promotionCommand must keep the required browser proof gate");
}
if (!String(gltfBrowserRow.remaining).includes("engine/canvas pixels")) {
  fail("VQ-GLTF-BROWSER-PTWEBGL2 remaining text must name the engine/canvas pixel-readback blocker");
}
const gltfBrowserStatusArtifact = gltfBrowserRow.proofArtifacts.find((artifact) =>
  artifact?.path === "tools/gltf-browser-proof/pt-webgl2-real-status.json"
);
if (gltfBrowserStatusArtifact == null) {
  fail("VQ-GLTF-BROWSER-PTWEBGL2 must cite pt-webgl2-real-status.json");
}
if (gltfBrowserStatusArtifact.json?.verdict !== "HOST-BLOCKED") {
  fail("VQ-GLTF-BROWSER-PTWEBGL2 status artifact must pin HOST-BLOCKED");
}
if (gltfBrowserStatusArtifact.json?.captureMode !== "engine-first") {
  fail("VQ-GLTF-BROWSER-PTWEBGL2 status artifact must pin engine-first capture mode");
}

const radiometricPtRow = queue.validationQueue.find((row) => row.id === "VQ-RADIOMETRIC-PT");
if (radiometricPtRow == null) fail("validationQueue missing VQ-RADIOMETRIC-PT");
const glossyResearchArtifact = radiometricPtRow.proofArtifacts.find((artifact) =>
  artifact?.path === "tools/radiometric-ab/results-restir-pt-glossy-research.json"
);
if (glossyResearchArtifact == null) {
  fail("VQ-RADIOMETRIC-PT must cite results-restir-pt-glossy-research.json");
}
if (glossyResearchArtifact.json?.verdict !== "FINDING") {
  fail("VQ-RADIOMETRIC-PT glossy research artifact must pin the FINDING verdict");
}
if (glossyResearchArtifact.json?.["promotion.defaultReady"] !== false) {
  fail("VQ-RADIOMETRIC-PT glossy research artifact must pin promotion.defaultReady=false");
}
if (String(radiometricPtRow.remaining).includes("glossy ReSTIR-PT research-mode proof")) {
  fail("VQ-RADIOMETRIC-PT remaining text is stale; glossy research proof is now committed");
}
if (!radiometricPtRow.proofArtifacts.some((artifact) =>
  artifact?.path === "packages/pt-webgpu/src/index.ts"
)) {
  fail("VQ-RADIOMETRIC-PT must cite pt-webgpu index.ts for the multi-vertex BDPT structured warning");
}
for (const needle of [
  "multi-vertex BDPT branch as a structured non-promotable research finding",
  "weighted against the regular eye-path strategy",
]) {
  if (!String(radiometricPtRow.remaining).includes(needle)) {
    fail(`VQ-RADIOMETRIC-PT remaining text must include ${needle}`);
  }
}
const ptWebgpuIndexSource = await readText("packages/pt-webgpu/src/index.ts");
for (const needle of [
  "pt-webgpu.bdpt-multivertex-research-mode",
  "promotionReady: false",
  "not-weighted-against-regular-eye-path-strategy",
  "multi-vertex-light-subpath-strategies-weighted-against-regular-eye-path-strategy",
  "tools/radiometric-ab/results-bdpt.json",
]) {
  if (!ptWebgpuIndexSource.includes(needle)) {
    fail(`VQ-RADIOMETRIC-PT structured BDPT warning source is stale: missing ${needle}`);
  }
}

const walkaroundAbRow = queue.validationQueue.find((row) => row.id === "VQ-WALKAROUND-RADIOMETRIC-AB");
if (walkaroundAbRow == null) fail("validationQueue missing VQ-WALKAROUND-RADIOMETRIC-AB");
if (walkaroundAbRow.status !== "partial-proof-green") {
  fail("VQ-WALKAROUND-RADIOMETRIC-AB must remain partial-proof-green until promotion evidence exists");
}
if (walkaroundAbRow.command !== "npm run radiometric-ab:walkaround") {
  fail("VQ-WALKAROUND-RADIOMETRIC-AB command must stay on the native walkaround A/B harness");
}
const walkaroundAbArtifactPaths = new Set(walkaroundAbRow.proofArtifacts.map((artifact) => artifact?.path));
for (const path of REQUIRED_WALKAROUND_AB_ARTIFACT_PATHS) {
  if (!walkaroundAbArtifactPaths.has(path)) {
    fail(`VQ-WALKAROUND-RADIOMETRIC-AB proofArtifacts must cite ${path}`);
  }
}
for (const needle of [
  "full case set",
  "PASS-PARTIAL",
  "do-not-promote",
  "glossy is a non-promotable FINDING",
  "browser/real-adapter",
  "case-specific references",
  "GRIS/ReSTIR-GI/PPG/NRC",
]) {
  if (!String(walkaroundAbRow.remaining).includes(needle)) {
    fail(`VQ-WALKAROUND-RADIOMETRIC-AB remaining text must include ${needle}`);
  }
}
const walkaroundAbHostStatus = await readJson("tools/radiometric-ab/walkaround-ab-host-status.json");
const walkaroundAbResults = await readJson("tools/radiometric-ab/walkaround-ab-results.json");
if (walkaroundAbHostStatus.verdict !== "PASS-PARTIAL") {
  fail("VQ-WALKAROUND-RADIOMETRIC-AB host status must pin PASS-PARTIAL");
}
if (walkaroundAbHostStatus.reason?.code !== "walkaround-ab-partial-proof") {
  fail("VQ-WALKAROUND-RADIOMETRIC-AB host status must pin walkaround-ab-partial-proof");
}
if (walkaroundAbResults.a8?.verdict !== "NEGLIGIBLE") fail("walkaround A/B A8 verdict must stay NEGLIGIBLE");
if (walkaroundAbResults.sun?.verdict !== "PASS") fail("walkaround A/B SUN verdict must stay PASS");
if (walkaroundAbResults.glass?.verdict !== "PASS") fail("walkaround A/B GLASS verdict must stay PASS");
if (walkaroundAbResults.glossy?.verdict !== "FINDING") fail("walkaround A/B GLOSSY verdict must stay FINDING");
if (walkaroundAbResults.glossy?.promotion?.defaultReady !== false) {
  fail("walkaround A/B GLOSSY finding must pin promotion.defaultReady=false");
}
if (walkaroundAbResults.glossy?.promotion?.blocker !== "ddgi-irradiance-cache-not-ggx-filtered-radiance") {
  fail("walkaround A/B GLOSSY finding must pin the GGX-filtered radiance blocker");
}
const walkaroundAbHarness = await readText("tools/radiometric-ab/walkaround-ab.mjs");
const walkaroundAbProofs = await readText("tools/radiometric-ab/proofs.mjs");
const walkaroundAbChecker = await readText("tools/radiometric-ab/check-results.mjs");
const walkaroundAbReadme = await readText("tools/radiometric-ab/README.md");
const walkaroundHybridOptions = await readText("packages/walkaround-hybrid/src/HybridEngineOptions.ts");
const walkaroundGgxSource = await readText("packages/walkaround-hybrid/src/shaders/ggxBrdf.wgsl.ts");
const walkaroundShadingTerms = await readText("packages/walkaround-hybrid/src/shaders/shadingTerms.wgsl.ts");
const walkaroundShade = await readText("packages/walkaround-hybrid/src/shaders/shade.wgsl.ts");
const walkaroundRisGi = await readText("packages/walkaround-hybrid/src/shaders/risGi.wgsl.ts");
const walkaroundGlossyTest = await readText("packages/walkaround-hybrid/src/shaders/__tests__/b1GlossyMetalGi.test.ts");
for (const needle of ["VITRUM_WALKAROUND_AB_CASES", "ddgi-irradiance-cache-not-ggx-filtered-radiance"]) {
  if (!walkaroundAbHarness.includes(needle)) {
    fail(`walkaround A/B harness source is stale: missing ${needle}`);
  }
}
for (const needle of [
  "WALKAROUND_AB_CASE_IDS",
  "assertWalkaroundFullFreshStatus",
  "checkWalkaroundGlossy",
  "Do not promote",
]) {
  if (!walkaroundAbChecker.includes(needle)) {
    fail(`walkaround A/B checker source is stale: missing ${needle}`);
  }
}
for (const needle of [
  "WALKAROUND_AB_RESULT_PROOF",
  "expectedVerdict: \"FINDING\"",
  "ddgi-irradiance-cache-not-ggx-filtered-radiance",
  "material-furnace-reference-ab-and-browser-real-adapter-recapture",
]) {
  if (!walkaroundAbProofs.includes(needle)) {
    fail(`walkaround A/B proof metadata is stale: missing ${needle}`);
  }
}
for (const needle of [
  "PASS-PARTIAL",
  "GLOSSY remains",
  "VITRUM_WALKAROUND_AB_CASES",
  "ddgi-irradiance-cache-not-ggx-filtered-radiance",
]) {
  if (!walkaroundAbReadme.includes(needle)) {
    fail(`walkaround A/B README evidence note is stale: missing ${needle}`);
  }
}
for (const [label, source, needles] of [
  ["HybridEngineOptions.ts", walkaroundHybridOptions, ["restirPtReuse", "B1"]],
  ["ggxBrdf.wgsl.ts", walkaroundGgxSource, ["SPEC_GI_ROUGH_MAX"]],
  ["shadingTerms.wgsl.ts", walkaroundShadingTerms, ["SPEC_GI_ROUGH_MAX", "lo_indirectSpecular"]],
  ["shade.wgsl.ts", walkaroundShade, ["lo_indirectSpecular"]],
  ["risGi.wgsl.ts", walkaroundRisGi, ["lo_indirectSpecular"]],
  ["b1GlossyMetalGi.test.ts", walkaroundGlossyTest, ["SPEC_GI_ROUGH_MAX", "lo_indirectSpecular"]],
]) {
  for (const needle of needles) {
    if (!source.includes(needle)) {
      fail(`walkaround A/B source citation ${label} is stale: missing ${needle}`);
    }
  }
}

for (const row of queue.futureContractRows) {
  if (row == null || typeof row !== "object") fail("future-contract row must be an object");
  assertNonEmptyString(row.id, "future-contract row id");
  assertNonEmptyString(row.title, `${row.id}: title`);
  assertNonEmptyString(row.currentContract, `${row.id}: currentContract`);
  if (row.status !== "future-contract") fail(`${row.id}: status must be future-contract`);
  if (row.codeNowBounded !== false) {
    fail(`${row.id}: codeNowBounded must be false until promoted with source-verified implementation scope`);
  }
  if (!Array.isArray(row.decisionBlockers) || row.decisionBlockers.length < 2) {
    fail(`${row.id}: decisionBlockers must list at least two concrete blockers`);
  }
  for (const [idx, blocker] of row.decisionBlockers.entries()) {
    assertNonEmptyString(blocker, `${row.id}: decisionBlockers[${idx}]`);
  }
  for (const needle of REQUIRED_FUTURE_BLOCKER_NEEDLES.get(row.id) ?? []) {
    if (!row.decisionBlockers.some((blocker) => String(blocker).includes(needle))) {
      fail(`${row.id}: decisionBlockers must include ${needle}`);
    }
  }
}

const adjointFutureRow = queue.futureContractRows.find((row) => row.id === "FC-ADJOINT-FULL-PATH-PARITY");
if (adjointFutureRow == null) fail("futureContractRows missing FC-ADJOINT-FULL-PATH-PARITY");
if (!road.includes("Still OPEN for full-path parity")) {
  fail("Road adjoint full-path parity boundary disappeared without queue reconciliation");
}
for (const needle of [
  "scoped direct-light path replay",
  "finite-difference fallback diagnostics",
  "unsupported render regimes",
  "indirect paths",
]) {
  if (!String(adjointFutureRow.currentContract).includes(needle)) {
    fail(`FC-ADJOINT-FULL-PATH-PARITY currentContract must include ${needle}`);
  }
}

const walkaroundSpecialtyFutureRow = queue.futureContractRows.find((row) =>
  row.id === "FC-WALKAROUND-SPECIALTY-MATERIAL-TRANSPORT"
);
if (walkaroundSpecialtyFutureRow == null) {
  fail("futureContractRows missing FC-WALKAROUND-SPECIALTY-MATERIAL-TRANSPORT");
}
for (const needle of [
  "walkaround-hybrid",
  "spectralAttenuation",
  "dispersionAbbeNumber",
  "thinFilmStack",
  "scatteringCoefficient",
  "scatteringCoefficientRGB",
  "frontLayer",
  "backLayer",
  "displacement",
  "PT backends",
]) {
  if (!String(walkaroundSpecialtyFutureRow.currentContract).includes(needle)) {
    fail(`FC-WALKAROUND-SPECIALTY-MATERIAL-TRANSPORT currentContract must include ${needle}`);
  }
}
for (const snippet of [
  "spectralAttenuation: 'unsupported'",
  "dispersionAbbeNumber: 'unsupported'",
  "thinFilmStack: 'unsupported'",
  "scatteringCoefficient: 'approximate'",
  "scatteringCoefficientRGB: 'approximate'",
  "frontLayer: 'approximate'",
  "backLayer: 'approximate'",
  "displacementMap: 'approximate'",
  "displacementScale: 'approximate'",
  "displacementBias: 'approximate'",
]) {
  if (!promiseLedger.includes(snippet)) {
    fail(`walkaround specialty material future-contract row is stale: missing promise ledger snippet ${snippet}`);
  }
}
for (const needle of [
  "unsupported specialty fields (spectral/dispersion/thin-film/full layered",
  "approximate walkaround scattering rows",
  "walkaround spectral/dispersion",
  "thin-film/full-layer",
  "approximate scattering/front/back face absorption layers",
]) {
  if (!ledger.includes(needle) && !road.includes(needle)) {
    fail(`Road/ledger specialty material boundary disappeared without queue reconciliation: ${needle}`);
  }
}

const counts = queue.validationQueue.reduce((acc, row) => {
  acc[row.status] = (acc[row.status] ?? 0) + 1;
  return acc;
}, /** @type {Record<string, number>} */ ({}));
console.log(
  `[road-to-100-validation-status] PASS (${queue.validationQueue.length} validation rows, ` +
  `${queue.futureContractRows.length} future-contract rows, statuses=${JSON.stringify(counts)})`,
);
