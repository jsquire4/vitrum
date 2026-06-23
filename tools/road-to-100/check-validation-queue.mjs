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
  "VQ-CWBVH-DEFAULT-PROMOTION",
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

const REQUIRED_MUTATION_ARTIFACT_PATHS = [
  "tools/behavioral-gate/gate.mjs",
  "tools/behavioral-gate/check-dzn-status.mjs",
  "tools/behavioral-gate/behavioral-gate-dzn-pt-mutation-status.json",
  "tools/behavioral-gate/behavioral-gate-dzn-wh-mutation-status.json",
  "packages/pt-webgpu/src/__tests__/mutationDesyncs.test.ts",
  "packages/pt-webgpu/src/sceneMutationRouter.ts",
  "packages/pt-webgpu/src/scene/incrementalPatch.ts",
  "packages/walkaround-hybrid/src/__tests__/mutationMatrix.test.ts",
  "packages/walkaround-hybrid/src/HybridEngine.ts",
  "packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts",
  "packages/walkaround-hybrid/src/HybridEngineGiPropagation.ts",
];

const REQUIRED_ADJOINT_ARTIFACT_PATHS = [
  "packages/pt-webgpu/src/inverse/inverseSession.ts",
  "packages/pt-webgpu/src/__tests__/inverseSession.test.ts",
  "packages/pt-webgpu/src/inverse/brdfAdjoint.ts",
  "packages/pt-webgpu/src/wgsl/pathTrace/pathTraceAdjoint.wgsl.ts",
  "packages/pt-webgpu/src/wgsl/pathTrace/adjointPass.wgsl.ts",
  "packages/pt-webgpu/src/adjointPass.ts",
  "packages/pt-webgpu/src/inverse/adjointHarness.wgsl.ts",
  "packages/pt-webgpu/src/__tests__/brdfAdjoint.test.ts",
  "packages/pt-webgpu/src/__tests__/brdfAdjointEmissiveIor.test.ts",
  "packages/pt-webgpu/src/__tests__/adjointHarness.test.ts",
  "packages/pt-webgpu/src/__tests__/adjointPassPacking.test.ts",
  "packages/pt-webgpu/src/__tests__/adjointEmitterGradientOracle.test.ts",
];

const REQUIRED_PT_WEBGPU_RUNTIME_ARTIFACT_PATHS = [
  "tools/behavioral-gate/gate.mjs",
  "tools/behavioral-gate/check-dzn-status.mjs",
  "tools/behavioral-gate/behavioral-gate-dzn-default-status.json",
  "tools/behavioral-gate/behavioral-gate-dzn-material-lobes-status.json",
  "tools/behavioral-gate/behavioral-gate-dzn-material-lobe-maps-status.json",
  "tools/reference-renders/pt-material-lobes-behavioral/pt-material-lobes.dzn-full.png",
  "tools/reference-renders/pt-material-lobes-behavioral/pt-material-lobe-maps.dzn-full.png",
  "packages/pt-webgpu/src/__tests__/ggxAnisotropicBrdf.test.ts",
  "packages/pt-webgpu/src/__tests__/ggxMultiscatterFurnace.test.ts",
  "packages/pt-webgpu/src/__tests__/restirPtSpecialtyReference.test.ts",
];

const REQUIRED_GLTF_REAL_ARTIFACT_PATHS = [
  "tools/gltf-real-asset-sweep/check-proofs.mjs",
  "tools/gltf-real-asset-sweep/proofs.mjs",
  "tools/gltf-real-asset-sweep/assetManifest.mjs",
  "tools/gltf-real-asset-sweep/sweep.mjs",
  "tools/reference-renders/gltf-real-behavioral/manifest.json",
  "tools/reference-renders/gltf-real-behavioral-dzn-full/manifest.json",
  "tools/reference-renders/gltf-real-behavioral-dzn-full/pt-gltf-real-box-textured.png",
  "tools/reference-renders/gltf-real-behavioral-dzn-full/pt-gltf-real-draco.png",
  "tools/reference-renders/gltf-real-behavioral-dzn-full/pt-gltf-real-meshopt.png",
];

const REQUIRED_GLTF_MATERIAL_TOPOLOGY_ARTIFACT_PATHS = [
  "tools/gltf-material-sweep/check-proofs.mjs",
  "tools/gltf-material-sweep/proofs.mjs",
  "tools/gltf-material-sweep/fixture.mjs",
  "tools/gltf-material-sweep/sweep.mjs",
  "tools/reference-renders/gltf-material-sweep-behavioral/manifest.json",
  "tools/behavioral-gate/behavioral-gate-dzn-gltf-material-sweep-status.json",
  "tools/reference-renders/gltf-material-sweep-behavioral/pt-gltf-material-sweep.png",
  "tools/gltf-topology-proofs/check-proofs.mjs",
  "tools/gltf-topology-proofs/proofs.mjs",
  "tools/reference-renders/gltf-point-line-behavioral/manifest.json",
  "tools/reference-renders/gltf-point-line-behavioral/pt-gltf-point-line-fallback.png",
  "tools/reference-renders/gltf-triangle-topology-behavioral/manifest.json",
  "tools/reference-renders/gltf-triangle-topology-behavioral/pt-gltf-triangle-strip-fan.png",
  "packages/gltf-adapter/src/primitiveModeFallback.ts",
  "packages/gltf-adapter/src/assetLoader.ts",
  "packages/gltf-adapter/src/featureReport.ts",
  "packages/gltf-adapter/src/gltfPointLinePrimitivePolicy.test.ts",
  "packages/gltf-adapter/src/gltfKhronosSweep.test.ts",
  "packages/gltf-adapter/src/gltfAssetApi.test.ts",
];

const REQUIRED_WALKAROUND_BEHAVIORAL_ROWS = [
  {
    statusPath: "tools/behavioral-gate/behavioral-gate-dzn-wh-default-status.json",
    label: "wh/default",
    goldenPath: "tools/reference-renders/wh-behavioral/wh-default.dzn-full.png",
  },
  {
    statusPath: "tools/behavioral-gate/behavioral-gate-dzn-wh-rcenabled-status.json",
    label: "wh/rcEnabled",
    goldenPath: "tools/reference-renders/wh-behavioral/wh-rcenabled.dzn-full.png",
  },
  {
    statusPath: "tools/behavioral-gate/behavioral-gate-dzn-wh-ppgenabled-status.json",
    label: "wh/ppgEnabled",
    goldenPath: "tools/reference-renders/wh-behavioral/wh-ppgenabled.dzn-full.png",
  },
  {
    statusPath: "tools/behavioral-gate/behavioral-gate-dzn-wh-gtao-off-status.json",
    label: "wh/gtao-off",
    goldenPath: "tools/reference-renders/wh-behavioral/wh-gtao-off.dzn-full.png",
  },
  {
    statusPath: "tools/behavioral-gate/behavioral-gate-dzn-wh-checkerboard-status.json",
    label: "wh/checkerboard",
    goldenPath: "tools/reference-renders/wh-behavioral/wh-checkerboard.dzn-full.png",
  },
  {
    statusPath: "tools/behavioral-gate/behavioral-gate-dzn-wh-skinned-mesh-status.json",
    label: "wh/skinned-mesh",
    goldenPath: "tools/reference-renders/wh-behavioral/wh-skinned-mesh.dzn-full.png",
  },
  {
    statusPath: "tools/behavioral-gate/behavioral-gate-dzn-wh-hdri-env-status.json",
    label: "wh/hdri-env",
    goldenPath: "tools/reference-renders/wh-behavioral/wh-hdri-env.dzn-full.png",
  },
  {
    statusPath: "tools/behavioral-gate/behavioral-gate-dzn-wh-rect-area-emitter-status.json",
    label: "wh/rect-area-emitter",
    goldenPath: "tools/reference-renders/wh-behavioral/wh-rect-area-emitter.dzn-full.png",
  },
  {
    statusPath: "tools/behavioral-gate/behavioral-gate-dzn-wh-directional-sun-status.json",
    label: "wh/directional-sun",
    goldenPath: "tools/reference-renders/wh-behavioral/wh-directional-sun.dzn-full.png",
  },
  {
    statusPath: "tools/behavioral-gate/behavioral-gate-dzn-wh-glass-gi-status.json",
    label: "wh/glass-gi",
    goldenPath: "tools/reference-renders/wh-behavioral/wh-glass-gi.dzn-full.png",
  },
  {
    statusPath: "tools/behavioral-gate/behavioral-gate-dzn-wh-transparent-oit-status.json",
    label: "wh/transparent-oit",
    goldenPath: "tools/reference-renders/wh-transparent-oit-behavioral/wh-transparent-oit.dzn-full.png",
  },
];

const REQUIRED_WALKAROUND_BEHAVIORAL_ARTIFACT_PATHS = [
  "tools/behavioral-gate/gate.mjs",
  "tools/behavioral-gate/check-dzn-status.mjs",
  "packages/walkaround-hybrid/src/__tests__/transparentAlphaTransportContract.test.ts",
  "packages/walkaround-hybrid/src/shaders/transparentOit.wgsl.ts",
  "packages/walkaround-hybrid/src/shaders/shadingTerms.wgsl.ts",
  "packages/walkaround-hybrid/src/shaders/restirCastPrimary.wgsl.ts",
  "packages/walkaround-hybrid/src/shaders/ris.wgsl.ts",
  "packages/walkaround-hybrid/src/shaders/risGi.wgsl.ts",
  "packages/walkaround-hybrid/src/shaders/shade.wgsl.ts",
  ...REQUIRED_WALKAROUND_BEHAVIORAL_ROWS.flatMap((row) => [row.statusPath, row.goldenPath]),
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
  "tools/radiometric-ab/run-walkaround-ab.mjs",
  "tools/radiometric-ab/walkaround-ab.mjs",
  "tools/radiometric-ab/walkaround-ab-host-status.json",
  "tools/radiometric-ab/walkaround-ab-results.json",
  "tools/radiometric-ab/walkaround-ab-glossy-spp64-status.json",
  "tools/radiometric-ab/walkaround-ab-glossy-spp64.json",
  "tools/radiometric-ab/walkaround-ab-all-spp64-status.json",
  "tools/radiometric-ab/walkaround-ab-all-spp64.json",
  "tools/radiometric-ab/walkaround-ab-promotion-status.json",
  "packages/walkaround-hybrid/src/HybridEngineOptions.ts",
  "packages/walkaround-hybrid/src/shaders/ggxBrdf.wgsl.ts",
  "packages/walkaround-hybrid/src/shaders/shadingTerms.wgsl.ts",
  "packages/walkaround-hybrid/src/shaders/shade.wgsl.ts",
  "packages/walkaround-hybrid/src/shaders/risGi.wgsl.ts",
  "packages/walkaround-hybrid/src/shaders/__tests__/b1GlossyMetalGi.test.ts",
];

const REQUIRED_RADIOMETRIC_PT_ARTIFACT_PATHS = [
  "tools/radiometric-ab/pt-ab-host-status.json",
  "tools/radiometric-ab/ab-sppm.mjs",
  "tools/radiometric-ab/ab-bdpt.mjs",
  "tools/radiometric-ab/ab-restir-pt.mjs",
  "tools/radiometric-ab/ab-restir-pt-glossy-research.mjs",
  "tools/radiometric-ab/ab-restir-pt-specialty.mjs",
  "tools/radiometric-ab/ab-sobol.mjs",
  "tools/radiometric-ab/check-results.mjs",
  "tools/radiometric-ab/proofs.mjs",
  "tools/radiometric-ab/results-bdpt.json",
  "tools/radiometric-ab/results-restir-pt.json",
  "tools/radiometric-ab/results-restir-pt-glossy-research.json",
  "tools/radiometric-ab/results-restir-pt-specialty.json",
  "tools/radiometric-ab/results-sppm.json",
  "tools/radiometric-ab/results-sobol.json",
  "packages/pt-webgpu/src/index.ts",
  "packages/pt-webgpu/src/wgsl/pathTrace/kernel.wgsl.ts",
  "packages/pt-webgpu/src/wgsl/bdpt/bdptConnection.wgsl.ts",
  "packages/pt-webgpu/src/wgsl/bdpt/bdptLightSubpath.wgsl.ts",
  "packages/pt-webgpu/src/__tests__/bdptConnectionMisFull.test.ts",
  "packages/pt-webgpu/src/__tests__/bdptGlossyLightSubpath.test.ts",
  "packages/pt-webgpu/src/__tests__/oracle.sppmPhotonFlux.test.ts",
  "packages/pt-webgpu/src/__tests__/restirPtSpecialtyReference.test.ts",
  "packages/pt-webgpu/src/__tests__/ggxAnisotropicBrdf.test.ts",
  "packages/pt-webgpu/src/__tests__/ggxMultiscatterFurnace.test.ts",
  "packages/shared-samplers/__tests__/bdptVeachFull.test.ts",
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
 * @param {unknown} row
 * @param {readonly string[]} paths
 * @param {string} label
 */
function assertRowCitesPaths(row, paths, label) {
  if (row == null || typeof row !== "object" || !Array.isArray(row.proofArtifacts)) {
    fail(`${label}: proofArtifacts must be an array`);
  }
  const cited = new Set(row.proofArtifacts.map((artifact) => artifact?.path));
  for (const path of paths) {
    if (!cited.has(path)) fail(`${label} proofArtifacts must cite ${path}`);
  }
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
const brdfAdjointSource = await readText("packages/pt-webgpu/src/inverse/brdfAdjoint.ts");
const pathTraceAdjointWgslSource = await readText("packages/pt-webgpu/src/wgsl/pathTrace/pathTraceAdjoint.wgsl.ts");
const adjointPassWgslSource = await readText("packages/pt-webgpu/src/wgsl/pathTrace/adjointPass.wgsl.ts");
const adjointPassSource = await readText("packages/pt-webgpu/src/adjointPass.ts");
const adjointHarnessSource = await readText("packages/pt-webgpu/src/inverse/adjointHarness.wgsl.ts");
const brdfAdjointTestSource = await readText("packages/pt-webgpu/src/__tests__/brdfAdjoint.test.ts");
const brdfAdjointEmissiveIorTestSource = await readText(
  "packages/pt-webgpu/src/__tests__/brdfAdjointEmissiveIor.test.ts",
);
const adjointHarnessTestSource = await readText("packages/pt-webgpu/src/__tests__/adjointHarness.test.ts");
const adjointPassPackingTestSource = await readText("packages/pt-webgpu/src/__tests__/adjointPassPacking.test.ts");
const adjointEmitterGradientOracleTestSource = await readText(
  "packages/pt-webgpu/src/__tests__/adjointEmitterGradientOracle.test.ts",
);
const learnedCheckpointManifest = await readJson(LEARNED_CHECKPOINT_MANIFEST_PATH);
const behavioralGateSource = await readText("tools/behavioral-gate/gate.mjs");
const transparentAlphaTransportContractTest = await readText(
  "packages/walkaround-hybrid/src/__tests__/transparentAlphaTransportContract.test.ts",
);
const transparentOitSource = await readText("packages/walkaround-hybrid/src/shaders/transparentOit.wgsl.ts");
const gltfAssetLoaderSource = await readText("packages/gltf-adapter/src/assetLoader.ts");
const gltfAssetApiTestSource = await readText("packages/gltf-adapter/src/gltfAssetApi.test.ts");
const gltfMaterialProofSource = await readText("tools/gltf-material-sweep/check-proofs.mjs");
const gltfMaterialFixtureSource = await readText("tools/gltf-material-sweep/fixture.mjs");
const gltfTopologyProofSource = await readText("tools/gltf-topology-proofs/check-proofs.mjs");
const gltfRealProofSource = await readText("tools/gltf-real-asset-sweep/check-proofs.mjs");
const gltfRealAssetManifestSource = await readText("tools/gltf-real-asset-sweep/assetManifest.mjs");
const radiometricProofSource = await readText("tools/radiometric-ab/proofs.mjs");
const radiometricCheckerSource = await readText("tools/radiometric-ab/check-results.mjs");
const ptMutationTestSource = await readText("packages/pt-webgpu/src/__tests__/mutationDesyncs.test.ts");
const ptMutationRouterSource = await readText("packages/pt-webgpu/src/sceneMutationRouter.ts");
const ptMutationPatchSource = await readText("packages/pt-webgpu/src/scene/incrementalPatch.ts");
const walkaroundMutationTestSource = await readText(
  "packages/walkaround-hybrid/src/__tests__/mutationMatrix.test.ts",
);
const walkaroundEngineSource = await readText("packages/walkaround-hybrid/src/HybridEngine.ts");
const walkaroundPrimitiveUpdatesSource = await readText(
  "packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts",
);
const walkaroundGiPropagationSource = await readText(
  "packages/walkaround-hybrid/src/HybridEngineGiPropagation.ts",
);

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
  for (const [key, value] of Object.entries(row)) {
    if (key === "command" || !key.endsWith("Command") || value == null) continue;
    assertNonEmptyString(value, `${row.id}: ${key}`);
    assertCommandScriptsExist(value, scripts);
  }
  if (!Array.isArray(row.proofArtifacts)) fail(`${row.id}: proofArtifacts must be an array`);
  for (const artifact of row.proofArtifacts) await assertArtifact(artifact, row.id);
}

const mutationRow = queue.validationQueue.find((row) => row.id === "VQ-MUTATION-MATRIX");
if (mutationRow == null) fail("validationQueue missing VQ-MUTATION-MATRIX");
if (mutationRow.status !== "committed-proof-green") {
  fail("VQ-MUTATION-MATRIX must stay committed-proof-green for the explicitly proven pt/walkaround dzn mutation shard scope");
}
assertRowCitesPaths(mutationRow, REQUIRED_MUTATION_ARTIFACT_PATHS, "VQ-MUTATION-MATRIX");
if (!String(mutationRow.remaining).includes("observable before/after pixel deltas")) {
  fail("VQ-MUTATION-MATRIX remaining text must keep the observable mutation delta proof explicit");
}
if (!String(mutationRow.remaining).includes("committed dzn-full post-mutation goldens")) {
  fail("VQ-MUTATION-MATRIX remaining text must keep committed dzn-full goldens explicit");
}
if (!String(mutationRow.remaining).includes("new backend-specific mutation gaps must enter the implementation queue")) {
  fail("VQ-MUTATION-MATRIX remaining text must route new mutation expansion through source-verified implementation rows");
}
const mutationArtifactPaths = new Set(mutationRow.proofArtifacts.map((artifact) => artifact?.path));
for (const mutationKind of REQUIRED_MUTATION_KINDS) {
  for (const prefix of ["pt", "wh"]) {
    const path = `tools/reference-renders/mutation-behavioral/${prefix}-mutation-${mutationKind}.dzn-full.png`;
    if (!mutationArtifactPaths.has(path)) fail(`VQ-MUTATION-MATRIX must cite ${path}`);
  }
  for (const prefix of ["pt", "wh"]) {
    const label = `${prefix}/mutation-${mutationKind}`;
    const goldenKey = `${prefix}-mutation-${mutationKind}`;
    if (!behavioralGateSource.includes(label)) {
      fail(`VQ-MUTATION-MATRIX behavioral gate source is stale: missing ${label}`);
    }
    if (!behavioralGateSource.includes(`mutationGolden("${goldenKey}")`)) {
      fail(`VQ-MUTATION-MATRIX behavioral gate source is stale: missing mutation golden ${goldenKey}`);
    }
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
for (const needle of [
  "canFastPathMaterialPatch",
  "materialPatchRepackFields",
  "updateEmitter writes the emitter buffer",
  "updateEnvironment writes same-sized HDRI buffers",
  "vertex/index-count topology patches invalidate cached bind groups before committing",
  "instanced-mesh count changes invalidate cached bind groups before committing",
  "pt-webgpu.primitive-material-repack",
]) {
  if (!ptMutationTestSource.includes(needle)) {
    fail(`VQ-MUTATION-MATRIX pt-webgpu mutation test proof is stale: missing ${needle}`);
  }
}
for (const needle of [
  "addPrimitive(primitive",
  "removePrimitive(id",
  "updatePrimitive(id",
  "updateEmitter(id",
  "updateEnvironment(env",
  "host.invalidateBindGroups",
  "host.syncLiteTextures",
  "host.reset",
]) {
  if (!ptMutationRouterSource.includes(needle)) {
    fail(`VQ-MUTATION-MATRIX pt-webgpu mutation router source is stale: missing ${needle}`);
  }
}
for (const needle of [
  "TEXTURE_MAP_FIELDS",
  "MATERIAL_TEXTURE_DESCRIPTOR_SCALAR_FIELDS",
  "GEOMETRY_MATERIAL_FIELDS",
  "canFastPathGeometryPatch",
  "canFastPathInstancedTopologyPatch",
  "canFastPathTopologyResizePatch",
  "canFastPathTransformPatch",
]) {
  if (!ptMutationPatchSource.includes(needle)) {
    fail(`VQ-MUTATION-MATRIX pt-webgpu incremental patch source is stale: missing ${needle}`);
  }
}
for (const needle of [
  "updatePrimitive(transform) refits TLAS",
  "updatePrimitive(material) refreshes DDGI material snapshots",
  "updateEmitter repacks emitters",
  "updateEnvironment bakes procedural-sky",
  "updateLighting republishes DDGI sun lights",
]) {
  if (!walkaroundMutationTestSource.includes(needle)) {
    fail(`VQ-MUTATION-MATRIX walkaround mutation test proof is stale: missing ${needle}`);
  }
}
for (const needle of [
  "addPrimitive(primitive",
  "removePrimitive(id",
  "updateEmitter(id",
  "updateLighting(opts",
  "updateEnvironment(env",
]) {
  if (!walkaroundEngineSource.includes(needle)) {
    fail(`VQ-MUTATION-MATRIX walkaround engine source is stale: missing ${needle}`);
  }
}
for (const needle of [
  "refreshTlasRefit",
  "refreshBvhFullRebuild",
  "refreshBvhMaterialSlice",
  "requestAccumReset",
  "markInstancesDirty",
  "invalidateProbeCache",
  "refreshDdgiMaterialSnapshot",
]) {
  if (!walkaroundPrimitiveUpdatesSource.includes(needle)) {
    fail(`VQ-MUTATION-MATRIX walkaround primitive update source is stale: missing ${needle}`);
  }
}
for (const needle of [
  "syncRestirBvhBuffers",
  "propagateBvhToGiSubsystems",
  "rebuildRcMergedSceneCoreFirst",
  "refitMergedInstance",
]) {
  if (!walkaroundGiPropagationSource.includes(needle)) {
    fail(`VQ-MUTATION-MATRIX walkaround GI propagation source is stale: missing ${needle}`);
  }
}

const ptRuntimeRow = queue.validationQueue.find((row) => row.id === "VQ-PT-WEBGPU-RUNTIME-GOLDENS");
if (ptRuntimeRow == null) fail("validationQueue missing VQ-PT-WEBGPU-RUNTIME-GOLDENS");
assertRowCitesPaths(ptRuntimeRow, REQUIRED_PT_WEBGPU_RUNTIME_ARTIFACT_PATHS, "VQ-PT-WEBGPU-RUNTIME-GOLDENS");
for (const needle of [
  "pt/material-lobes",
  "pt/material-lobe-maps",
  "MATERIAL_LOBE_GOLDEN",
  "MATERIAL_LOBE_MAP_GOLDEN",
]) {
  if (!behavioralGateSource.includes(needle)) {
    fail(`VQ-PT-WEBGPU-RUNTIME-GOLDENS behavioral gate source is stale: missing ${needle}`);
  }
}
for (const [path, label] of [
  ["tools/behavioral-gate/behavioral-gate-dzn-material-lobes-status.json", "pt/material-lobes"],
  ["tools/behavioral-gate/behavioral-gate-dzn-material-lobe-maps-status.json", "pt/material-lobe-maps"],
]) {
  const status = await readJson(path);
  if (status.verdict !== "PASS" || status.exitStatus !== 0) fail(`${path} must pin PASS/exitStatus=0`);
  if (status.goldenVariant !== "dzn-full") fail(`${path} must pin dzn-full`);
  const config = status.configs?.[0];
  if (config?.label !== label || config?.goldenStatus !== "ok" || config?.tier !== "full") {
    fail(`${path} must pin full-tier golden-ok config ${label}`);
  }
}

const walkaroundBehavioralRow = queue.validationQueue.find((row) => row.id === "VQ-WALKAROUND-BEHAVIORAL-MATRIX");
if (walkaroundBehavioralRow == null) fail("validationQueue missing VQ-WALKAROUND-BEHAVIORAL-MATRIX");
assertRowCitesPaths(
  walkaroundBehavioralRow,
  REQUIRED_WALKAROUND_BEHAVIORAL_ARTIFACT_PATHS,
  "VQ-WALKAROUND-BEHAVIORAL-MATRIX",
);
for (const row of REQUIRED_WALKAROUND_BEHAVIORAL_ROWS) {
  if (!behavioralGateSource.includes(`"${row.label}"`)) {
    fail(`VQ-WALKAROUND-BEHAVIORAL-MATRIX behavioral gate source is stale: missing ${row.label}`);
  }
  const status = await readJson(row.statusPath);
  if (status.verdict !== "PASS" || status.exitStatus !== 0) {
    fail(`${row.statusPath} must pin PASS/exitStatus=0`);
  }
  if (status.goldenVariant !== "dzn-full") fail(`${row.statusPath} must pin dzn-full`);
  if (status.summary?.totalConfigs !== 1) fail(`${row.statusPath} must contain exactly one focused config`);
  if (status.summary?.failures !== 0) fail(`${row.statusPath} must pin zero failures`);
  if (status.summary?.knownResiduals !== 0) fail(`${row.statusPath} must pin zero known residuals`);
  const config = status.configs?.[0];
  if (config?.label !== row.label) fail(`${row.statusPath} must pin config label ${row.label}`);
  if (config?.verdict !== "PASS" || config?.rawStatus !== "OK") {
    fail(`${row.statusPath} ${row.label} must pin PASS/OK`);
  }
  if (config?.gpuErrors !== 0 || config?.nan !== false) {
    fail(`${row.statusPath} ${row.label} must pin gpuErrors=0 and nan=false`);
  }
  if (config?.goldenStatus !== "ok" || config?.goldenVariant !== "dzn-full") {
    fail(`${row.statusPath} ${row.label} must pin goldenStatus=ok and dzn-full`);
  }
}
for (const needle of [
  "traceSceneFirstHitAlphaMaskTexturedOpaqueOnly(",
  "traceSceneAlphaTintTransmittanceTextured(",
  "fn oitLayerAreaEmitterNEE(",
  "sampleEmitterLeAtXi(e, xi)",
]) {
  if (!transparentAlphaTransportContractTest.includes(needle) && !transparentOitSource.includes(needle)) {
    fail(`VQ-WALKAROUND-BEHAVIORAL-MATRIX transparent-OIT proof is stale: missing ${needle}`);
  }
}
for (const needle of [
  "not.toMatch(/var<storage,\\s*(read|read_write)>[^;]*reservoir/i)",
  "not.toMatch(/\\b(load|store|update|resolve)\\w*Reservoir\\b/)",
  "not.toContain('selectedEmitter')",
  "not.toContain('risFinal')",
]) {
  if (!transparentAlphaTransportContractTest.includes(needle)) {
    fail(`VQ-WALKAROUND-BEHAVIORAL-MATRIX transparent-OIT reservoir exclusion test is stale: missing ${needle}`);
  }
}

const gltfRealRow = queue.validationQueue.find((row) => row.id === "VQ-GLTF-REAL-WEBGPU");
if (gltfRealRow == null) fail("validationQueue missing VQ-GLTF-REAL-WEBGPU");
assertRowCitesPaths(gltfRealRow, REQUIRED_GLTF_REAL_ARTIFACT_PATHS, "VQ-GLTF-REAL-WEBGPU");
for (const needle of [
  "REQUIRED_REAL_GLTF_PROOF_ROWS",
  "box-textured-glb",
  "cesium-milk-truck-draco",
  "meshopt-cube-real",
  "KHR_draco_mesh_compression",
  "KHR_meshopt_compression",
]) {
  if (!gltfRealProofSource.includes(needle) && !gltfRealAssetManifestSource.includes(needle)) {
    fail(`VQ-GLTF-REAL-WEBGPU proof source is stale: missing ${needle}`);
  }
}
for (const needle of [
  "pt/gltf-real-box-textured",
  "pt/gltf-real-draco",
  "pt/gltf-real-meshopt",
]) {
  if (!behavioralGateSource.includes(needle)) {
    fail(`VQ-GLTF-REAL-WEBGPU behavioral gate source is stale: missing ${needle}`);
  }
}
const realGltfDznManifest = await readJson("tools/reference-renders/gltf-real-behavioral-dzn-full/manifest.json");
if (realGltfDznManifest.kind !== "vitrum-real-gltf-behavioral-goldens") {
  fail("VQ-GLTF-REAL-WEBGPU dzn manifest kind mismatch");
}
if (realGltfDznManifest.goldenVariant !== "dzn-full") {
  fail("VQ-GLTF-REAL-WEBGPU dzn manifest must pin goldenVariant=dzn-full");
}
if (!Array.isArray(realGltfDznManifest.assets) || realGltfDznManifest.assets.length !== 3) {
  fail("VQ-GLTF-REAL-WEBGPU dzn manifest must pin the three public real-asset rows");
}

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

const cwbvhRow = queue.validationQueue.find((row) => row.id === "VQ-CWBVH-DEFAULT-PROMOTION");
if (cwbvhRow == null) fail("validationQueue missing VQ-CWBVH-DEFAULT-PROMOTION");
if (cwbvhRow.status !== "partial-proof-green") {
  fail("VQ-CWBVH-DEFAULT-PROMOTION must stay partial-proof-green until default-promotion throughput evidence lands");
}
if (cwbvhRow.command !== "npm run cwbvh-gpu-proof-check") {
  fail("VQ-CWBVH-DEFAULT-PROMOTION command must stay on the CWBVH proof checker");
}
assertRowCitesPaths(cwbvhRow, [
  "tools/behavioral-gate/check-cwbvh-parity-status.mjs",
  "tools/behavioral-gate/check-cwbvh-renderer-parity-status.mjs",
  "tools/behavioral-gate/cwbvh-parity-oracle.mjs",
  "tools/behavioral-gate/cwbvh-parity-status.json",
  "tools/behavioral-gate/behavioral-gate-cwbvh-status.json",
  "tools/behavioral-gate/behavioral-gate-dzn-cwbvh-binary-parity-status.json",
  "tools/behavioral-gate/behavioral-gate-dzn-cwbvh-complex-parity-status.json",
  "tools/behavioral-gate/behavioral-gate-dzn-cwbvh-broader-status.json",
  "tools/behavioral-gate/cwbvh-default-promotion-status.json",
  "tools/behavioral-gate/gate.mjs",
  "packages/pt-webgpu/src/index.ts",
  "packages/pt-webgpu/src/scene/uploadSceneBuffers.ts",
  "packages/pt-webgpu/src/__tests__/cwbvhSceneBuffers.test.ts",
  "packages/pt-webgpu/src/__tests__/cwbvhTraversalWiring.test.ts",
  "packages/shared-bvh/src/compressedWideBvh.ts",
  "packages/shared-bvh/src/wgsl/cwbvhIntersect.wgsl.ts",
  "packages/shared-bvh/src/__tests__/compressedWideBvh.test.ts",
  "packages/shared-bvh/src/__tests__/cwbvhWgsl.test.ts",
], "VQ-CWBVH-DEFAULT-PROMOTION");
for (const needle of [
  "opt-in CWBVH",
  "renderer binary-vs-CWBVH pixel parity",
  "broader dzn material/glTF workload shard",
  "Default promotion is still blocked",
  "dzn timing artifacts are uniformly slower",
  "material-lobe-map",
  "browser/real-adapter throughput A/B",
]) {
  if (!String(cwbvhRow.remaining).includes(needle)) {
    fail(`VQ-CWBVH-DEFAULT-PROMOTION remaining text must include ${needle}`);
  }
}
const cwbvhOracleStatus = await readJson("tools/behavioral-gate/cwbvh-parity-status.json");
if (cwbvhOracleStatus.verdict !== "PASS" || cwbvhOracleStatus.rootCount !== 2) {
  fail("VQ-CWBVH-DEFAULT-PROMOTION CWBVH oracle status must pin PASS multi-root traversal");
}
if (
  cwbvhOracleStatus.checks?.nonzeroRootClosest !== true ||
  cwbvhOracleStatus.checks?.nonzeroRootAny !== true
) {
  fail("VQ-CWBVH-DEFAULT-PROMOTION oracle status must prove nonzero-root closest/any traversal");
}
const cwbvhRendererStatus = await readJson("tools/behavioral-gate/behavioral-gate-cwbvh-status.json");
if (cwbvhRendererStatus.verdict !== "PASS" || cwbvhRendererStatus.summary?.failures !== 0) {
  fail("VQ-CWBVH-DEFAULT-PROMOTION renderer parity status must pin PASS and zero failures");
}
if (!Array.isArray(cwbvhRendererStatus.configs) || cwbvhRendererStatus.configs.length !== 2) {
  fail("VQ-CWBVH-DEFAULT-PROMOTION renderer parity status must contain simple and complex lanes");
}
for (const label of ["pt/cwbvh-binary-parity", "pt/cwbvh-complex-parity"]) {
  const config = cwbvhRendererStatus.configs.find((entry) => entry?.label === label);
  if (config == null) fail(`VQ-CWBVH-DEFAULT-PROMOTION renderer status missing ${label}`);
  if (config.verdict !== "PASS" || config.cwbvhParityKind !== "binary") {
    fail(`VQ-CWBVH-DEFAULT-PROMOTION ${label} must pin binary parity PASS`);
  }
  if (config.cwbvhParityRmse !== 0 || config.cwbvhParityMeanAbs !== 0 || config.cwbvhParityMaxAbs !== 0) {
    fail(`VQ-CWBVH-DEFAULT-PROMOTION ${label} must pin exact parity in the committed local artifact`);
  }
}
const dznCwbvhStatuses = [
  await readJson("tools/behavioral-gate/behavioral-gate-dzn-cwbvh-binary-parity-status.json"),
  await readJson("tools/behavioral-gate/behavioral-gate-dzn-cwbvh-complex-parity-status.json"),
];
for (const status of dznCwbvhStatuses) {
  if (status.verdict !== "PASS" || status.goldenVariant !== "dzn-full" || status.summary?.failures !== 0) {
    fail(`VQ-CWBVH-DEFAULT-PROMOTION ${status.filter} dzn status must pin PASS/dzn-full/zero failures`);
  }
  const config = Array.isArray(status.configs) ? status.configs[0] : null;
  if (config == null || config.cwbvhParityKind !== "binary" || config.cwbvhParityRmse !== 0) {
    fail(`VQ-CWBVH-DEFAULT-PROMOTION ${status.filter} dzn status must pin exact binary-vs-CWBVH parity`);
  }
  if (!(Number(config.cwbvhRenderMsRatio) > 1)) {
    fail(`VQ-CWBVH-DEFAULT-PROMOTION ${status.filter} dzn status must preserve the no-default-promotion slowdown finding`);
  }
}
const dznCwbvhBroaderStatus = await readJson("tools/behavioral-gate/behavioral-gate-dzn-cwbvh-broader-status.json");
if (
  dznCwbvhBroaderStatus.verdict !== "PASS" ||
  dznCwbvhBroaderStatus.filter !== "cwbvh-broader" ||
  dznCwbvhBroaderStatus.goldenVariant !== "dzn-full" ||
  dznCwbvhBroaderStatus.summary?.failures !== 0
) {
  fail("VQ-CWBVH-DEFAULT-PROMOTION broader dzn status must pin PASS/cwbvh-broader/dzn-full/zero failures");
}
const broaderLabels = [
  "pt/cwbvh-broader-material-lobes",
  "pt/cwbvh-broader-material-lobe-maps",
  "pt/cwbvh-broader-gltf-material-sweep",
];
if (!Array.isArray(dznCwbvhBroaderStatus.configs) || dznCwbvhBroaderStatus.configs.length !== broaderLabels.length) {
  fail("VQ-CWBVH-DEFAULT-PROMOTION broader dzn status must contain the material/glTF workload set");
}
for (const label of broaderLabels) {
  const config = dznCwbvhBroaderStatus.configs.find((entry) => entry?.label === label);
  if (config == null) fail(`VQ-CWBVH-DEFAULT-PROMOTION broader dzn status missing ${label}`);
  if (config.verdict !== "PASS" || config.rawStatus !== "OK" || config.tier !== "full") {
    fail(`VQ-CWBVH-DEFAULT-PROMOTION ${label} must pin a full-tier PASS`);
  }
  if (config.cwbvhParityKind !== "binary" || config.cwbvhParityRmse > 1 || config.cwbvhParityMeanAbs > 0.5 || config.cwbvhParityMaxAbs > 8) {
    fail(`VQ-CWBVH-DEFAULT-PROMOTION ${label} must preserve binary-vs-CWBVH parity bounds`);
  }
  if (config.cwbvhPerfKind !== "same-scene" || !(Number(config.cwbvhBinaryRenderMs) > 0) || !(Number(config.cwbvhRenderMs) > 0)) {
    fail(`VQ-CWBVH-DEFAULT-PROMOTION ${label} must preserve same-scene CWBVH timing evidence`);
  }
}
const cwbvhPromotionStatus = await readJson("tools/behavioral-gate/cwbvh-default-promotion-status.json");
const cwbvhTimingRows = [
  ...dznCwbvhStatuses.map((status) => status.configs[0]),
  ...broaderLabels.map((label) => dznCwbvhBroaderStatus.configs.find((entry) => entry?.label === label)),
];
const cwbvhExpectedRatios = cwbvhTimingRows.map((entry) => ({
  label: entry.label,
  ratio: Number(entry.cwbvhRenderMsRatio),
}));
const cwbvhSlowOrNeutralCount = cwbvhExpectedRatios.filter((entry) => entry.ratio >= 1).length;
const cwbvhFastCount = cwbvhExpectedRatios.filter((entry) => entry.ratio < 0.95).length;
if (
  cwbvhPromotionStatus.verdict !== "PASS-PARTIAL" ||
  cwbvhPromotionStatus.promotion?.defaultReady !== false ||
  cwbvhPromotionStatus.promotion?.classification !== "uniform-slower"
) {
  fail("VQ-CWBVH-DEFAULT-PROMOTION summary must pin PASS-PARTIAL/defaultReady=false/uniform-slower");
}
if (
  cwbvhPromotionStatus.rowCount !== cwbvhExpectedRatios.length ||
  cwbvhPromotionStatus.slowOrNeutralCount !== cwbvhSlowOrNeutralCount ||
  cwbvhPromotionStatus.fastCount !== cwbvhFastCount
) {
  fail("VQ-CWBVH-DEFAULT-PROMOTION summary counts must match committed timing rows");
}
if (JSON.stringify(cwbvhPromotionStatus.ratios) !== JSON.stringify(cwbvhExpectedRatios)) {
  fail("VQ-CWBVH-DEFAULT-PROMOTION summary ratios must match committed timing rows");
}
if (!String(cwbvhPromotionStatus.promotion?.requiredEvidence ?? "").includes("browser/real-adapter throughput A/B")) {
  fail("VQ-CWBVH-DEFAULT-PROMOTION summary must name browser/real-adapter throughput A/B");
}

const learnedRow = queue.validationQueue.find((row) => row.id === "VQ-LEARNED-SYSTEMS");
if (learnedRow == null) fail("validationQueue missing VQ-LEARNED-SYSTEMS");
for (const path of [
  "tools/learned-systems/check-status.mjs",
  "tools/learned-systems/qualityManifestValidator.mjs",
  "tools/learned-systems/learned-systems-status.json",
  "tools/neural-denoiser-training/checkpoints/manifest.json",
  "scripts/__tests__/learned-systems-status.test.mjs",
  "scripts/__tests__/learned-systems-quality-manifest.test.mjs",
  "packages/walkaround-hybrid/src/HybridEngineConfig.ts",
  "packages/walkaround-hybrid/src/HybridEngineOptions.ts",
  "packages/walkaround-hybrid/src/HybridEngine.ts",
  "packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts",
  "packages/walkaround-hybrid/src/neural/weights.ts",
  "packages/walkaround-hybrid/src/__tests__/learnedSystemConfig.test.ts",
  "packages/walkaround-hybrid/src/__tests__/capabilitiesPartition.test.ts",
  "packages/walkaround-hybrid/src/__tests__/hybridLiteTier.test.ts",
  "packages/walkaround-hybrid/src/__tests__/grisVariantPin.test.ts",
  "packages/walkaround-hybrid/src/pipeline/__tests__/nrcStructuralGate.test.ts",
  "packages/walkaround-hybrid/src/pipeline/__tests__/nrcDeviceCapability.test.ts",
  "packages/walkaround-hybrid/src/neural/nrc/__tests__/nrcGateBitIdentity.test.ts",
  "packages/walkaround-hybrid/src/pipeline/__tests__/ppgCoordinatorDiagnostics.test.ts",
  "packages/walkaround-hybrid/src/pipeline/__tests__/ppgCompilerGate.test.ts",
  "README.md",
  "plan/library-architecture.md",
  "packages/walkaround-hybrid/README.md",
  "tools/neural-denoiser-training/README.md",
  "tools/neural-denoiser-training/train.py",
  "tools/neural-denoiser-training/export_weights.py",
  "tools/neural-denoiser-training/capture-dataset.mjs",
  "tools/neural-denoiser-training/dataset_spec.md",
  "packages/walkaround-hybrid/__tests__/neuralWeightsRoundTrip.test.ts",
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
  for (const needle of [
    "production-scale dataset metadata",
    ">=500 samples",
    "1 spp noisy inputs",
    ">=4096 spp clean references",
    "albedo/normal buffers",
    "capture source",
    "tonemap",
  ]) {
    if (!String(learnedRow.remaining).includes(needle)) {
      fail(`VQ-LEARNED-SYSTEMS remaining text must cite stricter production quality gate: ${needle}`);
    }
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
if (adjointRow.status !== "committed-proof-green") {
  fail("VQ-ADJOINT-SCOPED-PATH-REPLAY must stay committed-proof-green while full-path parity is isolated in FC-ADJOINT-FULL-PATH-PARITY");
}
assertRowCitesPaths(adjointRow, REQUIRED_ADJOINT_ARTIFACT_PATHS, "VQ-ADJOINT-SCOPED-PATH-REPLAY");
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
  "evaluateBrdfWithClearcoat",
  "evaluateBrdfWithSheen",
  "evaluateBrdfWithIridescence",
  "evaluateBrdfWithAnisotropy",
  "dBrdf_dIridescenceThicknessRange",
  "dBrdf_dAnisotropyRotation",
]) {
  if (!brdfAdjointSource.includes(needle)) {
    fail(`VQ-ADJOINT-SCOPED-PATH-REPLAY BRDF adjoint CPU oracle is stale: missing ${needle}`);
  }
}
for (const needle of [
  "fn dBrdf_dBaseColor(",
  "fn dBrdf_dRoughness(",
  "fn dBrdf_dClearcoat(",
  "fn dBrdf_dSheen(",
  "fn dBrdf_dIridescence(",
  "fn dBrdf_dAnisotropy(",
]) {
  if (!pathTraceAdjointWgslSource.includes(needle)) {
    fail(`VQ-ADJOINT-SCOPED-PATH-REPLAY pathTraceAdjoint WGSL is stale: missing ${needle}`);
  }
}
for (const needle of [
  "ADJOINT_FROZEN_SEED_BASE",
  "sampleAdjointEnvironmentImportance",
  "fn adjointConcentricDiscSample",
  "meshAreaLights",
  "sampleAdjointBaseColorTexture",
  "ADJOINT_FIELD_CLEARCOAT_NORMAL_SCALE",
]) {
  if (!adjointPassWgslSource.includes(needle)) {
    fail(`VQ-ADJOINT-SCOPED-PATH-REPLAY adjoint pass WGSL is stale: missing ${needle}`);
  }
}
for (const needle of [
  "export class AdjointPass",
  "buildAdjointWorldSpaceGeometryOverride",
  "computeGradient",
  "AdjointGradientRequest",
]) {
  if (!adjointPassSource.includes(needle)) {
    fail(`VQ-ADJOINT-SCOPED-PATH-REPLAY adjoint pass host source is stale: missing ${needle}`);
  }
}
for (const needle of [
  "ADJOINT_HARNESS_WGSL",
  "ADJOINT_SHADING_FD_WGSL",
  "PT_WEBGPU_PATH_TRACE_ADJOINT_WGSL",
]) {
  if (!adjointHarnessSource.includes(needle)) {
    fail(`VQ-ADJOINT-SCOPED-PATH-REPLAY adjoint harness source is stale: missing ${needle}`);
  }
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
for (const needle of [
  "matches FD to <= 1e-4",
  "analytic KHR_materials_specular partials == finite difference",
  "analytic KHR_materials_clearcoat partials == finite difference",
  "map-free KHR_materials_anisotropy scalar partials == finite difference",
]) {
  if (!brdfAdjointTestSource.includes(needle)) {
    fail(`VQ-ADJOINT-SCOPED-PATH-REPLAY BRDF adjoint test proof is stale: missing ${needle}`);
  }
}
for (const needle of [
  "emissive",
  "emissiveIntensity",
  "iridescenceIor",
]) {
  if (!brdfAdjointEmissiveIorTestSource.includes(needle)) {
    fail(`VQ-ADJOINT-SCOPED-PATH-REPLAY emissive/IOR adjoint test proof is stale: missing ${needle}`);
  }
}
for (const needle of [
  "engine adjoint PASS bundles the real partials + re-trace + faceforward + scatter",
  "shading-adjoint kernel bundles the forward + real partials + adjoint-vs-FD",
  "bundles the REAL path-replay adjoint partials",
]) {
  if (!adjointHarnessTestSource.includes(needle)) {
    fail(`VQ-ADJOINT-SCOPED-PATH-REPLAY adjoint harness test proof is stale: missing ${needle}`);
  }
}
for (const needle of [
  "buildAdjointWorldSpaceGeometryOverride",
  "AdjointGradientRequest",
  "ADJOINT_EMITTER_TARGET_MESH",
]) {
  if (!adjointPassPackingTestSource.includes(needle)) {
    fail(`VQ-ADJOINT-SCOPED-PATH-REPLAY adjoint pass packing test proof is stale: missing ${needle}`);
  }
}
for (const needle of [
  "partials match finite differences",
  "finite area",
  "mapped mesh-area gradients",
  "ADJOINT_EMITTER_TARGET_DIRECTIONAL",
]) {
  if (!adjointEmitterGradientOracleTestSource.includes(needle)) {
    fail(`VQ-ADJOINT-SCOPED-PATH-REPLAY emitter-gradient oracle test proof is stale: missing ${needle}`);
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
if (!["engine-first", "canvas-first", "canvas-only"].includes(gltfBrowserStatusArtifact.json?.captureMode)) {
  fail("VQ-GLTF-BROWSER-PTWEBGL2 status artifact must pin a bounded browser proof capture mode");
}
if (gltfBrowserStatusArtifact.json?.["hostBlockClasses.0"] !== "engine-readback-timeout") {
  fail("VQ-GLTF-BROWSER-PTWEBGL2 status artifact must pin the current WSL host-block class");
}

const radiometricPtRow = queue.validationQueue.find((row) => row.id === "VQ-RADIOMETRIC-PT");
if (radiometricPtRow == null) fail("validationQueue missing VQ-RADIOMETRIC-PT");
if (radiometricPtRow.workClass !== "research-promotion") {
  fail("VQ-RADIOMETRIC-PT must stay classified as research-promotion rather than generic proof work");
}
const radiometricPtArtifactPaths = new Set(radiometricPtRow.proofArtifacts.map((artifact) => artifact?.path));
for (const path of REQUIRED_RADIOMETRIC_PT_ARTIFACT_PATHS) {
  if (!radiometricPtArtifactPaths.has(path)) {
    fail(`VQ-RADIOMETRIC-PT proofArtifacts must cite ${path}`);
  }
}
const ptRadiometricHostStatus = await readJson("tools/radiometric-ab/pt-ab-host-status.json");
if (ptRadiometricHostStatus.harness !== "pt-radiometric-ab") {
  fail("VQ-RADIOMETRIC-PT host status must pin harness=pt-radiometric-ab");
}
if (ptRadiometricHostStatus.verdict !== "PASS") {
  fail("VQ-RADIOMETRIC-PT host status must remain full PASS while the row claims committed proof");
}
if (ptRadiometricHostStatus.reason?.code !== "pt-radiometric-ab-complete") {
  fail("VQ-RADIOMETRIC-PT host status PASS must carry pt-radiometric-ab-complete");
}
const expectedPtRadiometricCases = [
  ["sppm", "tools/radiometric-ab/ab-sppm.mjs", "tools/radiometric-ab/results-sppm.json"],
  ["bdpt", "tools/radiometric-ab/ab-bdpt.mjs", "tools/radiometric-ab/results-bdpt.json"],
  ["restir-pt", "tools/radiometric-ab/ab-restir-pt.mjs", "tools/radiometric-ab/results-restir-pt.json"],
  ["sobol", "tools/radiometric-ab/ab-sobol.mjs", "tools/radiometric-ab/results-sobol.json"],
];
if (JSON.stringify(ptRadiometricHostStatus.selectedCases) !== JSON.stringify(expectedPtRadiometricCases.map((row) => row[0]))) {
  fail("VQ-RADIOMETRIC-PT host status must preserve the complete four-case selectedCases list");
}
if (JSON.stringify(ptRadiometricHostStatus.preservedResultFiles) !== JSON.stringify(expectedPtRadiometricCases.map((row) => row[2]))) {
  fail("VQ-RADIOMETRIC-PT host status must preserve all four result files");
}
if (Array.isArray(ptRadiometricHostStatus.nextSteps) && ptRadiometricHostStatus.nextSteps.length !== 0) {
  fail("VQ-RADIOMETRIC-PT full PASS host status must not carry host-blocked nextSteps");
}
if (!Array.isArray(ptRadiometricHostStatus.cases) || ptRadiometricHostStatus.cases.length !== expectedPtRadiometricCases.length) {
  fail("VQ-RADIOMETRIC-PT host status must include exactly four case records");
}
for (const [id, script, resultFile] of expectedPtRadiometricCases) {
  const entry = ptRadiometricHostStatus.cases.find((item) => item?.id === id);
  if (entry == null) fail(`VQ-RADIOMETRIC-PT host status missing case ${id}`);
  if (entry.status !== "PASS") fail(`VQ-RADIOMETRIC-PT host status ${id} must be PASS`);
  if (entry.exitStatus !== 0) fail(`VQ-RADIOMETRIC-PT host status ${id} must pin exitStatus 0`);
  if (entry.signal !== null) fail(`VQ-RADIOMETRIC-PT host status ${id} must pin signal null`);
  if (entry.reason !== null) fail(`VQ-RADIOMETRIC-PT host status ${id} must pin reason null`);
  if (entry.script !== script) fail(`VQ-RADIOMETRIC-PT host status ${id} script mismatch`);
  if (entry.resultFile !== resultFile) fail(`VQ-RADIOMETRIC-PT host status ${id} resultFile mismatch`);
}
for (const needle of [
  "RADIOMETRIC_AB_PROOFS",
  "RESTIR_PT_SPECIALTY_PROOF",
  "RESTIR_PT_GLOSSY_RESEARCH_PROOF",
  "ab-sppm.mjs",
  "ab-restir-pt.mjs",
  "ab-sobol.mjs",
  "ab-restir-pt-specialty.mjs",
]) {
  if (!radiometricProofSource.includes(needle) && !radiometricCheckerSource.includes(needle)) {
    fail(`VQ-RADIOMETRIC-PT non-BDPT proof source is stale: missing ${needle}`);
  }
}
for (const needle of [
  "checkSppm",
  "checkRestirPt",
  "checkSobol",
  "checkRestirPtSpecialty",
  "checkRestirPtGlossyResearch",
]) {
  if (!radiometricCheckerSource.includes(needle)) {
    fail(`VQ-RADIOMETRIC-PT proof checker is stale: missing ${needle}`);
  }
}
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
const sobolArtifact = radiometricPtRow.proofArtifacts.find((artifact) =>
  artifact?.path === "tools/radiometric-ab/results-sobol.json"
);
if (sobolArtifact == null) {
  fail("VQ-RADIOMETRIC-PT must cite results-sobol.json");
}
if (sobolArtifact.json?.["promotion.defaultReady"] !== false) {
  fail("VQ-RADIOMETRIC-PT Sobol artifact must pin promotion.defaultReady=false");
}
if (sobolArtifact.json?.["promotion.requiredEvidence"] !== "full-tier/real-adapter equal-time Sobol RMSE A/B") {
  fail("VQ-RADIOMETRIC-PT Sobol artifact must pin required equal-time real-adapter evidence");
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
  "structured research-mode warning",
  "weighted against the regular eye-path strategy",
  "not yet composed against the ordinary eye-path estimator",
  "full-tier/real-adapter equal-time Sobol RMSE A/B",
  "Sobol default promotion",
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
  "pt-webgpu.restir-pt-glossy-reuse-research-mode",
  "restirPtReuseOptions.experimentalGlossyReuse=true",
  "glossy-visible-vertex-reuse-outside-diffuse-safe-validation-envelope",
  "tools/radiometric-ab/results-restir-pt-glossy-research.json",
]) {
  if (!ptWebgpuIndexSource.includes(needle)) {
    fail(`VQ-RADIOMETRIC-PT structured research warning source is stale: missing ${needle}`);
  }
}
const ptWebgpuBdptKernelSource = await readText("packages/pt-webgpu/src/wgsl/pathTrace/kernel.wgsl.ts");
for (const needle of [
  "radiance = radiance + directLi",
  "radiance = radiance + evaluateBdptConnection",
  "bdptOptions.experimentalMultiVertex",
  "ordinary eye-path estimator at",
  "not a promotable production estimator",
]) {
  if (!ptWebgpuBdptKernelSource.includes(needle)) {
    fail(`VQ-RADIOMETRIC-PT BDPT kernel non-promotion boundary is stale: missing ${needle}`);
  }
}
const ptWebgpuBdptConnectionSource = await readText("packages/pt-webgpu/src/wgsl/bdpt/bdptConnection.wgsl.ts");
for (const needle of [
  "Full Veach",
  "bdptMISWeightFull",
  "buildBDPTStrategyPDFs_full",
  "REAL light-vertex BSDF",
]) {
  if (!ptWebgpuBdptConnectionSource.includes(needle)) {
    fail(`VQ-RADIOMETRIC-PT BDPT connection proof source is stale: missing ${needle}`);
  }
}
const ptWebgpuBdptLightSubpathSource = await readText("packages/pt-webgpu/src/wgsl/bdpt/bdptLightSubpath.wgsl.ts");
for (const needle of [
  "bdptExtendLightSubpath",
  "sampleNextBounceDirectionWithClearcoatNormal",
  "pdfRevAtPrev",
  "ONE invocation",
]) {
  if (!ptWebgpuBdptLightSubpathSource.includes(needle)) {
    fail(`VQ-RADIOMETRIC-PT BDPT light-subpath proof source is stale: missing ${needle}`);
  }
}
const ptWebgpuBdptConnectionTest = await readText("packages/pt-webgpu/src/__tests__/bdptConnectionMisFull.test.ts");
if (!ptWebgpuBdptConnectionTest.includes("MIS weight matches the oracle")) {
  fail("VQ-RADIOMETRIC-PT BDPT connection MIS oracle test citation is stale");
}
const ptWebgpuBdptLightSubpathTest = await readText("packages/pt-webgpu/src/__tests__/bdptGlossyLightSubpath.test.ts");
for (const needle of [
  "samples the REAL BSDF",
  "pdfRevAtPrev",
  "BdptLightPathBufferWebGPU",
]) {
  if (!ptWebgpuBdptLightSubpathTest.includes(needle)) {
    fail(`VQ-RADIOMETRIC-PT BDPT light-subpath test citation is stale: missing ${needle}`);
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
if (walkaroundAbRow.promotionCommand !== "npm run radiometric-ab:walkaround-glossy-spp64") {
  fail("VQ-WALKAROUND-RADIOMETRIC-AB promotionCommand must name the high-quality glossy recapture lane");
}
if (walkaroundAbRow.allCasesHighSppCommand !== "npm run radiometric-ab:walkaround-all-spp64") {
  fail("VQ-WALKAROUND-RADIOMETRIC-AB allCasesHighSppCommand must name the high-quality all-cases recapture lane");
}
if (packageJson.scripts?.["radiometric-ab:walkaround-glossy-spp64"] !== "node tools/radiometric-ab/run-walkaround-ab.mjs --glossy-spp64") {
  fail("package.json must expose the high-quality walkaround glossy recapture command");
}
if (packageJson.scripts?.["radiometric-ab:walkaround-all-spp64"] !== "node tools/radiometric-ab/run-walkaround-ab.mjs --all-spp64") {
  fail("package.json must expose the high-quality walkaround all-cases recapture command");
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
  "glossy remains a non-promotable FINDING",
  "64-SPP all-cases recapture lane",
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
const walkaroundAllSpp64Status = await readJson("tools/radiometric-ab/walkaround-ab-all-spp64-status.json");
const walkaroundAllSpp64Results = await readJson("tools/radiometric-ab/walkaround-ab-all-spp64.json");
const walkaroundPromotionStatus = await readJson("tools/radiometric-ab/walkaround-ab-promotion-status.json");
if (walkaroundAllSpp64Status.verdict !== "PASS-PARTIAL") {
  fail("walkaround all-spp64 status must pin PASS-PARTIAL");
}
if (walkaroundAllSpp64Status.renderConfig?.spp !== "64" || walkaroundAllSpp64Status.renderConfig?.qualityProfile !== "all-spp64") {
  fail("walkaround all-spp64 status must pin 64-SPP all-spp64 config");
}
if (walkaroundAllSpp64Status.reason?.code !== "walkaround-ab-partial-proof") {
  fail("walkaround all-spp64 status must pin walkaround-ab-partial-proof");
}
if (walkaroundAllSpp64Results.a8?.verdict !== "NEGLIGIBLE") fail("walkaround all-spp64 A8 verdict must stay NEGLIGIBLE");
if (walkaroundAllSpp64Results.sun?.verdict !== "PASS") fail("walkaround all-spp64 SUN verdict must stay PASS");
if (walkaroundAllSpp64Results.glass?.verdict !== "PASS") fail("walkaround all-spp64 GLASS verdict must stay PASS");
if (walkaroundAllSpp64Results.glossy?.verdict !== "FINDING") fail("walkaround all-spp64 GLOSSY verdict must stay FINDING");
if (walkaroundAllSpp64Results.glossy?.promotion?.defaultReady !== false) {
  fail("walkaround all-spp64 GLOSSY finding must pin promotion.defaultReady=false");
}
const walkaroundExpectedCaseVerdicts = {
  a8: walkaroundAbResults.a8?.verdict,
  sun: walkaroundAbResults.sun?.verdict,
  glass: walkaroundAbResults.glass?.verdict,
  glossy: walkaroundAbResults.glossy?.verdict,
};
const walkaroundExpectedHighSppVerdicts = {
  a8: walkaroundAllSpp64Results.a8?.verdict,
  sun: walkaroundAllSpp64Results.sun?.verdict,
  glass: walkaroundAllSpp64Results.glass?.verdict,
  glossy: walkaroundAllSpp64Results.glossy?.verdict,
};
if (
  walkaroundPromotionStatus.verdict !== "PASS-PARTIAL" ||
  walkaroundPromotionStatus.promotion?.defaultReady !== false ||
  walkaroundPromotionStatus.promotion?.classification !== "glossy-finding"
) {
  fail("walkaround promotion summary must pin PASS-PARTIAL/defaultReady=false/glossy-finding");
}
if (walkaroundPromotionStatus.promotion?.blocker !== "ddgi-irradiance-cache-not-ggx-filtered-radiance") {
  fail("walkaround promotion summary must pin the DDGI/GGX blocker");
}
if (!String(walkaroundPromotionStatus.promotion?.requiredEvidence ?? "").includes("browser-real-adapter")) {
  fail("walkaround promotion summary must name browser-real-adapter evidence");
}
if (JSON.stringify(walkaroundPromotionStatus.caseVerdicts) !== JSON.stringify(walkaroundExpectedCaseVerdicts)) {
  fail("walkaround promotion summary case verdicts must match the baseline result snapshot");
}
if (JSON.stringify(walkaroundPromotionStatus.highSppCaseVerdicts) !== JSON.stringify(walkaroundExpectedHighSppVerdicts)) {
  fail("walkaround promotion summary high-SPP verdicts must match the all-spp64 result snapshot");
}
if (!Array.isArray(walkaroundPromotionStatus.glossyProfiles) || walkaroundPromotionStatus.glossyProfiles.length !== 3) {
  fail("walkaround promotion summary must preserve baseline, glossy-spp64, and all-spp64 glossy profiles");
}
for (const profile of walkaroundPromotionStatus.glossyProfiles) {
  if (profile.verdict !== "FINDING" || profile.materialEffectObserved !== true) {
    fail(`walkaround promotion summary ${profile.label} must keep the glossy FINDING/material-effect evidence`);
  }
}
const walkaroundAbHarness = await readText("tools/radiometric-ab/walkaround-ab.mjs");
const walkaroundAbRunner = await readText("tools/radiometric-ab/run-walkaround-ab.mjs");
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
for (const needle of ["--all-spp64", "all-spp64", "walkaround-ab-all-spp64-status.json"]) {
  if (!walkaroundAbRunner.includes(needle)) {
    fail(`walkaround A/B runner source is stale: missing ${needle}`);
  }
}
for (const needle of [
  "WALKAROUND_AB_CASE_IDS",
  "assertWalkaroundFullFreshStatus",
  "checkWalkaroundGlossy",
  "checkWalkaroundAllSpp64Status",
  "Do not promote",
]) {
  if (!walkaroundAbChecker.includes(needle)) {
    fail(`walkaround A/B checker source is stale: missing ${needle}`);
  }
}
for (const needle of [
  "WALKAROUND_AB_RESULT_PROOF",
  "WALKAROUND_ALL_SPP64_STATUS_PROOF",
  "WALKAROUND_AB_PROMOTION_STATUS_PROOF",
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

const gltfMaterialTopologyRow = queue.validationQueue.find((row) => row.id === "VQ-GLTF-MATERIAL-TOPOLOGY");
if (gltfMaterialTopologyRow == null) fail("validationQueue missing VQ-GLTF-MATERIAL-TOPOLOGY");
assertRowCitesPaths(
  gltfMaterialTopologyRow,
  REQUIRED_GLTF_MATERIAL_TOPOLOGY_ARTIFACT_PATHS,
  "VQ-GLTF-MATERIAL-TOPOLOGY",
);
for (const needle of [
  "REQUIRED_SWEEP_MAPS",
  "behavioral-gate-dzn-gltf-material-sweep-status.json",
  "materialMapCount",
  "SWEEP_MAPS",
  "FIELD_TEXTURE_INDEX",
]) {
  if (!gltfMaterialProofSource.includes(needle) && !gltfMaterialFixtureSource.includes(needle)) {
    fail(`VQ-GLTF-MATERIAL-TOPOLOGY material proof source is stale: missing ${needle}`);
  }
}
for (const needle of [
  "baseColorMap",
  "normalMap",
  "specularColorMap",
  "clearcoatNormalMap",
  "iridescenceThicknessMap",
  "anisotropyMap",
  "thicknessMap",
]) {
  if (!gltfMaterialFixtureSource.includes(needle)) {
    fail(`VQ-GLTF-MATERIAL-TOPOLOGY material fixture is stale: missing ${needle}`);
  }
}
for (const needle of [
  "REQUIRED_TOPOLOGY_PROOFS",
  "fallback-generated-mesh",
  "adapter-generated-triangle-list",
  "POINTS",
  "LINE_STRIP",
  "TRIANGLE_FAN",
]) {
  if (!gltfTopologyProofSource.includes(needle)) {
    fail(`VQ-GLTF-MATERIAL-TOPOLOGY topology proof source is stale: missing ${needle}`);
  }
}
for (const needle of [
  "pt/gltf-material-sweep",
  "gltfReal",
  "gltf: \"material-sweep\"",
  "configMatchesFilter",
  "cfg.label.startsWith(\"pt/cwbvh-\") && !filter.includes(\"cwbvh\")",
]) {
  if (!behavioralGateSource.includes(needle)) {
    fail(`VQ-GLTF-MATERIAL-TOPOLOGY behavioral gate source is stale: missing ${needle}`);
  }
}
for (const needle of [
  "bakePtWebgpuLiteCompatibleVertexColors",
  "ptWebgpuLiteBakeableVertexColor",
  "canBakeLiteVertexColors",
  "materialVariantBindings",
]) {
  if (!gltfAssetLoaderSource.includes(needle)) {
    fail(`VQ-GLTF-MATERIAL-TOPOLOGY asset loader source is stale: missing ${needle}`);
  }
}
for (const needle of [
  "allows direct pt-webgpu-lite strict loads for primitive-constant RGB COLOR_0 scenes",
  "primitive.colors).toBeUndefined",
  "rejects direct pt-webgpu-lite strict loads for constant COLOR_0 material-variant scenes",
]) {
  if (!gltfAssetApiTestSource.includes(needle)) {
    fail(`VQ-GLTF-MATERIAL-TOPOLOGY gltfAssetApi test source is stale: missing ${needle}`);
  }
}
const materialSweepStatus = await readJson("tools/behavioral-gate/behavioral-gate-dzn-gltf-material-sweep-status.json");
if (materialSweepStatus.verdict !== "PASS" || materialSweepStatus.exitStatus !== 0) {
  fail("VQ-GLTF-MATERIAL-TOPOLOGY material sweep dzn status must pin PASS/exitStatus=0");
}
if (materialSweepStatus.configs?.[0]?.label !== "pt/gltf-material-sweep") {
  fail("VQ-GLTF-MATERIAL-TOPOLOGY material sweep dzn status must pin pt/gltf-material-sweep");
}
if (materialSweepStatus.configs?.[0]?.goldenStatus !== "ok") {
  fail("VQ-GLTF-MATERIAL-TOPOLOGY material sweep dzn status must pin goldenStatus=ok");
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
if (promiseLedger.includes("rest of the texture-map family is not sampled")) {
  fail(
    "promise ledger has stale walkaround texture-map prose: sampled atlas-backed map families " +
    "must not be described as unsampled",
  );
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
