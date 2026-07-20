// @ts-check
// Data tables extracted verbatim from check-validation-queue.mjs (D17-2).
// Single source for the ~450 lines of hardcoded manifest/required-path tables the
// Road-to-100 validation-queue checker asserts against. Values are byte-identical
// to the pre-extraction inline consts — every pin must still fire. Do NOT edit
// values here without matching the source-of-truth artifacts they mirror.

export const ALLOWED_STATUSES = new Set([
  "committed-proof-green",
  "partial-proof-green",
  "host-blocked",
  "evidence-needed",
  "provisioning-needed",
  "decision-needed",
  "future-contract",
]);

export const UNRESOLVED_VALIDATION_STATUSES = new Set([
  "partial-proof-green",
  "host-blocked",
  "evidence-needed",
  "provisioning-needed",
  "decision-needed",
]);

export const ALLOWED_EXECUTION_SCOPES = new Set([
  "external-browser-host",
  "external-real-adapter-validation",
  "external-real-adapter-throughput",
  "research-design-and-real-adapter-validation",
  "asset-provisioning-and-quality-ab",
]);

export const REQUIRED_VALIDATION_IDS = [
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

export const REQUIRED_FUTURE_IDS = [
  "FC-DISPLACEMENT-MICROTESSELLATION",
  "FC-TRANSPARENT-GI-TRANSPORT",
  "FC-WALKAROUND-SPECIALTY-MATERIAL-TRANSPORT",
  "FC-NATIVE-POINT-LINE",
  "FC-ARBITRARY-UV-ARRAYS",
  "FC-NATIVE-INSTANCED-SKINNED-MORPHED",
  "FC-ADJOINT-FULL-PATH-PARITY",
];

export const REQUIRED_FUTURE_BLOCKER_NEEDLES = new Map([
  ["FC-DISPLACEMENT-MICROTESSELLATION", ["tessellation", "BVH"]],
  ["FC-TRANSPARENT-GI-TRANSPORT", ["reservoir", "DDGI/RC"]],
  ["FC-WALKAROUND-SPECIALTY-MATERIAL-TRANSPORT", ["spectral", "PT backends"]],
  ["FC-NATIVE-POINT-LINE", ["core point/line", "backend fidelity"]],
  ["FC-ARBITRARY-UV-ARRAYS", ["TextureRef", "shader descriptor"]],
  ["FC-NATIVE-INSTANCED-SKINNED-MORPHED", ["instanced-skinned", "TLAS/BLAS"]],
  ["FC-ADJOINT-FULL-PATH-PARITY", ["differentiable transport", "scoped direct-light replay"]],
]);

export const REQUIRED_MUTATION_KINDS = [
  "material",
  "environment",
  "emitter",
  "transform",
  "topology",
  "instanced-count",
  "add-primitive",
  "remove-primitive",
];

export const REQUIRED_MUTATION_ARTIFACT_PATHS = [
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

export const REQUIRED_ADJOINT_ARTIFACT_PATHS = [
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

export const REQUIRED_PT_WEBGPU_RUNTIME_ARTIFACT_PATHS = [
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

export const REQUIRED_GLTF_REAL_ARTIFACT_PATHS = [
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

export const REQUIRED_GLTF_REAL_MANIFEST_ROWS = [
  {
    assetId: `box-textured-glb`,
    label: `pt/gltf-real-box-textured`,
    kind: `textured-glb`,
    requiredExtensions: [],
    baseGoldenPath: `tools/reference-renders/gltf-real-behavioral/pt-gltf-real-box-textured.png`,
    dznFullGoldenPath: `tools/reference-renders/gltf-real-behavioral-dzn-full/pt-gltf-real-box-textured.png`,
    thresholds: { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 },
  },
  {
    assetId: `cesium-milk-truck-draco`,
    label: `pt/gltf-real-draco`,
    kind: `draco`,
    requiredExtensions: [`KHR_draco_mesh_compression`],
    baseGoldenPath: `tools/reference-renders/gltf-real-behavioral/pt-gltf-real-draco.png`,
    dznFullGoldenPath: `tools/reference-renders/gltf-real-behavioral-dzn-full/pt-gltf-real-draco.png`,
    thresholds: { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 },
  },
  {
    assetId: `meshopt-cube-real`,
    label: `pt/gltf-real-meshopt`,
    kind: `meshopt`,
    requiredExtensions: [`KHR_meshopt_compression`],
    baseGoldenPath: `tools/reference-renders/gltf-real-behavioral/pt-gltf-real-meshopt.png`,
    dznFullGoldenPath: `tools/reference-renders/gltf-real-behavioral-dzn-full/pt-gltf-real-meshopt.png`,
    thresholds: { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 },
  },
];

export const REQUIRED_GLTF_BROWSER_PROVENANCE_GOLDEN_FILES = [
  {
    assetId: "box-textured-glb",
    path: "tools/reference-renders/gltf-real-browser-pt-webgl2/pt-webgl2-gltf-real-box-textured.png",
  },
  {
    assetId: "cesium-milk-truck-draco",
    path: "tools/reference-renders/gltf-real-browser-pt-webgl2/pt-webgl2-gltf-real-draco.png",
  },
  {
    assetId: "meshopt-cube-real",
    path: "tools/reference-renders/gltf-real-browser-pt-webgl2/pt-webgl2-gltf-real-meshopt.png",
  },
];
export const REQUIRED_GLTF_BROWSER_MANIFEST = {
  kind: `vitrum-browser-gltf-pt-webgl2-goldens`,
  backend: `pt-webgl2`,
  browserHarness: `tools/gltf-browser-proof/capture-pt-webgl2-real.mjs`,
  resolution: [64, 64],
  samplesPerPixel: 1,
  updateCommand: `node tools/gltf-browser-proof/capture-pt-webgl2-real.mjs --update-golden`,
  checkCommand: `node tools/gltf-browser-proof/capture-pt-webgl2-real.mjs`,
  residualQueue: [`Browser PNG readback/golden capture for all three rows on a host that can read WebGL2 canvases.`],
};

export const REQUIRED_GLTF_BROWSER_MANIFEST_ROWS = [
  {
    assetId: `box-textured-glb`,
    label: `browser/pt-webgl2-gltf-real-box-textured`,
    kind: `textured-glb`,
    minTextures: 1,
    requiredExtensions: [],
    requiredHooks: [],
    goldenPath: `tools/reference-renders/gltf-real-browser-pt-webgl2/pt-webgl2-gltf-real-box-textured.png`,
    thresholds: { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 },
  },
  {
    assetId: `cesium-milk-truck-draco`,
    label: `browser/pt-webgl2-gltf-real-draco`,
    kind: `draco`,
    minTextures: 0,
    requiredExtensions: [`KHR_draco_mesh_compression`],
    requiredHooks: [`draco`],
    goldenPath: `tools/reference-renders/gltf-real-browser-pt-webgl2/pt-webgl2-gltf-real-draco.png`,
    thresholds: { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 },
  },
  {
    assetId: `meshopt-cube-real`,
    label: `browser/pt-webgl2-gltf-real-meshopt`,
    kind: `meshopt`,
    minTextures: 0,
    requiredExtensions: [`KHR_meshopt_compression`],
    requiredHooks: [`meshopt`],
    goldenPath: `tools/reference-renders/gltf-real-browser-pt-webgl2/pt-webgl2-gltf-real-meshopt.png`,
    thresholds: { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 },
  },
];

export const REQUIRED_GLTF_MATERIAL_TOPOLOGY_ARTIFACT_PATHS = [
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

export const REQUIRED_GLTF_MATERIAL_SWEEP_MANIFEST = {
  kind: `vitrum-gltf-material-sweep-behavioral-goldens`,
  backend: `pt-webgpu`,
  adapter: `lavapipe`,
  resolution: [64, 64],
  samplesPerPixel: 8,
  traceTier: `auto`,
  fixture: `synthetic-material-sweep`,
  label: `pt/gltf-material-sweep`,
  goldenPath: `tools/reference-renders/gltf-material-sweep-behavioral/pt-gltf-material-sweep.png`,
  dznStatusPath: `tools/behavioral-gate/behavioral-gate-dzn-gltf-material-sweep-status.json`,
  thresholds: { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 },
  materialMapCount: 18,
  compareFullTierDznCommand: `npm run behavioral-gate:dzn -- --filter gltf-material-sweep --require-full-tier`,
};

export const REQUIRED_GLTF_TOPOLOGY_MANIFEST_ROWS = [
  {
    kind: `vitrum-gltf-point-line-behavioral-goldens`,
    id: `point-line-fallback`,
    label: `pt/gltf-point-line-fallback`,
    fixture: `synthetic-points-lines-loop-strip`,
    sourceModes: [`POINTS`, `LINES`, `LINE_LOOP`, `LINE_STRIP`],
    proof: `fallback-generated-mesh`,
    goldenPath: `tools/reference-renders/gltf-point-line-behavioral/pt-gltf-point-line-fallback.png`,
    manifestPath: `tools/reference-renders/gltf-point-line-behavioral/manifest.json`,
    thresholds: { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 },
  },
  {
    kind: `vitrum-gltf-triangle-topology-behavioral-goldens`,
    id: `triangle-strip-fan`,
    label: `pt/gltf-triangle-strip-fan`,
    fixture: `synthetic-triangle-strip-fan`,
    sourceModes: [`TRIANGLE_STRIP`, `TRIANGLE_FAN`],
    proof: `adapter-generated-triangle-list`,
    goldenPath: `tools/reference-renders/gltf-triangle-topology-behavioral/pt-gltf-triangle-strip-fan.png`,
    manifestPath: `tools/reference-renders/gltf-triangle-topology-behavioral/manifest.json`,
    thresholds: { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 },
  },
];

export const REQUIRED_WALKAROUND_BEHAVIORAL_ROWS = [
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

export const REQUIRED_WALKAROUND_BEHAVIORAL_ARTIFACT_PATHS = [
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

export const REQUIRED_RENDERER_FIDELITY_ARTIFACT_PATHS = [
  "tools/renderer-fidelity-proof/check-proofs.mjs",
  "tools/renderer-fidelity-proof/promotion-status.json",
  "plan/renderer-fidelity-matrix.md",
  "plan/fidelity-promotion-playbook.md",
  "README.md",
  "plan/library-architecture.md",
  "HARDWARE-VALIDATION-NEEDS.md",
  "plan/gap-closure-execution-plan.md",
  "tools/gltf-browser-proof/pt-webgl2-real-status.json",
  "tools/gltf-browser-proof/pt-webgl2-real-canvas-first-status.json",
  "tools/reference-renders/gltf-real-browser-pt-webgl2/manifest.json",
  "tools/behavioral-gate/behavioral-gate-dzn-spectral-status.json",
  "tools/behavioral-gate/behavioral-gate-dzn-light-status.json",
  "tools/behavioral-gate/behavioral-gate-dzn-caustic-status.json",
  "tools/behavioral-gate/behavioral-gate-dzn-bdpt-status.json",
  "tools/radiometric-ab/pt-promotion-status.json",
  "tools/radiometric-ab/results-bdpt.json",
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

export const REQUIRED_RENDERER_FIDELITY_SOURCE_STATUS_PATHS = [
  "tools/gltf-browser-proof/pt-webgl2-real-status.json",
  "tools/gltf-browser-proof/pt-webgl2-real-canvas-first-status.json",
  "tools/reference-renders/gltf-real-browser-pt-webgl2/manifest.json",
  "tools/behavioral-gate/behavioral-gate-dzn-spectral-status.json",
  "tools/behavioral-gate/behavioral-gate-dzn-light-status.json",
  "tools/behavioral-gate/behavioral-gate-dzn-caustic-status.json",
  "tools/behavioral-gate/behavioral-gate-dzn-bdpt-status.json",
  "tools/radiometric-ab/pt-promotion-status.json",
  "tools/radiometric-ab/results-bdpt.json",
];

export const REQUIRED_WALKAROUND_AB_ARTIFACT_PATHS = [
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

export const REQUIRED_RADIOMETRIC_PT_ARTIFACT_PATHS = [
  "tools/radiometric-ab/pt-ab-host-status.json",
  "tools/radiometric-ab/ab-sppm.mjs",
  "tools/radiometric-ab/ab-bdpt.mjs",
  "tools/radiometric-ab/ab-restir-pt.mjs",
  "tools/radiometric-ab/ab-restir-pt-glossy-research.mjs",
  "tools/radiometric-ab/ab-restir-pt-specialty.mjs",
  "tools/radiometric-ab/ab-sobol.mjs",
  "tools/radiometric-ab/check-results.mjs",
  "tools/radiometric-ab/proofs.mjs",
  "tools/radiometric-ab/pt-promotion-status.json",
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

export const REQUIRED_PT_RADIOMETRIC_PROMOTION_SOURCE_STATUS_PATHS = [
  "tools/radiometric-ab/pt-ab-host-status.json",
  "tools/radiometric-ab/results-sppm.json",
  "tools/radiometric-ab/results-bdpt.json",
  "tools/radiometric-ab/results-restir-pt.json",
  "tools/radiometric-ab/results-restir-pt-specialty.json",
  "tools/radiometric-ab/results-restir-pt-glossy-research.json",
  "tools/radiometric-ab/results-sobol.json",
];

export const REQUIRED_WALKAROUND_PROMOTION_SOURCE_STATUS_PATHS = [
  "tools/radiometric-ab/walkaround-ab-host-status.json",
  "tools/radiometric-ab/walkaround-ab-glossy-spp64-status.json",
  "tools/radiometric-ab/walkaround-ab-all-spp64-status.json",
];

export const REQUIRED_WALKAROUND_PROMOTION_SOURCE_RESULT_PATHS = [
  "tools/radiometric-ab/walkaround-ab-results.json",
  "tools/radiometric-ab/walkaround-ab-glossy-spp64.json",
  "tools/radiometric-ab/walkaround-ab-all-spp64.json",
];

export const REQUIRED_CWBVH_PROMOTION_SOURCE_STATUS_PATHS = [
  "tools/behavioral-gate/behavioral-gate-dzn-cwbvh-binary-parity-status.json",
  "tools/behavioral-gate/behavioral-gate-dzn-cwbvh-complex-parity-status.json",
  "tools/behavioral-gate/behavioral-gate-dzn-cwbvh-broader-status.json",
];
