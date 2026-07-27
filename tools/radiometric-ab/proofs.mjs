// @ts-check
// Metadata for committed radiometric A/B result snapshots.

import { PT_RADIOMETRIC_RUNTIME_SOURCE_ROOTS } from "./resultProvenance.mjs";

export const RADIOMETRIC_AB_PROOFS = [
  {
    id: "sppm",
    schema: "vitrum.radiometric-ab.result.v1",
    ab: "sppm-vs-manifold-nee",
    scriptPath: "tools/radiometric-ab/ab-sppm.mjs",
    resultPath: "tools/radiometric-ab/results-sppm.json",
    sourceRoots: PT_RADIOMETRIC_RUNTIME_SOURCE_ROOTS,
    resolution: { W: 80, H: 80 },
    reference: { strategy: "manifold-nee", frames: 80 },
    checkpoints: [20, 50, 80],
    thresholds: { finalRelErrMax: 0.50, monotonicRelErrSlack: 1.5 },
  },
  {
    id: "bdpt",
    schema: "vitrum.radiometric-ab.result.v1",
    ab: "bdpt-vs-unidirectional",
    scriptPath: "tools/radiometric-ab/ab-bdpt.mjs",
    resultPath: "tools/radiometric-ab/results-bdpt.json",
    sourceRoots: PT_RADIOMETRIC_RUNTIME_SOURCE_ROOTS,
    resolution: { W: 80, H: 80 },
    meanFrames: 60,
    varianceRuns: 8,
    varianceFramesPerRun: 8,
    thresholds: { globalRelErrMax: 0.10, varRatioMax: 2.0 },
    controls: {
      depths: [1, 2, 3, 8],
      endpointGlobalRelErrMax: 0.10,
      endpointRoiRelErrMax: 0.15,
    },
  },
  {
    id: "restir-pt",
    schema: "vitrum.radiometric-ab.result.v1",
    ab: "restir-pt-reuse-on-vs-off",
    scriptPath: "tools/radiometric-ab/ab-restir-pt.mjs",
    resultPath: "tools/radiometric-ab/results-restir-pt.json",
    sourceRoots: PT_RADIOMETRIC_RUNTIME_SOURCE_ROOTS,
    resolution: { W: 80, H: 80 },
    roi: { x0: 20, y0: 25, x1: 60, y1: 55 },
    meanFrames: 60,
    varianceRuns: 8,
    varianceFramesPerRun: 8,
    thresholds: {
      globalRelErrMax: 0.10,
      pairedEquivalenceMargin: 0.10,
      varRatioMax: 2.0,
    },
    pairedSeedAnalysis: {
      confidenceLevel: 0.95,
      tCritical: 2.364624251,
      runs: 8,
    },
    capture: {
      scene: "cornell-indirect-v1",
      traceTier: "full",
      colorSpace: "linear",
      requireFullTier: true,
      requireRadiometricSignal: true,
      maxBounces: 6,
      effectiveMClamp: 20,
      seedSchedule: {
        schema: "vitrum.radiometric-ab.seed-schedule.v1",
        arithmetic: "unsigned-32-bit-wrap",
        multiplier: "6364136223846793005",
        increment: "1442695040888963407",
        runStride: "97",
        meanFormula: "u32((frame + seedOffset) * multiplier + increment)",
        varianceFormula:
          "u32((run * framesPerRun + frame) * multiplier + increment + run * runStride)",
      },
    },
    clampControls: [10, 100],
    diagnosticClamp: 10,
    professionalDefaultWeightCeiling: 3.4028234663852886e38,
  },
  {
    id: "sobol",
    schema: "vitrum.radiometric-ab.result.v1",
    ab: "sobol-equal-frame-rmse",
    scriptPath: "tools/radiometric-ab/ab-sobol.mjs",
    resultPath: "tools/radiometric-ab/results-sobol.json",
    sourceRoots: PT_RADIOMETRIC_RUNTIME_SOURCE_ROOTS,
    resolution: { width: 80, height: 80 },
    traceTier: "lite",
    reference: {
      sampling: "pcg",
      frames: 40,
      note:
        "Higher-frame PCG reference. Candidate arms use equal frame budgets and record wall time.",
    },
    candidateFrames: 12,
    sceneIds: ["cornell-indirect", "caustic-floor"],
    thresholds: {
      maxGlobalRmseRatio: 1.5,
      maxRoiRmseRatio: 1.5,
      maxElapsedMsRatio: 20.0,
    },
    defaultSelection: {
      selected: false,
      evidenceClass: "wsl-lite-equal-frame-proxy",
      reason:
        "PCG remains the default until full-tier equal-time measurements justify changing it.",
      requiredEvidence: "full-tier/real-adapter equal-time Sobol RMSE A/B",
    },
  },
];

export const RESTIR_PT_SPECIALTY_PROOF = {
  schema: "vitrum.restir-pt.specialty-reference.v1",
  mode: "cpu-static",
  scriptPath: "tools/radiometric-ab/ab-restir-pt-specialty.mjs",
  resultPath: "tools/radiometric-ab/results-restir-pt-specialty.json",
  coverage: {
    specialtyLobes: [
      "anisotropy",
      "clearcoat",
      "iridescence",
      "sheen",
      "specular",
    ],
    materialSources: ["map-backed-effective-values", "scalar"],
    requiresGpuRecapture: false,
  },
  summary: {
    caseCount: 4,
    maxAbsoluteError: 0,
    maxRelativeError: 0,
    luminanceChecksum: 10.258282571792,
    pdfChecksum: 4.024098414883,
  },
  cases: [
    {
      id: "clearcoat-sheen",
      materialSource: "scalar",
      activeLobes: ["clearcoat", "sheen"],
      minAbsLobeDeltaFromNeutral: 0.1,
    },
    {
      id: "iridescent-anisotropic",
      materialSource: "scalar",
      activeLobes: ["iridescence", "anisotropy"],
      minAbsLobeDeltaFromNeutral: 0.1,
    },
    {
      id: "all-specialty-lobes",
      materialSource: "scalar",
      activeLobes: ["clearcoat", "sheen", "iridescence", "anisotropy"],
      minAbsLobeDeltaFromNeutral: 0.1,
    },
    {
      id: "map-backed-effective-lobes",
      materialSource: "map-backed-effective-values",
      activeLobes: [
        "clearcoat",
        "sheen",
        "iridescence",
        "anisotropy",
        "specular",
      ],
      minAbsLobeDeltaFromNeutral: 0.1,
    },
  ],
};

/** Test-source and production-source pins; `npm test` executes the named tests. */
export const PT_LOCAL_ACCEPTANCE_PROOFS = {
  bdpt: {
    paths: [
      "packages/pt-webgpu/src/__tests__/bdptConnectionMisFull.test.ts",
      "packages/pt-webgpu/src/__tests__/bdptEstimatorOwnership.test.ts",
      "packages/pt-webgpu/src/__tests__/bdptDeltaTransport.test.ts",
    ],
    sourcePath: "packages/pt-webgpu/src/ptWebgpuValidation.ts",
    needles: ["BDPT_DEFAULT_LIGHT_BOUNCES = 2", "BDPT_MAX_LIGHT_BOUNCES = 8"],
  },
  restirPtGlossy: {
    paths: [
      "packages/pt-webgpu/src/__tests__/restirPtGlossyReuseClosure.test.ts",
      "packages/pt-webgpu/src/__tests__/restirPtReuseContract.test.ts",
    ],
    sourcePath: "packages/pt-webgpu/src/index.ts",
    needles: [
      "readonly restirPtReuse?: boolean",
      "readonly mClamp?: number",
      "readonly wCap?: number",
    ],
  },
  sobol: {
    paths: ["packages/pt-webgpu/src/__tests__/samplingOptions.test.ts"],
    sourcePath: "packages/core/src/engine/promiseLedger.ts",
    needles: [
      "default: 'pcg'",
      "lowDiscrepancyDimensions: 4",
      "continuation: 'independent-pcg'",
    ],
  },
  cwbvh: {
    paths: [
      "packages/pt-webgpu/src/__tests__/cwbvhTraversalWiring.test.ts",
      "packages/pt-webgpu/src/__tests__/cwbvhTlasDifferential.test.ts",
    ],
    sourcePath: "packages/pt-webgpu/src/ptWebgpuValidation.ts",
    needles: [
      "const cwbvhClosestRequested = opts.bvhTraversal === 'cwbvh-closest'",
      "assertCwbvhClosestSupported",
    ],
  },
  sppm: {
    paths: ["packages/pt-webgpu/src/__tests__/sppmProductionClosure.test.ts"],
    sourcePath: "packages/pt-webgpu/src/capabilities.ts",
    needles: [
      "pt-webgpu-photon-map-sppm",
      "causticStrategy: flags.traceTier === 'lite' ? 'none' : flags.causticStrategy",
    ],
  },
  liteProfile: {
    paths: ["packages/pt-webgpu/src/__tests__/liteTierCapabilities.test.ts"],
    sourcePath: "packages/pt-webgpu/src/index.ts",
    needles: [
      "readonly backendProfileId: 'pt-webgpu' | 'pt-webgpu-lite'",
      "this.backendProfileId = traceTier === 'lite' ? 'pt-webgpu-lite' : 'pt-webgpu'",
    ],
  },
};

/** @param {string} id */
export function proofForRadiometricAb(id) {
  return RADIOMETRIC_AB_PROOFS.find((proof) => proof.id === id) ?? null;
}
