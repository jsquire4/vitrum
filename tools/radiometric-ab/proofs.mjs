// @ts-check
// Metadata for committed radiometric A/B result snapshots.

export const RADIOMETRIC_AB_PROOFS = [
  {
    id: "sppm",
    ab: "sppm-vs-manifold-nee",
    scriptPath: "tools/radiometric-ab/ab-sppm.mjs",
    resultPath: "tools/radiometric-ab/results-sppm.json",
    resolution: { W: 80, H: 80 },
    reference: { strategy: "manifold-nee", frames: 80 },
    checkpoints: [20, 50, 80],
    thresholds: {
      finalRelErrMax: 5.0,
      monotonicRelErrSlack: 1.5,
    },
  },
  {
    id: "bdpt",
    ab: "bdpt-vs-unidirectional",
    scriptPath: "tools/radiometric-ab/ab-bdpt.mjs",
    resultPath: "tools/radiometric-ab/results-bdpt.json",
    resolution: { W: 80, H: 80 },
    meanFrames: 60,
    varianceRuns: 8,
    varianceFramesPerRun: 8,
    thresholds: {
      globalRelErrMax: 0.10,
      varRatioMax: 2.0,
    },
    controls: {
      endpointOnlyMatchesUni: true,
      multiVertexFindingStartsAt: 2,
    },
  },
  {
    id: "restir-pt",
    ab: "restir-pt-reuse-on-vs-off",
    scriptPath: "tools/radiometric-ab/ab-restir-pt.mjs",
    resultPath: "tools/radiometric-ab/results-restir-pt.json",
    resolution: { W: 80, H: 80 },
    meanFrames: 60,
    varianceRuns: 8,
    varianceFramesPerRun: 8,
    thresholds: {
      globalRelErrMax: 0.10,
      varRatioMax: 3.0,
    },
  },
];

export const RESTIR_PT_SPECIALTY_PROOF = {
  schema: "vitrum.restir-pt.specialty-reference.v1",
  mode: "cpu-static",
  scriptPath: "tools/radiometric-ab/ab-restir-pt-specialty.mjs",
  resultPath: "tools/radiometric-ab/results-restir-pt-specialty.json",
  coverage: {
    specialtyLobes: ["anisotropy", "clearcoat", "iridescence", "sheen"],
    requiresGpuRecapture: false,
  },
  summary: {
    caseCount: 3,
    maxAbsoluteError: 0,
    maxRelativeError: 0,
  },
};

export const WALKAROUND_AB_HOST_STATUS_PROOF = {
  harness: "walkaround-ab",
  statusPath: "tools/radiometric-ab/walkaround-ab-host-status.json",
  preservedResultFile: "tools/radiometric-ab/walkaround-ab-results.json",
  expectedVerdict: "HOST-BLOCKED",
  reasonCode: "deno-wgpu-hal-gles-index-oob",
};

export const WALKAROUND_AB_RESULT_PROOF = {
  resultPath: "tools/radiometric-ab/walkaround-ab-results.json",
  resolution: "128x128",
  spp: 16,
  cases: {
    a8: {
      id: "A8",
      allowedVerdicts: ["NEGLIGIBLE", "SMALL"],
      maxAbsOverallDelta: 0.03,
    },
    sun: {
      id: "SUN",
      allowedVerdicts: ["PASS", "PASS-PARTIAL"],
      maxAnalyticRatioError: 0.5,
    },
    glass: {
      id: "GLASS",
      allowedVerdicts: ["PASS", "PASS-WEAK"],
      minCentreRatio: 0.5,
    },
    glossy: {
      id: "GLOSSY",
      allowedVerdicts: ["PASS"],
      minFloorRatio: 0.8,
    },
  },
};

/** @param {string} id */
export function proofForRadiometricAb(id) {
  return RADIOMETRIC_AB_PROOFS.find((proof) => proof.id === id) ?? null;
}
