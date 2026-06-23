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
      depths: [1, 2, 3],
      endpointOnlyMatchesUni: true,
      // Fresh dzn recaptures can differ from the UNI arm by a few 1e-9 in the
      // serialized mean due to float accumulation/readback order; keep the
      // endpoint-only invariant tight without requiring exact textual zero.
      endpointOnlyMaxRelErr: 1e-7,
      multiVertexFindingStartsAt: 2,
      multiVertexMinGlobalRelErr: 0.10,
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
  {
    id: "sobol",
    ab: "sobol-equal-frame-rmse",
    scriptPath: "tools/radiometric-ab/ab-sobol.mjs",
    resultPath: "tools/radiometric-ab/results-sobol.json",
    resolution: { width: 80, height: 80 },
    traceTier: "lite",
    reference: { sampling: "pcg", frames: 40 },
    candidateFrames: 12,
    sceneIds: ["cornell-indirect", "caustic-floor"],
    thresholds: {
      maxGlobalRmseRatio: 1.5,
      maxRoiRmseRatio: 1.5,
      maxElapsedMsRatio: 20.0,
    },
    promotion: {
      defaultReady: false,
      evidenceClass: "wsl-lite-equal-frame-proxy",
      reason: "WSL-lite evidence bounds correctness but does not show equal-time convergence superiority.",
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
    specialtyLobes: ["anisotropy", "clearcoat", "iridescence", "sheen", "specular"],
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
      activeLobes: ["clearcoat", "sheen", "iridescence", "anisotropy", "specular"],
      minAbsLobeDeltaFromNeutral: 0.1,
    },
  ],
};

export const RESTIR_PT_GLOSSY_RESEARCH_PROOF = {
  ab: "restir-pt-glossy-research-vs-base",
  mode: "research",
  scriptPath: "tools/radiometric-ab/ab-restir-pt-glossy-research.mjs",
  resultPath: "tools/radiometric-ab/results-restir-pt-glossy-research.json",
  resolution: { W: 80, H: 80 },
  meanFrames: 60,
  varianceRuns: 8,
  varianceFramesPerRun: 8,
  thresholds: {
    globalRelErrMax: 0.10,
    varRatioMax: 3.0,
  },
  candidate: {
    restirPtReuse: true,
    restirPtReuseOptions: { experimentalGlossyReuse: true },
  },
  reference: { restirPtReuse: false },
  allowedVerdicts: ["PASS", "FINDING"],
  promotion: {
    defaultReady: false,
  },
};

export const BDPT_MULTIVERTEX_RESEARCH_PROOF = {
  resultPath: "tools/radiometric-ab/results-bdpt.json",
  sourcePath: "packages/pt-webgpu/src/index.ts",
  warningCode: "pt-webgpu.bdpt-multivertex-research-mode",
  blocker: "not-weighted-against-regular-eye-path-strategy",
  requiredEstimator: "multi-vertex-light-subpath-strategies-weighted-against-regular-eye-path-strategy",
  evidencePath: "tools/radiometric-ab/results-bdpt.json",
  promotion: {
    defaultReady: false,
  },
  controls: {
    findingStartsAt: 2,
    minFindingGlobalRelErr: 0.10,
  },
};

export const PT_RADIOMETRIC_AB_HOST_STATUS_PROOF = {
  harness: "pt-radiometric-ab",
  statusPath: "tools/radiometric-ab/pt-ab-host-status.json",
  preservedResultFiles: [
    "tools/radiometric-ab/results-sppm.json",
    "tools/radiometric-ab/results-bdpt.json",
    "tools/radiometric-ab/results-restir-pt.json",
    "tools/radiometric-ab/results-sobol.json",
  ],
  allowedVerdicts: ["PASS", "PASS-PARTIAL", "HOST-BLOCKED"],
  blockedReasonCodes: [
    "pt-radiometric-ab-timeout",
    "pt-radiometric-full-tier-unavailable",
    "pt-radiometric-no-adapter",
    "pt-radiometric-deno-wgpu-panic",
  ],
};

export const WALKAROUND_AB_HOST_STATUS_PROOF = {
  harness: "walkaround-ab",
  statusPath: "tools/radiometric-ab/walkaround-ab-host-status.json",
  preservedResultFile: "tools/radiometric-ab/walkaround-ab-results.json",
  allowedVerdicts: ["PASS", "PASS-PARTIAL", "HOST-BLOCKED"],
  blockedReasonCodes: ["deno-wgpu-hal-gles-index-oob", "walkaround-ab-timeout"],
  partialReasonCode: "walkaround-ab-partial-proof",
};

export const WALKAROUND_AB_RESULT_PROOF = {
  resultPath: "tools/radiometric-ab/walkaround-ab-results.json",
  resolution: "128x128",
  spp: 16,
  cases: {
    a8: {
      id: "A8",
      expectedVerdict: "NEGLIGIBLE",
      allowedVerdicts: ["NEGLIGIBLE", "SMALL"],
      maxAbsOverallDelta: 0.03,
      maxAbsRegionDelta: 0.03,
      maxRelativeOverallDelta: 0.05,
    },
    sun: {
      id: "SUN",
      expectedVerdict: "PASS",
      allowedVerdicts: ["PASS", "PASS-PARTIAL"],
      maxAnalyticRatioError: 0.5,
    },
    glass: {
      id: "GLASS",
      expectedVerdict: "PASS",
      allowedVerdicts: ["PASS", "SMOKE"],
      minCentreRatio: 0.5,
      minSignalDeltaForPass: 1e-4,
    },
    glossy: {
      id: "GLOSSY",
      expectedVerdict: "FINDING",
      allowedVerdicts: ["PASS", "PASS-WEAK", "FINDING"],
      sampleRegion: "visible-back-wall-center-crop",
      minSampleRatio: 0.8,
      // Legacy field retained while committed result snapshots still expose
      // floorRatio for older readers. The sampled region is the visible
      // back-wall center crop, not the geometric floor.
      minFloorRatio: 0.8,
      minSignalDeltaForPass: 1e-4,
      promotion: {
        defaultReady: false,
        blocker: "ddgi-irradiance-cache-not-ggx-filtered-radiance",
        requiredEvidence: "material-furnace-reference-ab-and-browser-real-adapter-recapture",
      },
    },
  },
};

export const WALKAROUND_GLOSSY_SPP64_STATUS_PROOF = {
  harness: "walkaround-ab",
  statusPath: "tools/radiometric-ab/walkaround-ab-glossy-spp64-status.json",
  preservedResultFile: "tools/radiometric-ab/walkaround-ab-glossy-spp64.json",
  selectedCases: "glossy",
  expectedRenderConfig: {
    width: "128",
    height: "128",
    spp: "64",
    qualityProfile: "glossy-spp64",
  },
  allowedVerdicts: ["PASS", "PASS-PARTIAL", "HOST-BLOCKED"],
  blockedReasonCodes: ["deno-wgpu-hal-gles-index-oob", "walkaround-ab-timeout"],
  partialReasonCode: "walkaround-ab-partial-proof",
  doNotPromoteText: "Do not promote",
};

export const WALKAROUND_ALL_SPP64_STATUS_PROOF = {
  harness: "walkaround-ab",
  statusPath: "tools/radiometric-ab/walkaround-ab-all-spp64-status.json",
  preservedResultFile: "tools/radiometric-ab/walkaround-ab-all-spp64.json",
  selectedCases: null,
  expectedRenderConfig: {
    width: "128",
    height: "128",
    spp: "64",
    qualityProfile: "all-spp64",
  },
  allowedVerdicts: ["PASS", "PASS-PARTIAL", "HOST-BLOCKED"],
  blockedReasonCodes: ["deno-wgpu-hal-gles-index-oob", "walkaround-ab-timeout"],
  partialReasonCode: "walkaround-ab-partial-proof",
  doNotPromoteText: "Do not promote",
};

export const WALKAROUND_AB_PROMOTION_STATUS_PROOF = {
  harness: "walkaround-ab-promotion-proof",
  statusPath: "tools/radiometric-ab/walkaround-ab-promotion-status.json",
  verdict: "PASS-PARTIAL",
  promotion: {
    defaultReady: false,
    classification: "glossy-finding",
    blocker: "ddgi-irradiance-cache-not-ggx-filtered-radiance",
    requiredEvidence: "material-furnace-reference-ab-and-browser-real-adapter-recapture",
  },
  sourceStatuses: [
    WALKAROUND_AB_HOST_STATUS_PROOF.statusPath,
    WALKAROUND_GLOSSY_SPP64_STATUS_PROOF.statusPath,
    WALKAROUND_ALL_SPP64_STATUS_PROOF.statusPath,
  ],
  glossyProfiles: [
    {
      label: "baseline",
      resultPath: WALKAROUND_AB_RESULT_PROOF.resultPath,
      resultKey: "glossy",
      expectedQualityProfile: "baseline",
      expectedSpp: WALKAROUND_AB_RESULT_PROOF.spp,
    },
    {
      label: "glossy-spp64",
      resultPath: WALKAROUND_GLOSSY_SPP64_STATUS_PROOF.preservedResultFile,
      resultKey: "glossy",
      expectedQualityProfile: "glossy-spp64",
      expectedSpp: 64,
    },
    {
      label: "all-spp64",
      resultPath: WALKAROUND_ALL_SPP64_STATUS_PROOF.preservedResultFile,
      resultKey: "glossy",
      expectedQualityProfile: "all-spp64",
      expectedSpp: 64,
    },
  ],
};

/** @param {string} id */
export function proofForRadiometricAb(id) {
  return RADIOMETRIC_AB_PROOFS.find((proof) => proof.id === id) ?? null;
}
