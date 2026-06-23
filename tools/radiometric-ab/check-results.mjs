#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
// Verifies that committed radiometric A/B result JSONs still satisfy their proof metadata.

import {
  BDPT_MULTIVERTEX_RESEARCH_PROOF,
  PT_RADIOMETRIC_AB_HOST_STATUS_PROOF,
  PT_RADIOMETRIC_PROMOTION_STATUS_PROOF,
  RADIOMETRIC_AB_PROOFS,
  RESTIR_PT_GLOSSY_RESEARCH_PROOF,
  RESTIR_PT_SPECIALTY_PROOF,
  WALKAROUND_AB_HOST_STATUS_PROOF,
  WALKAROUND_AB_PROMOTION_STATUS_PROOF,
  WALKAROUND_AB_RESULT_PROOF,
  WALKAROUND_ALL_SPP64_STATUS_PROOF,
  WALKAROUND_GLOSSY_SPP64_STATUS_PROOF,
} from "./proofs.mjs";

const REQUIRED_RADIOMETRIC_AB_ROWS = [
  {
    id: "sppm",
    scriptPath: "tools/radiometric-ab/ab-sppm.mjs",
    resultPath: "tools/radiometric-ab/results-sppm.json",
  },
  {
    id: "bdpt",
    scriptPath: "tools/radiometric-ab/ab-bdpt.mjs",
    resultPath: "tools/radiometric-ab/results-bdpt.json",
  },
  {
    id: "restir-pt",
    scriptPath: "tools/radiometric-ab/ab-restir-pt.mjs",
    resultPath: "tools/radiometric-ab/results-restir-pt.json",
  },
  {
    id: "sobol",
    scriptPath: "tools/radiometric-ab/ab-sobol.mjs",
    resultPath: "tools/radiometric-ab/results-sobol.json",
  },
];

const REQUIRED_RESTIR_PT_SPECIALTY = {
  schema: "vitrum.restir-pt.specialty-reference.v1",
  mode: "cpu-static",
  scriptPath: "tools/radiometric-ab/ab-restir-pt-specialty.mjs",
  resultPath: "tools/radiometric-ab/results-restir-pt-specialty.json",
  specialtyLobes: ["anisotropy", "clearcoat", "iridescence", "sheen", "specular"],
  materialSources: ["map-backed-effective-values", "scalar"],
  caseCount: 4,
  luminanceChecksum: 10.258282571792,
  pdfChecksum: 4.024098414883,
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

const WALKAROUND_AB_CASE_IDS = ["a8", "sun", "glass", "glossy"];

/** @param {string} message */
function fail(message) {
  throw new Error(`[radiometric-ab-proof-check] ${message}`);
}

/**
 * @param {unknown} a
 * @param {unknown} b
 */
function sameJson(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

/**
 * @param {unknown} value
 * @param {string} label
 */
function assertFiniteNumber(value, label) {
  if (typeof value !== "number" || !Number.isFinite(value)) fail(`${label} must be a finite number`);
}

function assertRequiredRadiometricRows() {
  const byId = new Map();
  for (const proof of RADIOMETRIC_AB_PROOFS) {
    if (byId.has(proof.id)) fail(`duplicate radiometric proof id ${proof.id}`);
    byId.set(proof.id, proof);
  }
  for (const required of REQUIRED_RADIOMETRIC_AB_ROWS) {
    const proof = byId.get(required.id);
    if (!proof) fail(`missing required radiometric proof row ${required.id}`);
    if (proof.scriptPath !== required.scriptPath) {
      fail(`${required.id}: scriptPath differs from required radiometric proof contract`);
    }
    if (proof.resultPath !== required.resultPath) {
      fail(`${required.id}: resultPath differs from required radiometric proof contract`);
    }
  }
}

/**
 * @param {any} proof
 * @param {any} result
 */
async function assertCommon(proof, result) {
  if (result.ab !== proof.ab) fail(`${proof.id}: result ab ${result.ab} differs from proofs.mjs`);
  if (result.verdict !== "PASS") fail(`${proof.id}: committed result verdict is ${result.verdict}`);
  if (!sameJson(result.resolution, proof.resolution)) fail(`${proof.id}: resolution differs from proofs.mjs`);
  for (const [key, value] of Object.entries(result.roi ?? {})) {
    assertFiniteNumber(value, `${proof.id}: roi.${key}`);
  }
  const scriptUrl = new URL(`../../${proof.scriptPath}`, import.meta.url);
  const scriptStat = await Deno.stat(scriptUrl);
  if (!scriptStat.isFile) fail(`${proof.id}: script path is missing`);
}

/**
 * @param {any} proof
 * @param {any} result
 */
function checkSppm(proof, result) {
  if (result.reference?.strategy !== proof.reference.strategy) fail("sppm: reference strategy differs from proofs.mjs");
  if (result.reference?.frames !== proof.reference.frames) fail("sppm: reference frame count differs from proofs.mjs");
  if (!Array.isArray(result.sppm)) fail("sppm: result.sppm must be an array");
  /** @type {any[]} */
  const sppmEntries = result.sppm;
  const frames = sppmEntries.map((entry) => entry.frames);
  if (!sameJson(frames, proof.checkpoints)) fail(`sppm: checkpoints ${JSON.stringify(frames)} differ from proofs.mjs`);
  if (result.converging !== true) fail("sppm: committed result must have converging=true");
  if (result.inBallpark !== true) fail("sppm: committed result must have inBallpark=true");
  let prevRelErr = null;
  for (const entry of result.sppm) {
    assertFiniteNumber(entry.lum, `sppm: checkpoint ${entry.frames} lum`);
    assertFiniteNumber(entry.globalLum, `sppm: checkpoint ${entry.frames} globalLum`);
    assertFiniteNumber(entry.rmse, `sppm: checkpoint ${entry.frames} rmse`);
    assertFiniteNumber(entry.relErr, `sppm: checkpoint ${entry.frames} relErr`);
    if (prevRelErr != null && entry.relErr > prevRelErr * proof.thresholds.monotonicRelErrSlack) {
      fail(`sppm: relErr ${entry.relErr} exceeds monotonic slack after ${prevRelErr}`);
    }
    prevRelErr = entry.relErr;
  }
  const finalRelErr = result.sppm[result.sppm.length - 1]?.relErr;
  assertFiniteNumber(finalRelErr, "sppm: final relErr");
  if (finalRelErr >= proof.thresholds.finalRelErrMax) {
    fail(`sppm: final relErr ${finalRelErr} exceeds ${proof.thresholds.finalRelErrMax}`);
  }
}

/**
 * @param {any} proof
 * @param {any} result
 */
function checkBdpt(proof, result) {
  if (result.meanFrames !== proof.meanFrames) fail("bdpt: meanFrames differs from proofs.mjs");
  if (result.varianceRuns !== proof.varianceRuns) fail("bdpt: varianceRuns differs from proofs.mjs");
  if (result.varianceFramesPerRun !== proof.varianceFramesPerRun) fail("bdpt: varianceFramesPerRun differs from proofs.mjs");
  if (result.meanAgreement !== true) fail("bdpt: committed result must have meanAgreement=true");
  if (result.varianceImproved !== true) fail("bdpt: committed result must have varianceImproved=true");
  assertFiniteNumber(result.globalRelErr, "bdpt: globalRelErr");
  assertFiniteNumber(result.varRatio, "bdpt: varRatio");
  if (result.globalRelErr >= proof.thresholds.globalRelErrMax) {
    fail(`bdpt: globalRelErr ${result.globalRelErr} exceeds ${proof.thresholds.globalRelErrMax}`);
  }
  if (result.varRatio > proof.thresholds.varRatioMax) {
    fail(`bdpt: varRatio ${result.varRatio} exceeds ${proof.thresholds.varRatioMax}`);
  }
  if (result.controls?.endpointOnlyMatchesUni !== proof.controls.endpointOnlyMatchesUni) {
    fail("bdpt: endpointOnlyMatchesUni differs from proofs.mjs");
  }
  if (result.controls?.multiVertexFindingStartsAt !== proof.controls.multiVertexFindingStartsAt) {
    fail("bdpt: multiVertexFindingStartsAt differs from proofs.mjs");
  }
  /** @type {any[]} */
  const controls = result.controls?.byMaxLightBounces ?? [];
  const controlDepths = controls.map((entry) => entry.maxLightBounces);
  if (!sameJson(controlDepths, proof.controls.depths)) {
    fail(`bdpt: control depths ${JSON.stringify(controlDepths)} differ from proofs.mjs`);
  }
  const endpoint = controls.find((entry) => entry.maxLightBounces === 1);
  if (endpoint == null) fail("bdpt: missing maxLightBounces=1 endpoint-only control");
  assertFiniteNumber(endpoint.globalRelErr, "bdpt: endpoint globalRelErr");
  assertFiniteNumber(endpoint.roiRelErr, "bdpt: endpoint roiRelErr");
  if (
    endpoint.globalRelErr > proof.controls.endpointOnlyMaxRelErr ||
    endpoint.roiRelErr > proof.controls.endpointOnlyMaxRelErr
  ) {
    fail(
      `bdpt: endpoint-only control drifted from UNI ` +
      `(global=${endpoint.globalRelErr}, roi=${endpoint.roiRelErr})`,
    );
  }
  for (const entry of controls) {
    if (entry.maxLightBounces < proof.controls.multiVertexFindingStartsAt) continue;
    assertFiniteNumber(entry.globalRelErr, `bdpt: maxLightBounces=${entry.maxLightBounces} globalRelErr`);
    if (entry.globalRelErr < proof.controls.multiVertexMinGlobalRelErr) {
      fail(
        `bdpt: maxLightBounces=${entry.maxLightBounces} no longer records the expected ` +
        `multi-vertex finding (${entry.globalRelErr} < ${proof.controls.multiVertexMinGlobalRelErr})`,
      );
    }
  }
}

/** @param {any} proof */
async function checkBdptMultiVertexResearch(proof) {
  const resultUrl = new URL(`../../${proof.resultPath}`, import.meta.url);
  const result = JSON.parse(await Deno.readTextFile(resultUrl));
  if (result.ab !== "bdpt-vs-unidirectional") fail("bdpt multi-vertex: result ab mismatch");
  if (proof.promotion?.defaultReady !== false) {
    fail("bdpt multi-vertex: promotion.defaultReady must be false");
  }
  const controls = /** @type {any[]} */ (result.controls?.byMaxLightBounces ?? []);
  const firstFinding = controls.find((entry) =>
    entry.maxLightBounces === proof.controls.findingStartsAt
  );
  if (firstFinding == null) {
    fail(`bdpt multi-vertex: missing maxLightBounces=${proof.controls.findingStartsAt} control`);
  }
  assertFiniteNumber(firstFinding.globalRelErr, "bdpt multi-vertex: first finding globalRelErr");
  if (firstFinding.globalRelErr < proof.controls.minFindingGlobalRelErr) {
    fail(
      `bdpt multi-vertex: first finding globalRelErr ${firstFinding.globalRelErr} ` +
      `< ${proof.controls.minFindingGlobalRelErr}`,
    );
  }
  if (result.controls?.multiVertexFindingStartsAt !== proof.controls.findingStartsAt) {
    fail("bdpt multi-vertex: multiVertexFindingStartsAt differs from proof metadata");
  }
  const promotion = result.controls?.multiVertexPromotion ?? null;
  if (promotion == null || typeof promotion !== "object") {
    fail("bdpt multi-vertex: result must carry controls.multiVertexPromotion metadata");
  }
  const finding = result.researchFindings?.bdptMultiVertex ?? null;
  if (finding == null || typeof finding !== "object") {
    fail("bdpt multi-vertex: result must carry researchFindings.bdptMultiVertex metadata");
  }
  if (promotion.defaultReady !== proof.promotion.defaultReady) {
    fail("bdpt multi-vertex: result promotion.defaultReady differs from proof metadata");
  }
  for (const [key, expected] of Object.entries({
    warningCode: proof.warningCode,
    blocker: proof.blocker,
    requiredEstimator: proof.requiredEstimator,
    evidencePath: proof.evidencePath,
  })) {
    if (promotion[key] !== expected) {
      fail(`bdpt multi-vertex: result promotion.${key} must be ${expected}`);
    }
    if (finding[key] !== expected) {
      fail(`bdpt multi-vertex: result researchFindings.bdptMultiVertex.${key} must be ${expected}`);
    }
  }
  if (finding.defaultReady !== proof.promotion.defaultReady) {
    fail("bdpt multi-vertex: result research finding defaultReady differs from proof metadata");
  }
  if (finding.firstFindingMaxLightBounces !== proof.controls.findingStartsAt) {
    fail("bdpt multi-vertex: top-level firstFindingMaxLightBounces differs from proof metadata");
  }
  assertFiniteNumber(finding.firstFindingGlobalRelErr, "bdpt multi-vertex: top-level first finding globalRelErr");
  if (finding.firstFindingGlobalRelErr !== firstFinding.globalRelErr) {
    fail("bdpt multi-vertex: top-level firstFindingGlobalRelErr differs from control run");
  }
  const source = await Deno.readTextFile(new URL(`../../${proof.sourcePath}`, import.meta.url));
  for (const needle of [
    proof.warningCode,
    proof.blocker,
    proof.requiredEstimator,
    proof.evidencePath,
    "promotionReady: false",
  ]) {
    if (!source.includes(needle)) {
      fail(`bdpt multi-vertex: source warning missing ${needle}`);
    }
  }
}

/**
 * @param {any} proof
 * @param {any} result
 */
function checkRestirPt(proof, result) {
  if (result.meanFrames !== proof.meanFrames) fail("restir-pt: meanFrames differs from proofs.mjs");
  if (result.varianceRuns !== proof.varianceRuns) fail("restir-pt: varianceRuns differs from proofs.mjs");
  if (result.varianceFramesPerRun !== proof.varianceFramesPerRun) fail("restir-pt: varianceFramesPerRun differs from proofs.mjs");
  if (result.meanAgreement !== true) fail("restir-pt: committed result must have meanAgreement=true");
  if (result.varianceNotWorse !== true) fail("restir-pt: committed result must have varianceNotWorse=true");
  assertFiniteNumber(result.globalRelErr, "restir-pt: globalRelErr");
  assertFiniteNumber(result.varRatio, "restir-pt: varRatio");
  if (result.globalRelErr >= proof.thresholds.globalRelErrMax) {
    fail(`restir-pt: globalRelErr ${result.globalRelErr} exceeds ${proof.thresholds.globalRelErrMax}`);
  }
  if (result.varRatio > proof.thresholds.varRatioMax) {
    fail(`restir-pt: varRatio ${result.varRatio} exceeds ${proof.thresholds.varRatioMax}`);
  }
}

/**
 * @param {any} proof
 * @param {any} result
 */
function checkSobol(proof, result) {
  if (result.traceTier !== proof.traceTier) fail("sobol: traceTier differs from proofs.mjs");
  if (result.reference?.sampling !== proof.reference.sampling) fail("sobol: reference sampling differs from proofs.mjs");
  if (result.reference?.frames !== proof.reference.frames) fail("sobol: reference frame count differs from proofs.mjs");
  if (result.candidateFrames !== proof.candidateFrames) fail("sobol: candidate frame count differs from proofs.mjs");
  if (result.promotion == null || typeof result.promotion !== "object") {
    fail("sobol: result must carry promotion metadata");
  }
  for (const [key, expected] of Object.entries(proof.promotion ?? {})) {
    if (result.promotion[key] !== expected) {
      fail(`sobol: result promotion.${key} must be ${String(expected)}`);
    }
  }
  const finding = result.researchFindings?.sobolDefault ?? null;
  if (finding == null || typeof finding !== "object") {
    fail("sobol: result must carry researchFindings.sobolDefault metadata");
  }
  for (const [key, expected] of Object.entries({
    defaultReady: proof.researchFindings?.sobolDefault?.defaultReady,
    evidenceClass: proof.researchFindings?.sobolDefault?.evidenceClass,
    requiredEvidence: proof.researchFindings?.sobolDefault?.requiredEvidence,
    evidencePath: proof.researchFindings?.sobolDefault?.evidencePath,
  })) {
    if (finding[key] !== expected) {
      fail(`sobol: researchFindings.sobolDefault.${key} must be ${String(expected)}`);
    }
  }
  if (result.thresholds?.maxGlobalRmseRatio !== proof.thresholds.maxGlobalRmseRatio) {
    fail("sobol: maxGlobalRmseRatio differs from proofs.mjs");
  }
  if (result.thresholds?.maxRoiRmseRatio !== proof.thresholds.maxRoiRmseRatio) {
    fail("sobol: maxRoiRmseRatio differs from proofs.mjs");
  }
  if (result.thresholds?.maxElapsedMsRatio !== proof.thresholds.maxElapsedMsRatio) {
    fail("sobol: maxElapsedMsRatio differs from proofs.mjs");
  }
  /** @type {any[]} */
  const scenes = result.scenes ?? [];
  const sceneIds = scenes.map((scene) => scene.id);
  if (!sameJson(sceneIds, proof.sceneIds)) {
    fail(`sobol: scene ids ${JSON.stringify(sceneIds)} differ from proofs.mjs`);
  }
  for (const scene of scenes) {
    if (scene.referenceFrames !== proof.reference.frames) {
      fail(`sobol ${scene.id}: referenceFrames differs from proofs.mjs`);
    }
    if (scene.candidateFrames !== proof.candidateFrames) {
      fail(`sobol ${scene.id}: candidateFrames differs from proofs.mjs`);
    }
    if (scene.pass !== true) fail(`sobol ${scene.id}: scene pass must be true`);
    assertFiniteNumber(scene.pcg?.globalRmse, `sobol ${scene.id}: pcg.globalRmse`);
    assertFiniteNumber(scene.sobol?.globalRmse, `sobol ${scene.id}: sobol.globalRmse`);
    assertFiniteNumber(scene.pcg?.roiRmse, `sobol ${scene.id}: pcg.roiRmse`);
    assertFiniteNumber(scene.sobol?.roiRmse, `sobol ${scene.id}: sobol.roiRmse`);
    assertFiniteNumber(scene.ratios?.globalRmse, `sobol ${scene.id}: ratios.globalRmse`);
    assertFiniteNumber(scene.ratios?.roiRmse, `sobol ${scene.id}: ratios.roiRmse`);
    assertFiniteNumber(scene.ratios?.elapsedMs, `sobol ${scene.id}: ratios.elapsedMs`);
    if (scene.ratios.globalRmse > proof.thresholds.maxGlobalRmseRatio) {
      fail(`sobol ${scene.id}: global RMSE ratio ${scene.ratios.globalRmse} exceeds ${proof.thresholds.maxGlobalRmseRatio}`);
    }
    if (scene.ratios.roiRmse > proof.thresholds.maxRoiRmseRatio) {
      fail(`sobol ${scene.id}: ROI RMSE ratio ${scene.ratios.roiRmse} exceeds ${proof.thresholds.maxRoiRmseRatio}`);
    }
    if (scene.ratios.elapsedMs > proof.thresholds.maxElapsedMsRatio) {
      fail(`sobol ${scene.id}: elapsed ratio ${scene.ratios.elapsedMs} exceeds ${proof.thresholds.maxElapsedMsRatio}`);
    }
  }
  const maxGlobalRmseRatio = Math.max(...scenes.map((scene) => scene.ratios.globalRmse));
  const maxRoiRmseRatio = Math.max(...scenes.map((scene) => scene.ratios.roiRmse));
  const maxElapsedMsRatio = Math.max(...scenes.map((scene) => scene.ratios.elapsedMs));
  for (const [key, expected] of Object.entries({
    maxGlobalRmseRatio,
    maxRoiRmseRatio,
    maxElapsedMsRatio,
  })) {
    assertFiniteNumber(finding[key], `sobol: researchFindings.sobolDefault.${key}`);
    if (finding[key] !== expected) {
      fail(`sobol: researchFindings.sobolDefault.${key} must match committed scene ratios`);
    }
  }
}

/** @param {any} proof */
async function checkRestirPtSpecialty(proof) {
  if (proof.schema !== REQUIRED_RESTIR_PT_SPECIALTY.schema) fail("restir-pt-specialty: proof schema drifted");
  if (proof.mode !== REQUIRED_RESTIR_PT_SPECIALTY.mode) fail("restir-pt-specialty: proof mode drifted");
  if (proof.scriptPath !== REQUIRED_RESTIR_PT_SPECIALTY.scriptPath) fail("restir-pt-specialty: proof scriptPath drifted");
  if (proof.resultPath !== REQUIRED_RESTIR_PT_SPECIALTY.resultPath) fail("restir-pt-specialty: proof resultPath drifted");
  if (!sameJson(proof.coverage?.specialtyLobes, REQUIRED_RESTIR_PT_SPECIALTY.specialtyLobes)) {
    fail("restir-pt-specialty: proof specialtyLobes drifted");
  }
  if (!sameJson(proof.coverage?.materialSources, REQUIRED_RESTIR_PT_SPECIALTY.materialSources)) {
    fail("restir-pt-specialty: proof materialSources drifted");
  }
  if (proof.summary?.caseCount !== REQUIRED_RESTIR_PT_SPECIALTY.caseCount) {
    fail("restir-pt-specialty: proof caseCount drifted");
  }
  if (proof.summary?.luminanceChecksum !== REQUIRED_RESTIR_PT_SPECIALTY.luminanceChecksum) {
    fail("restir-pt-specialty: proof luminanceChecksum drifted");
  }
  if (proof.summary?.pdfChecksum !== REQUIRED_RESTIR_PT_SPECIALTY.pdfChecksum) {
    fail("restir-pt-specialty: proof pdfChecksum drifted");
  }
  if (!sameJson(proof.cases, REQUIRED_RESTIR_PT_SPECIALTY.cases)) {
    fail("restir-pt-specialty: proof cases drifted");
  }

  const resultUrl = new URL(`../../${proof.resultPath}`, import.meta.url);
  const result = JSON.parse(await Deno.readTextFile(resultUrl));
  const scriptUrl = new URL(`../../${proof.scriptPath}`, import.meta.url);
  const scriptStat = await Deno.stat(scriptUrl);
  if (!scriptStat.isFile) fail("restir-pt-specialty: script path is missing");
  if (result.schema !== proof.schema) fail(`restir-pt-specialty: schema ${result.schema} differs from proofs.mjs`);
  if (result.mode !== proof.mode) fail(`restir-pt-specialty: mode ${result.mode} differs from proofs.mjs`);
  if (!sameJson(result.coverage?.specialtyLobes, proof.coverage.specialtyLobes)) {
    fail("restir-pt-specialty: specialtyLobes coverage differs from proofs.mjs");
  }
  if (!sameJson(result.coverage?.materialSources, proof.coverage.materialSources)) {
    fail("restir-pt-specialty: materialSources coverage differs from proofs.mjs");
  }
  if (result.coverage?.requiresGpuRecapture !== proof.coverage.requiresGpuRecapture) {
    fail("restir-pt-specialty: requiresGpuRecapture differs from proofs.mjs");
  }
  if (result.summary?.caseCount !== proof.summary.caseCount) {
    fail("restir-pt-specialty: caseCount differs from proofs.mjs");
  }
  if (result.summary?.maxAbsoluteError !== proof.summary.maxAbsoluteError) {
    fail("restir-pt-specialty: maxAbsoluteError differs from proofs.mjs");
  }
  if (result.summary?.maxRelativeError !== proof.summary.maxRelativeError) {
    fail("restir-pt-specialty: maxRelativeError differs from proofs.mjs");
  }
  if (result.summary?.luminanceChecksum !== proof.summary.luminanceChecksum) {
    fail("restir-pt-specialty: luminanceChecksum differs from proofs.mjs");
  }
  if (result.summary?.pdfChecksum !== proof.summary.pdfChecksum) {
    fail("restir-pt-specialty: pdfChecksum differs from proofs.mjs");
  }
  /** @type {any[]} */
  const cases = result.cases ?? [];
  if (cases.length !== proof.cases.length) {
    fail(`restir-pt-specialty: expected ${proof.cases.length} cases, got ${cases.length}`);
  }
  const byId = new Map(cases.map((entry) => [String(entry.id), entry]));
  const actualCaseIds = [...byId.keys()].sort();
  const expectedCases = /** @type {any[]} */ (proof.cases);
  const expectedCaseIds = expectedCases.map((entry) => entry.id).sort();
  if (!sameJson(actualCaseIds, expectedCaseIds)) {
    fail(`restir-pt-specialty: case ids ${JSON.stringify(actualCaseIds)} differ from proofs.mjs`);
  }
  for (const expected of proof.cases) {
    const entry = byId.get(expected.id);
    if (entry == null) fail(`restir-pt-specialty: missing case ${expected.id}`);
    if (entry.materialSource !== expected.materialSource) {
      fail(`restir-pt-specialty ${expected.id}: materialSource differs from proofs.mjs`);
    }
    if (!sameJson(entry.activeLobes, expected.activeLobes)) {
      fail(`restir-pt-specialty ${expected.id}: activeLobes differ from proofs.mjs`);
    }
    assertFiniteNumber(entry.reference?.pdfSrc, `restir-pt-specialty ${expected.id}: reference.pdfSrc`);
    assertFiniteNumber(entry.restirPt?.luminance, `restir-pt-specialty ${expected.id}: restirPt.luminance`);
    assertFiniteNumber(entry.ab?.absDiff, `restir-pt-specialty ${expected.id}: ab.absDiff`);
    assertFiniteNumber(entry.ab?.relativeError, `restir-pt-specialty ${expected.id}: ab.relativeError`);
    assertFiniteNumber(entry.ab?.lobeDeltaFromNeutral, `restir-pt-specialty ${expected.id}: ab.lobeDeltaFromNeutral`);
    if (entry.reference.pdfSrc <= 0) fail(`restir-pt-specialty ${expected.id}: reference.pdfSrc must be positive`);
    if (entry.restirPt.luminance <= 0) fail(`restir-pt-specialty ${expected.id}: restirPt.luminance must be positive`);
    if (entry.ab.absDiff !== 0) fail(`restir-pt-specialty ${expected.id}: absDiff must stay zero`);
    if (entry.ab.relativeError !== 0) fail(`restir-pt-specialty ${expected.id}: relativeError must stay zero`);
    if (Math.abs(entry.ab.lobeDeltaFromNeutral) < expected.minAbsLobeDeltaFromNeutral) {
      fail(
        `restir-pt-specialty ${expected.id}: lobeDeltaFromNeutral ${entry.ab.lobeDeltaFromNeutral} ` +
        `is below ${expected.minAbsLobeDeltaFromNeutral}`,
      );
    }
  }
}

/** @param {any} proof */
async function checkRestirPtGlossyResearch(proof) {
  const scriptUrl = new URL(`../../${proof.scriptPath}`, import.meta.url);
  const scriptStat = await Deno.stat(scriptUrl);
  if (!scriptStat.isFile) fail("restir-pt-glossy-research: script path is missing");

  const resultUrl = new URL(`../../${proof.resultPath}`, import.meta.url);
  const result = JSON.parse(await Deno.readTextFile(resultUrl));
  if (result.ab !== proof.ab) fail("restir-pt-glossy-research: ab differs from proofs.mjs");
  if (result.mode !== proof.mode) fail("restir-pt-glossy-research: mode differs from proofs.mjs");
  if (!sameJson(result.resolution, proof.resolution)) {
    fail("restir-pt-glossy-research: resolution differs from proofs.mjs");
  }
  if (result.meanFrames !== proof.meanFrames) {
    fail("restir-pt-glossy-research: meanFrames differs from proofs.mjs");
  }
  if (result.varianceRuns !== proof.varianceRuns) {
    fail("restir-pt-glossy-research: varianceRuns differs from proofs.mjs");
  }
  if (result.varianceFramesPerRun !== proof.varianceFramesPerRun) {
    fail("restir-pt-glossy-research: varianceFramesPerRun differs from proofs.mjs");
  }
  if (!sameJson(result.thresholds, proof.thresholds)) {
    fail("restir-pt-glossy-research: thresholds differ from proofs.mjs");
  }
  if (!sameJson(result.reference, proof.reference)) {
    fail("restir-pt-glossy-research: reference options differ from proofs.mjs");
  }
  if (!sameJson(result.candidate, proof.candidate)) {
    fail("restir-pt-glossy-research: candidate options differ from proofs.mjs");
  }
  if (!proof.allowedVerdicts.includes(result.verdict)) {
    fail(`restir-pt-glossy-research: verdict ${result.verdict} is outside ${proof.allowedVerdicts.join(", ")}`);
  }
  if (result.promotion?.defaultReady !== proof.promotion.defaultReady) {
    fail("restir-pt-glossy-research: promotion.defaultReady must remain false");
  }
  const finding = result.researchFindings?.restirPtGlossyResearch ?? null;
  if (finding == null || typeof finding !== "object") {
    fail("restir-pt-glossy-research: result must carry researchFindings.restirPtGlossyResearch metadata");
  }
  for (const [key, expected] of Object.entries({
    warningCode: proof.warningCode,
    blocker: proof.blocker,
    requiredEvidence: proof.requiredEvidence,
    evidencePath: proof.resultPath,
  })) {
    if (finding[key] !== expected) {
      fail(`restir-pt-glossy-research: researchFindings.restirPtGlossyResearch.${key} must be ${expected}`);
    }
  }
  if (finding.defaultReady !== proof.promotion.defaultReady) {
    fail("restir-pt-glossy-research: research finding defaultReady differs from proof metadata");
  }
  /** @type {Array<[string, string[]]>} */
  const finiteGroups = [
    ["base", ["globalLum", "roiLum", "variance"]],
    ["glossyResearch", ["globalLum", "roiLum", "variance"]],
  ];
  for (const [group, fields] of finiteGroups) {
    for (const field of fields) {
      assertFiniteNumber(result[group]?.[field], `restir-pt-glossy-research: ${group}.${field}`);
    }
  }
  for (const field of ["globalRelErr", "roiRelErr", "varRatio"]) {
    assertFiniteNumber(result[field], `restir-pt-glossy-research: ${field}`);
  }
  if (result.meanAgreement !== (result.globalRelErr < proof.thresholds.globalRelErrMax)) {
    fail("restir-pt-glossy-research: meanAgreement does not match committed globalRelErr");
  }
  if (result.varianceNotWorse !== (result.varRatio <= proof.thresholds.varRatioMax)) {
    fail("restir-pt-glossy-research: varianceNotWorse does not match committed varRatio");
  }
  const expectedVerdict = result.meanAgreement && result.varianceNotWorse ? "PASS" : "FINDING";
  if (result.verdict !== expectedVerdict) {
    fail(`restir-pt-glossy-research: verdict ${result.verdict} should be ${expectedVerdict}`);
  }
  if (finding.verdict !== result.verdict) {
    fail("restir-pt-glossy-research: research finding verdict differs from result verdict");
  }
  if (finding.globalRelErr !== result.globalRelErr) {
    fail("restir-pt-glossy-research: research finding globalRelErr differs from result");
  }
  if (finding.varRatio !== result.varRatio) {
    fail("restir-pt-glossy-research: research finding varRatio differs from result");
  }
}

/** @param {any} proof */
async function checkPtRadiometricHostStatus(proof) {
  const statusUrl = new URL(`../../${proof.statusPath}`, import.meta.url);
  const status = JSON.parse(await Deno.readTextFile(statusUrl));
  if (status.harness !== proof.harness) fail(`pt-radiometric-ab: harness ${status.harness} differs from proofs.mjs`);
  if (!proof.allowedVerdicts.includes(status.verdict)) {
    fail(`pt-radiometric-ab: verdict ${status.verdict} is outside ${proof.allowedVerdicts.join(", ")}`);
  }
  for (const resultFile of proof.preservedResultFiles) {
    const resultUrl = new URL(`../../${resultFile}`, import.meta.url);
    const resultStat = await Deno.stat(resultUrl);
    if (!resultStat.isFile || resultStat.size <= 2) {
      fail(`pt-radiometric-ab: preserved result ${resultFile} is missing or empty`);
    }
  }
  if (!sameJson(status.preservedResultFiles, proof.preservedResultFiles)) {
    fail("pt-radiometric-ab: preservedResultFiles must match proofs.mjs exactly");
  }
  /** @type {any[]} */
  const cases = status.cases ?? [];
  if (cases.length === 0) fail("pt-radiometric-ab: status must include at least one case");
  /** @type {any[]} */
  const expectedEntries = [];
  for (const resultFile of proof.preservedResultFiles) {
    const entry = RADIOMETRIC_AB_PROOFS.find((candidate) => candidate.resultPath === resultFile);
    if (!entry) fail(`pt-radiometric-ab: no proof metadata found for preserved result ${resultFile}`);
    expectedEntries.push(entry);
  }
  const expectedById = new Map();
  const expectedCaseIds = [];
  for (const entry of expectedEntries) {
    expectedById.set(entry.id, entry);
    expectedCaseIds.push(entry.id);
  }
  expectedCaseIds.sort();
  const actualCaseIds = cases.map((entry) => entry.id).sort();
  if (!sameJson(actualCaseIds, expectedCaseIds)) {
    fail(
      `pt-radiometric-ab: status case ids ${JSON.stringify(actualCaseIds)} ` +
      `must match proof ids ${JSON.stringify(expectedCaseIds)}`,
      );
  }
  for (const entry of cases) {
    const expected = expectedById.get(entry.id);
    if (!expected) fail(`pt-radiometric-ab: unexpected case id ${entry.id}`);
    if (entry.script !== expected.scriptPath) {
      fail(`pt-radiometric-ab: ${entry.id} script ${entry.script} differs from proofs.mjs`);
    }
    if (entry.resultFile !== expected.resultPath) {
      fail(`pt-radiometric-ab: ${entry.id} resultFile ${entry.resultFile} differs from proofs.mjs`);
    }
  }
  const blockedCases = cases.filter((entry) => entry.status === "HOST-BLOCKED");
  const failedCases = cases.filter((entry) => entry.status === "FAIL");
  if (failedCases.length > 0) {
    fail(`pt-radiometric-ab: committed status contains FAIL case(s): ${failedCases.map((entry) => entry.id).join(", ")}`);
  }
  if (status.verdict === "HOST-BLOCKED" || status.verdict === "PASS-PARTIAL") {
    if (blockedCases.length === 0) fail("pt-radiometric-ab: blocked/partial status must include blocked cases");
    for (const entry of blockedCases) {
      if (!proof.blockedReasonCodes.includes(entry.reason?.code)) {
        fail(`pt-radiometric-ab: blocked reason code ${entry.reason?.code} is not allowed`);
      }
    }
    const nextSteps = /** @type {unknown[]} */ (status.nextSteps ?? []);
    if (!nextSteps.some((step) => String(step).includes("Do not promote"))) {
      fail("pt-radiometric-ab: host-blocked status must preserve the do-not-promote warning");
    }
  }
  if (status.verdict === "PASS" && cases.some((entry) => entry.status !== "PASS")) {
    fail("pt-radiometric-ab: PASS status requires every case to pass");
  }
  if (status.verdict === "PASS" && status.reason?.code !== "pt-radiometric-ab-complete") {
    fail(`pt-radiometric-ab: PASS status must carry pt-radiometric-ab-complete, got ${status.reason?.code}`);
  }
}

/** @param {any} proof */
async function checkPtRadiometricPromotionStatus(proof) {
  const status = JSON.parse(await Deno.readTextFile(new URL(`../../${proof.statusPath}`, import.meta.url)));
  if (status.harness !== proof.harness) fail("pt-radiometric-promotion: harness mismatch");
  if (status.verdict !== proof.verdict) fail("pt-radiometric-promotion: verdict must stay PASS-PARTIAL");
  if (!sameJson(status.sourceStatuses, proof.sourceStatuses)) {
    fail("pt-radiometric-promotion: sourceStatuses mismatch");
  }

  const host = JSON.parse(await Deno.readTextFile(new URL(`../../${proof.hostStatusPath}`, import.meta.url)));
  const selectedCases = host.selectedCases ?? [];
  if (status.hostStatus?.verdict !== host.verdict) fail("pt-radiometric-promotion: host verdict mismatch");
  if (status.hostStatus?.caseCount !== selectedCases.length) fail("pt-radiometric-promotion: host caseCount mismatch");
  if (!sameJson(status.hostStatus?.selectedCases, selectedCases)) {
    fail("pt-radiometric-promotion: selectedCases mismatch");
  }
  if (host.verdict !== "PASS") fail("pt-radiometric-promotion: host recapture must remain PASS");

  const sppm = JSON.parse(await Deno.readTextFile(new URL(`../../${proof.safeDefaultProofs.sppm.resultPath}`, import.meta.url)));
  const bdpt = JSON.parse(await Deno.readTextFile(new URL(`../../${proof.safeDefaultProofs.bdptEndpointOnly.resultPath}`, import.meta.url)));
  const restirPt = JSON.parse(await Deno.readTextFile(new URL(`../../${proof.safeDefaultProofs.restirPtDiffuse.resultPath}`, import.meta.url)));
  const specialty = JSON.parse(await Deno.readTextFile(new URL(`../../${proof.safeDefaultProofs.restirPtSpecialty.resultPath}`, import.meta.url)));
  const glossyResearch = JSON.parse(await Deno.readTextFile(new URL(`../../${proof.researchFindings.restirPtGlossyResearch.resultPath}`, import.meta.url)));
  const sobol = JSON.parse(await Deno.readTextFile(new URL(`../../${proof.researchFindings.sobolDefault.resultPath}`, import.meta.url)));

  const finalSppm = sppm.sppm?.[sppm.sppm.length - 1];
  const endpoint = /** @type {any[]} */ (bdpt.controls?.byMaxLightBounces ?? []).find((entry) =>
    entry.maxLightBounces === proof.safeDefaultProofs.bdptEndpointOnly.maxLightBounces
  );
  const firstBdptFinding = /** @type {any[]} */ (bdpt.controls?.byMaxLightBounces ?? []).find((entry) =>
    entry.maxLightBounces === proof.researchFindings.bdptMultiVertex.firstFindingMaxLightBounces
  );
  const expectedSafeDefaults = {
    sppm: {
      verdict: sppm.verdict,
      converging: sppm.converging,
      inBallpark: sppm.inBallpark,
      finalRelErr: finalSppm?.relErr,
    },
    bdptEndpointOnly: {
      verdict: bdpt.verdict,
      endpointOnlyMatchesUni: bdpt.controls?.endpointOnlyMatchesUni,
      maxLightBounces: endpoint?.maxLightBounces,
      globalRelErr: endpoint?.globalRelErr,
      roiRelErr: endpoint?.roiRelErr,
    },
    restirPtDiffuse: {
      verdict: restirPt.verdict,
      meanAgreement: restirPt.meanAgreement,
      varianceNotWorse: restirPt.varianceNotWorse,
      globalRelErr: restirPt.globalRelErr,
      varRatio: restirPt.varRatio,
    },
    restirPtSpecialty: {
      mode: specialty.mode,
      caseCount: specialty.summary?.caseCount,
      maxAbsoluteError: specialty.summary?.maxAbsoluteError,
      maxRelativeError: specialty.summary?.maxRelativeError,
    },
  };
  if (!sameJson(status.safeDefaultProofs, expectedSafeDefaults)) {
    fail("pt-radiometric-promotion: safeDefaultProofs do not match committed result snapshots");
  }

  const sobolRatios = /** @type {any[]} */ (sobol.scenes ?? []).map((scene) => scene.ratios ?? {});
  const sobolFinding = sobol.researchFindings?.sobolDefault ?? null;
  if (sobolFinding == null || typeof sobolFinding !== "object") {
    fail("pt-radiometric-promotion: Sobol result must carry researchFindings.sobolDefault");
  }
  for (const [key, expected] of Object.entries({
    defaultReady: false,
    evidenceClass: proof.researchFindings.sobolDefault.evidenceClass,
    requiredEvidence: proof.researchFindings.sobolDefault.requiredEvidence,
    evidencePath: proof.researchFindings.sobolDefault.resultPath,
  })) {
    if (sobolFinding[key] !== expected) {
      fail(`pt-radiometric-promotion: Sobol research finding ${key} differs from proof metadata`);
    }
  }
  for (const [key, expected] of Object.entries({
    maxGlobalRmseRatio: Math.max(...sobolRatios.map((ratio) => ratio.globalRmse)),
    maxRoiRmseRatio: Math.max(...sobolRatios.map((ratio) => ratio.roiRmse)),
    maxElapsedMsRatio: Math.max(...sobolRatios.map((ratio) => ratio.elapsedMs)),
  })) {
    assertFiniteNumber(sobolFinding[key], `pt-radiometric-promotion: Sobol research finding ${key}`);
    if (sobolFinding[key] !== expected) {
      fail(`pt-radiometric-promotion: Sobol research finding ${key} does not match result ratios`);
    }
  }
  const glossyFinding = glossyResearch.researchFindings?.restirPtGlossyResearch ?? null;
  if (glossyFinding == null || typeof glossyFinding !== "object") {
    fail("pt-radiometric-promotion: glossy research result must carry researchFindings.restirPtGlossyResearch");
  }
  const expectedResearchFindings = {
    bdptMultiVertex: {
      defaultReady: bdpt.controls?.multiVertexPromotion?.defaultReady,
      warningCode: proof.researchFindings.bdptMultiVertex.warningCode,
      blocker: bdpt.controls?.multiVertexPromotion?.blocker,
      requiredEstimator: bdpt.controls?.multiVertexPromotion?.requiredEstimator,
      firstFindingMaxLightBounces: firstBdptFinding?.maxLightBounces,
      firstFindingGlobalRelErr: firstBdptFinding?.globalRelErr,
      evidencePath: proof.researchFindings.bdptMultiVertex.resultPath,
    },
    restirPtGlossyResearch: {
      verdict: glossyFinding.verdict,
      defaultReady: glossyFinding.defaultReady,
      warningCode: glossyFinding.warningCode,
      blocker: glossyFinding.blocker,
      requiredEvidence: glossyFinding.requiredEvidence,
      globalRelErr: glossyFinding.globalRelErr,
      varRatio: glossyFinding.varRatio,
      evidencePath: glossyFinding.evidencePath,
    },
    sobolDefault: {
      defaultReady: sobolFinding.defaultReady,
      evidenceClass: sobolFinding.evidenceClass,
      requiredEvidence: sobolFinding.requiredEvidence,
      maxGlobalRmseRatio: sobolFinding.maxGlobalRmseRatio,
      maxRoiRmseRatio: sobolFinding.maxRoiRmseRatio,
      maxElapsedMsRatio: sobolFinding.maxElapsedMsRatio,
      evidencePath: sobolFinding.evidencePath,
    },
  };
  if (!sameJson(status.researchFindings, expectedResearchFindings)) {
    fail("pt-radiometric-promotion: researchFindings do not match committed result snapshots");
  }
  if (
    status.researchFindings.bdptMultiVertex.defaultReady !== false ||
    status.researchFindings.restirPtGlossyResearch.defaultReady !== false ||
    status.researchFindings.sobolDefault.defaultReady !== false
  ) {
    fail("pt-radiometric-promotion: research/default promotion blockers must remain explicit");
  }
}

/** @param {any} proof */
async function checkWalkaroundHostStatus(proof) {
  const statusUrl = new URL(`../../${proof.statusPath}`, import.meta.url);
  const status = JSON.parse(await Deno.readTextFile(statusUrl));
  if (status.harness !== proof.harness) fail(`walkaround-ab: harness ${status.harness} differs from proofs.mjs`);
  if (!proof.allowedVerdicts.includes(status.verdict)) {
    fail(`walkaround-ab: verdict ${status.verdict} is outside ${proof.allowedVerdicts.join(", ")}`);
  }
  const resultFile = status.preservedResultFile ?? status.resultFile;
  if (resultFile !== proof.preservedResultFile) {
    fail("walkaround-ab: result file differs from proofs.mjs");
  }
  const preservedUrl = new URL(`../../${resultFile}`, import.meta.url);
  const preservedStat = await Deno.stat(preservedUrl);
  if (!preservedStat.isFile || preservedStat.size <= 2) fail("walkaround-ab: preserved result file is missing or empty");
  const preservedResult = JSON.parse(await Deno.readTextFile(preservedUrl));
  if (status.verdict === "HOST-BLOCKED") {
    if (!proof.blockedReasonCodes.includes(status.reason?.code)) {
      fail(`walkaround-ab: blocked reason code ${status.reason?.code} is not allowed`);
    }
    const nextSteps = /** @type {unknown[]} */ (status.nextSteps ?? []);
    if (!nextSteps.some((step) => String(step).includes("Do not promote"))) {
      fail("walkaround-ab: HOST-BLOCKED status must preserve the do-not-promote warning");
    }
    return;
  }
  assertWalkaroundFullFreshStatus(status, preservedResult);
  if (status.verdict === "PASS-PARTIAL") {
    if (status.reason?.code !== proof.partialReasonCode) {
      fail(`walkaround-ab: partial reason code ${status.reason?.code} differs from proofs.mjs`);
    }
    /** @type {any[]} */
    const nextSteps = status.nextSteps ?? [];
    if (!nextSteps.some((step) => String(step).includes("Do not promote"))) {
      fail("walkaround-ab: PASS-PARTIAL status must preserve the do-not-promote warning");
    }
    return;
  }
  if (status.verdict === "PASS" && status.reason?.code !== "walkaround-ab-complete") {
    fail(`walkaround-ab: PASS status must carry walkaround-ab-complete, got ${status.reason?.code}`);
  }
}

/** @param {any} proof */
async function checkWalkaroundGlossySpp64Status(proof) {
  const statusUrl = new URL(`../../${proof.statusPath}`, import.meta.url);
  const status = JSON.parse(await Deno.readTextFile(statusUrl));
  if (status.harness !== proof.harness) fail("walkaround glossy-spp64: harness mismatch");
  if (!proof.allowedVerdicts.includes(status.verdict)) {
    fail(`walkaround glossy-spp64: verdict ${status.verdict} is outside ${proof.allowedVerdicts.join(", ")}`);
  }
  if (status.selectedCases !== proof.selectedCases) {
    fail(`walkaround glossy-spp64: selectedCases ${status.selectedCases} differs from ${proof.selectedCases}`);
  }
  const resultFile = status.preservedResultFile ?? status.resultFile;
  if (resultFile !== proof.preservedResultFile) {
    fail("walkaround glossy-spp64: result file differs from proofs.mjs");
  }
  if (!sameJson(status.renderConfig, proof.expectedRenderConfig)) {
    fail(
      `walkaround glossy-spp64: renderConfig ${JSON.stringify(status.renderConfig)} ` +
      `differs from ${JSON.stringify(proof.expectedRenderConfig)}`,
    );
  }
  if (status.verdict === "HOST-BLOCKED") {
    if (!proof.blockedReasonCodes.includes(status.reason?.code)) {
      fail(`walkaround glossy-spp64: blocked reason code ${status.reason?.code} is not allowed`);
    }
    const nextSteps = status.nextSteps ?? [];
    if (!nextSteps.some((/** @type {unknown} */ step) => String(step).includes(proof.doNotPromoteText))) {
      fail("walkaround glossy-spp64: HOST-BLOCKED status must preserve the do-not-promote warning");
    }
    return;
  }
  if (status.verdict === "PASS-PARTIAL" && status.reason?.code !== proof.partialReasonCode) {
    fail(`walkaround glossy-spp64: partial reason code ${status.reason?.code} differs from proofs.mjs`);
  }
  const resultUrl = new URL(`../../${resultFile}`, import.meta.url);
  const result = JSON.parse(await Deno.readTextFile(resultUrl));
  const glossy = result.glossy;
  if (glossy?.qualityProfile !== proof.expectedRenderConfig.qualityProfile) {
    fail("walkaround glossy-spp64: glossy result qualityProfile mismatch");
  }
  if (!sameJson(glossy?.renderConfig, {
    width: Number(proof.expectedRenderConfig.width),
    height: Number(proof.expectedRenderConfig.height),
    spp: Number(proof.expectedRenderConfig.spp),
  })) {
    fail("walkaround glossy-spp64: glossy result renderConfig mismatch");
  }
  checkWalkaroundGlossy({ ...WALKAROUND_AB_RESULT_PROOF.cases.glossy, expectedVerdict: null }, {
    ...glossy,
    resolution: WALKAROUND_AB_RESULT_PROOF.resolution,
    spp: WALKAROUND_AB_RESULT_PROOF.spp,
  });
}

/** @param {any} proof */
async function checkWalkaroundAllSpp64Status(proof) {
  const statusUrl = new URL(`../../${proof.statusPath}`, import.meta.url);
  const status = JSON.parse(await Deno.readTextFile(statusUrl));
  if (status.harness !== proof.harness) fail("walkaround all-spp64: harness mismatch");
  if (!proof.allowedVerdicts.includes(status.verdict)) {
    fail(`walkaround all-spp64: verdict ${status.verdict} is outside ${proof.allowedVerdicts.join(", ")}`);
  }
  if ((status.selectedCases ?? null) !== proof.selectedCases) {
    fail(`walkaround all-spp64: selectedCases ${status.selectedCases} differs from ${proof.selectedCases}`);
  }
  const resultFile = status.preservedResultFile ?? status.resultFile;
  if (resultFile !== proof.preservedResultFile) {
    fail("walkaround all-spp64: result file differs from proofs.mjs");
  }
  if (!sameJson(status.renderConfig, proof.expectedRenderConfig)) {
    fail(
      `walkaround all-spp64: renderConfig ${JSON.stringify(status.renderConfig)} ` +
      `differs from ${JSON.stringify(proof.expectedRenderConfig)}`,
    );
  }
  if (status.verdict === "HOST-BLOCKED") {
    if (!proof.blockedReasonCodes.includes(status.reason?.code)) {
      fail(`walkaround all-spp64: blocked reason code ${status.reason?.code} is not allowed`);
    }
    const nextSteps = status.nextSteps ?? [];
    if (!nextSteps.some((/** @type {unknown} */ step) => String(step).includes(proof.doNotPromoteText))) {
      fail("walkaround all-spp64: HOST-BLOCKED status must preserve the do-not-promote warning");
    }
    return;
  }
  assertWalkaroundFullFreshStatus(status, JSON.parse(await Deno.readTextFile(new URL(`../../${resultFile}`, import.meta.url))));
  if (status.verdict === "PASS-PARTIAL" && status.reason?.code !== proof.partialReasonCode) {
    fail(`walkaround all-spp64: partial reason code ${status.reason?.code} differs from proofs.mjs`);
  }
  const resultUrl = new URL(`../../${resultFile}`, import.meta.url);
  const result = JSON.parse(await Deno.readTextFile(resultUrl));
  const resolution = `${proof.expectedRenderConfig.width}x${proof.expectedRenderConfig.height}`;
  const spp = Number(proof.expectedRenderConfig.spp);
  const highSppCases = {
    a8: { ...WALKAROUND_AB_RESULT_PROOF.cases.a8, resolution, spp, expectedVerdict: null },
    sun: { ...WALKAROUND_AB_RESULT_PROOF.cases.sun, resolution, spp, expectedVerdict: null },
    glass: { ...WALKAROUND_AB_RESULT_PROOF.cases.glass, resolution, spp, expectedVerdict: null },
    glossy: { ...WALKAROUND_AB_RESULT_PROOF.cases.glossy, resolution, spp, expectedVerdict: null },
  };
  checkWalkaroundA8(highSppCases.a8, result.a8);
  checkWalkaroundSun(highSppCases.sun, result.sun);
  checkWalkaroundGlass(highSppCases.glass, result.glass);
  checkWalkaroundGlossy(highSppCases.glossy, result.glossy);
}

/**
 * @param {any} status
 * @param {Record<string, any>} preservedResult
 */
function assertWalkaroundFullFreshStatus(status, preservedResult) {
  const selected = status.selectedCases;
  if (selected != null && selected !== "") {
    const selectedCases = String(selected).split(",").map((part) => part.trim().toLowerCase()).filter(Boolean).sort();
    if (!sameJson(selectedCases, [...WALKAROUND_AB_CASE_IDS].sort())) {
      fail(
        `walkaround-ab: status selectedCases=${JSON.stringify(selected)} was a subset run; ` +
        "full proof status must refresh a8,sun,glass,glossy together",
      );
    }
  }
  const resultCaseIds = Object.keys(preservedResult).sort();
  if (!sameJson(resultCaseIds, [...WALKAROUND_AB_CASE_IDS].sort())) {
    fail(`walkaround-ab: preserved result cases ${JSON.stringify(resultCaseIds)} differ from required full case set`);
  }
  const expectedVerdicts = Object.fromEntries(
    WALKAROUND_AB_CASE_IDS.map((id) => [id, preservedResult[id]?.verdict ?? "UNKNOWN"]),
  );
  if (!sameJson(status.caseVerdicts, expectedVerdicts)) {
    fail(
      `walkaround-ab: status caseVerdicts ${JSON.stringify(status.caseVerdicts)} ` +
      `must match preserved results ${JSON.stringify(expectedVerdicts)}`,
    );
  }
}

/**
 * @param {string} label
 * @param {any} proof
 * @param {any} result
 */
function checkWalkaroundCaseCommon(label, proof, result) {
  const expectedResolution = proof.resolution ?? WALKAROUND_AB_RESULT_PROOF.resolution;
  const expectedSpp = proof.spp ?? WALKAROUND_AB_RESULT_PROOF.spp;
  if (result?.id !== proof.id) fail(`walkaround-ab ${label}: id ${result?.id} differs from ${proof.id}`);
  if (result?.resolution !== expectedResolution) {
    fail(`walkaround-ab ${label}: resolution ${result?.resolution} differs from ${expectedResolution}`);
  }
  if (result?.spp !== expectedSpp) {
    fail(`walkaround-ab ${label}: spp ${result?.spp} differs from ${expectedSpp}`);
  }
  if (!proof.allowedVerdicts.includes(result?.verdict)) {
    fail(`walkaround-ab ${label}: verdict ${result?.verdict} is outside ${proof.allowedVerdicts.join(", ")}`);
  }
  if (proof.expectedVerdict != null && result?.verdict !== proof.expectedVerdict) {
    fail(`walkaround-ab ${label}: verdict ${result?.verdict} differs from expected ${proof.expectedVerdict}`);
  }
}

/**
 * @param {any} proof
 * @param {any} result
 */
function checkWalkaroundA8(proof, result) {
  checkWalkaroundCaseCommon("A8", proof, result);
  assertFiniteNumber(result.delta?.overall, "walkaround-ab A8: delta.overall");
  assertFiniteNumber(result.biased?.overall, "walkaround-ab A8: biased.overall");
  if (Math.abs(result.delta.overall) > proof.maxAbsOverallDelta) {
    fail(`walkaround-ab A8: |overall delta| ${Math.abs(result.delta.overall)} exceeds ${proof.maxAbsOverallDelta}`);
  }
  const relativeOverallDelta = Math.abs(result.delta.overall) / Math.max(Math.abs(result.biased.overall), 1e-8);
  if (relativeOverallDelta > proof.maxRelativeOverallDelta) {
    fail(
      `walkaround-ab A8: relative overall delta ${relativeOverallDelta} exceeds ${proof.maxRelativeOverallDelta}`,
    );
  }
  for (const group of ["biased", "unbiased"]) {
    for (const key of ["overall", "floor", "ceiling", "leftWall", "rightWall"]) {
      assertFiniteNumber(result[group]?.[key], `walkaround-ab A8: ${group}.${key}`);
    }
  }
  for (const key of ["floor", "ceiling", "leftWall", "rightWall"]) {
    assertFiniteNumber(result.delta?.[key], `walkaround-ab A8: delta.${key}`);
    if (Math.abs(result.delta[key]) > proof.maxAbsRegionDelta) {
      fail(`walkaround-ab A8: |delta.${key}| ${Math.abs(result.delta[key])} exceeds ${proof.maxAbsRegionDelta}`);
    }
  }
}

/**
 * @param {any} proof
 * @param {any} result
 */
function checkWalkaroundSun(proof, result) {
  checkWalkaroundCaseCommon("SUN", proof, result);
  const analyticRatio = result.receiverRatioToAnalytic ?? result.floorRatioToAnalytic;
  assertFiniteNumber(analyticRatio, "walkaround-ab SUN: receiverRatioToAnalytic");
  if (result.verdict === "PASS" && result.analyticAgreement !== true) {
    fail("walkaround-ab SUN: PASS requires analyticAgreement=true");
  }
  if (result.verdict === "PASS" && Math.abs(analyticRatio - 1) > proof.maxAnalyticRatioError) {
    fail(`walkaround-ab SUN: analytic ratio ${analyticRatio} is outside ±${proof.maxAnalyticRatioError}`);
  }
  const receiverLum = result.rendered?.receiverLum ?? result.rendered?.floorLum;
  assertFiniteNumber(receiverLum, "walkaround-ab SUN: rendered.receiverLum");
  if (result.rendered?.sideDiagnosticLum != null) {
    assertFiniteNumber(result.rendered.sideDiagnosticLum, "walkaround-ab SUN: rendered.sideDiagnosticLum");
  }
  if (result.shadowAssertionAuthored === true && result.shadowCorrect !== true) {
    fail("walkaround-ab SUN: authored shadow assertion requires shadowCorrect=true");
  }
}

/**
 * @param {any} proof
 * @param {any} result
 */
function checkWalkaroundGlass(proof, result) {
  checkWalkaroundCaseCommon("GLASS", proof, result);
  assertFiniteNumber(result.centreRatio, "walkaround-ab GLASS: centreRatio");
  assertFiniteNumber(result.overallRatio, "walkaround-ab GLASS: overallRatio");
  if (result.centreRatio < proof.minCentreRatio) {
    fail(`walkaround-ab GLASS: centreRatio ${result.centreRatio} is below ${proof.minCentreRatio}`);
  }
  const withinPromotionBounds =
    result.centreRatio <= proof.maxCentreRatio &&
    result.overallRatio <= proof.maxOverallRatio;
  if (result.ratioWithinPromotionBounds !== withinPromotionBounds) {
    fail("walkaround-ab GLASS: ratioWithinPromotionBounds does not match committed ratios");
  }
  assertFiniteNumber(result.delta?.centreRegionLum, "walkaround-ab GLASS: delta.centreRegionLum");
  assertFiniteNumber(result.delta?.overall, "walkaround-ab GLASS: delta.overall");
  const signal = Math.max(Math.abs(result.delta.centreRegionLum), Math.abs(result.delta.overall));
  if (result.materialEffectObserved !== (signal >= proof.minSignalDeltaForPass)) {
    fail("walkaround-ab GLASS: materialEffectObserved does not match committed deltas");
  }
  if (result.verdict === "PASS" && !withinPromotionBounds) {
    fail(
      `walkaround-ab GLASS: PASS requires bounded ratios; centre=${result.centreRatio}, ` +
      `overall=${result.overallRatio}`,
    );
  }
  if (result.verdict === "PASS" && signal < proof.minSignalDeltaForPass) {
    fail(
      `walkaround-ab GLASS: PASS requires observed material effect; max delta ${signal} ` +
      `is below ${proof.minSignalDeltaForPass}`,
    );
  }
  if (result.verdict === "FINDING") {
    if (signal < proof.minSignalDeltaForPass) {
      fail(
        `walkaround-ab GLASS: FINDING requires an observed material delta; max delta ${signal} ` +
        `is below ${proof.minSignalDeltaForPass}`,
      );
    }
    if (withinPromotionBounds) {
      fail("walkaround-ab GLASS: FINDING requires an out-of-bounds promotion ratio");
    }
    if (result.materialEffectObserved !== true) {
      fail("walkaround-ab GLASS: FINDING requires materialEffectObserved=true");
    }
    if (result.promotion?.defaultReady !== proof.promotion?.defaultReady) {
      fail("walkaround-ab GLASS: FINDING must carry promotion.defaultReady=false");
    }
    if (result.promotion?.blocker !== proof.promotion?.blocker) {
      fail(
        `walkaround-ab GLASS: blocker ${result.promotion?.blocker} ` +
        `differs from ${proof.promotion?.blocker}`,
      );
    }
    if (result.promotion?.requiredEvidence !== proof.promotion?.requiredEvidence) {
      fail(
        `walkaround-ab GLASS: requiredEvidence ${result.promotion?.requiredEvidence} ` +
        `differs from ${proof.promotion?.requiredEvidence}`,
      );
    }
  }
  if (result.verdict === "SMOKE") {
    if (result.materialEffectObserved !== false) {
      fail("walkaround-ab GLASS: SMOKE must not masquerade as observed material transport");
    }
    if (signal >= proof.minSignalDeltaForPass) {
      fail(
        `walkaround-ab GLASS: SMOKE should be promoted to PASS/FINDING once signal ${signal} ` +
        `exceeds ${proof.minSignalDeltaForPass}`,
      );
    }
  }
}

/**
 * @param {any} proof
 * @param {any} result
 */
function checkWalkaroundGlossy(proof, result) {
  checkWalkaroundCaseCommon("GLOSSY", proof, result);
  if (result.sampleRegion !== proof.sampleRegion) {
    fail(`walkaround-ab GLOSSY: sampleRegion ${result.sampleRegion} differs from ${proof.sampleRegion}`);
  }
  const sampleRatio = result.sampleRatio ?? result.floorRatio;
  assertFiniteNumber(sampleRatio, "walkaround-ab GLOSSY: sampleRatio");
  if (result.verdict !== "FINDING" && sampleRatio < proof.minSampleRatio) {
    fail(`walkaround-ab GLOSSY: sampleRatio ${sampleRatio} is below ${proof.minSampleRatio}`);
  }
  const sampleDelta = result.delta?.sampleRegionLum ?? result.delta?.floorLum;
  assertFiniteNumber(sampleDelta, "walkaround-ab GLOSSY: delta.sampleRegionLum");
  assertFiniteNumber(result.delta?.overall, "walkaround-ab GLOSSY: delta.overall");
  const signal = Math.max(Math.abs(sampleDelta), Math.abs(result.delta.overall));
  if (result.materialEffectObserved !== (signal >= proof.minSignalDeltaForPass)) {
    fail("walkaround-ab GLOSSY: materialEffectObserved does not match committed deltas");
  }
  if (result.verdict === "PASS" && signal < proof.minSignalDeltaForPass) {
    fail(
      `walkaround-ab GLOSSY: PASS requires observed material effect; max delta ${signal} ` +
      `is below ${proof.minSignalDeltaForPass}`,
    );
  }
  if (result.verdict === "FINDING") {
    if (signal < proof.minSignalDeltaForPass) {
      fail(
        `walkaround-ab GLOSSY: FINDING requires an observed material delta; max delta ${signal} ` +
        `is below ${proof.minSignalDeltaForPass}`,
      );
    }
    if (result.materialEffectObserved !== true) {
      fail("walkaround-ab GLOSSY: FINDING requires materialEffectObserved=true");
    }
    if (result.promotion?.defaultReady !== proof.promotion?.defaultReady) {
      fail("walkaround-ab GLOSSY: FINDING must carry promotion.defaultReady=false");
    }
    if (result.promotion?.blocker !== proof.promotion?.blocker) {
      fail(
        `walkaround-ab GLOSSY: blocker ${result.promotion?.blocker} ` +
        `differs from ${proof.promotion?.blocker}`,
      );
    }
    if (result.promotion?.requiredEvidence !== proof.promotion?.requiredEvidence) {
      fail(
        `walkaround-ab GLOSSY: requiredEvidence ${result.promotion?.requiredEvidence} ` +
        `differs from ${proof.promotion?.requiredEvidence}`,
      );
    }
  }
}

/** @param {any} proof */
async function checkWalkaroundResults(proof) {
  const resultUrl = new URL(`../../${proof.resultPath}`, import.meta.url);
  const result = JSON.parse(await Deno.readTextFile(resultUrl));
  checkWalkaroundA8(proof.cases.a8, result.a8);
  checkWalkaroundSun(proof.cases.sun, result.sun);
  checkWalkaroundGlass(proof.cases.glass, result.glass);
  checkWalkaroundGlossy(proof.cases.glossy, result.glossy);
}

/** @param {any} proof */
async function checkWalkaroundPromotionStatus(proof) {
  const status = JSON.parse(await Deno.readTextFile(new URL(`../../${proof.statusPath}`, import.meta.url)));
  if (status.harness !== proof.harness) fail("walkaround promotion status: harness mismatch");
  if (status.verdict !== proof.verdict) {
    fail(`walkaround promotion status: verdict ${status.verdict} differs from ${proof.verdict}`);
  }
  if (status.promotion?.defaultReady !== proof.promotion.defaultReady) {
    fail("walkaround promotion status: promotion.defaultReady must stay false");
  }
  if (status.promotion?.classification !== proof.promotion.classification) {
    fail("walkaround promotion status: promotion.classification mismatch");
  }
  if (status.promotion?.blocker !== proof.promotion.blocker) {
    fail("walkaround promotion status: blocker mismatch");
  }
  if (!sameJson(status.promotion?.blockers, proof.promotion.blockers)) {
    fail("walkaround promotion status: blockers mismatch");
  }
  if (status.promotion?.requiredEvidence !== proof.promotion.requiredEvidence) {
    fail("walkaround promotion status: requiredEvidence mismatch");
  }
  if (!sameJson(status.sourceStatuses, proof.sourceStatuses)) {
    fail("walkaround promotion status: sourceStatuses mismatch");
  }

  const hostStatus = JSON.parse(await Deno.readTextFile(new URL(`../../${WALKAROUND_AB_HOST_STATUS_PROOF.statusPath}`, import.meta.url)));
  const hostResults = JSON.parse(await Deno.readTextFile(new URL(`../../${WALKAROUND_AB_RESULT_PROOF.resultPath}`, import.meta.url)));
  const allSpp64Status = JSON.parse(await Deno.readTextFile(new URL(`../../${WALKAROUND_ALL_SPP64_STATUS_PROOF.statusPath}`, import.meta.url)));
  const allSpp64Results = JSON.parse(await Deno.readTextFile(new URL(`../../${WALKAROUND_ALL_SPP64_STATUS_PROOF.preservedResultFile}`, import.meta.url)));
  if (hostStatus.verdict !== "PASS-PARTIAL" || hostStatus.reason?.code !== WALKAROUND_AB_HOST_STATUS_PROOF.partialReasonCode) {
    fail("walkaround promotion status: baseline host status no longer pins PASS-PARTIAL");
  }
  if (allSpp64Status.verdict !== "PASS-PARTIAL" || allSpp64Status.reason?.code !== WALKAROUND_ALL_SPP64_STATUS_PROOF.partialReasonCode) {
    fail("walkaround promotion status: all-spp64 status no longer pins PASS-PARTIAL");
  }
  const baselineVerdicts = Object.fromEntries(WALKAROUND_AB_CASE_IDS.map((id) => [id, hostResults[id]?.verdict]));
  const allSpp64Verdicts = Object.fromEntries(WALKAROUND_AB_CASE_IDS.map((id) => [id, allSpp64Results[id]?.verdict]));
  if (!sameJson(status.caseVerdicts, baselineVerdicts)) {
    fail("walkaround promotion status: caseVerdicts do not match baseline result snapshot");
  }
  if (!sameJson(status.highSppCaseVerdicts, allSpp64Verdicts)) {
    fail("walkaround promotion status: highSppCaseVerdicts do not match all-spp64 result snapshot");
  }
  if (baselineVerdicts.glossy !== "FINDING" || allSpp64Verdicts.glossy !== "FINDING") {
    fail("walkaround promotion status: glossy must remain a FINDING until promotion evidence lands");
  }
  if (baselineVerdicts.glass !== "PASS" || allSpp64Verdicts.glass !== "PASS") {
    fail("walkaround promotion status: glass must remain PASS in baseline and all-spp64 proofs");
  }

  const expectedGlassProfiles = [];
  for (const profile of proof.glassProfiles) {
    const result = JSON.parse(await Deno.readTextFile(new URL(`../../${profile.resultPath}`, import.meta.url)));
    const glass = result[profile.resultKey];
    expectedGlassProfiles.push({
      label: profile.label,
      resultPath: profile.resultPath,
      spp: profile.expectedSpp,
      qualityProfile: profile.expectedQualityProfile,
      verdict: glass?.verdict,
      centreRatio: glass?.centreRatio,
      overallRatio: glass?.overallRatio,
      ratioWithinPromotionBounds: glass?.ratioWithinPromotionBounds,
      materialEffectObserved: glass?.materialEffectObserved,
    });
    if (glass?.verdict !== "PASS" || glass?.ratioWithinPromotionBounds !== true || glass?.materialEffectObserved !== true) {
      fail(`walkaround promotion status: ${profile.label} glass must remain bounded PASS evidence`);
    }
    if (glass?.promotion != null) {
      fail(`walkaround promotion status: ${profile.label} glass PASS must not carry stale promotion blockers`);
    }
  }
  if (!sameJson(status.glassProfiles, expectedGlassProfiles)) {
    fail("walkaround promotion status: glassProfiles do not match committed result snapshots");
  }

  const expectedProfiles = [];
  for (const profile of proof.glossyProfiles) {
    const result = JSON.parse(await Deno.readTextFile(new URL(`../../${profile.resultPath}`, import.meta.url)));
    const glossy = result[profile.resultKey];
    expectedProfiles.push({
      label: profile.label,
      resultPath: profile.resultPath,
      spp: profile.expectedSpp,
      qualityProfile: profile.expectedQualityProfile,
      verdict: glossy?.verdict,
      sampleRatio: glossy?.sampleRatio ?? glossy?.floorRatio,
      materialEffectObserved: glossy?.materialEffectObserved,
    });
    if (glossy?.promotion?.defaultReady !== false || glossy?.promotion?.blocker !== WALKAROUND_AB_RESULT_PROOF.cases.glossy.promotion.blocker) {
      fail(`walkaround promotion status: ${profile.label} glossy promotion metadata drifted`);
    }
  }
  if (!sameJson(status.glossyProfiles, expectedProfiles)) {
    fail("walkaround promotion status: glossyProfiles do not match committed result snapshots");
  }
}

assertRequiredRadiometricRows();
for (const proof of RADIOMETRIC_AB_PROOFS) {
  const resultUrl = new URL(`../../${proof.resultPath}`, import.meta.url);
  const result = JSON.parse(await Deno.readTextFile(resultUrl));
  await assertCommon(proof, result);
  if (proof.id === "sppm") checkSppm(proof, result);
  else if (proof.id === "bdpt") checkBdpt(proof, result);
  else if (proof.id === "restir-pt") checkRestirPt(proof, result);
  else if (proof.id === "sobol") checkSobol(proof, result);
  else fail(`unknown proof id ${proof.id}`);
}

await checkBdptMultiVertexResearch(BDPT_MULTIVERTEX_RESEARCH_PROOF);
await checkRestirPtSpecialty(RESTIR_PT_SPECIALTY_PROOF);
await checkRestirPtGlossyResearch(RESTIR_PT_GLOSSY_RESEARCH_PROOF);
await checkPtRadiometricHostStatus(PT_RADIOMETRIC_AB_HOST_STATUS_PROOF);
await checkPtRadiometricPromotionStatus(PT_RADIOMETRIC_PROMOTION_STATUS_PROOF);
await checkWalkaroundHostStatus(WALKAROUND_AB_HOST_STATUS_PROOF);
await checkWalkaroundResults(WALKAROUND_AB_RESULT_PROOF);
await checkWalkaroundGlossySpp64Status(WALKAROUND_GLOSSY_SPP64_STATUS_PROOF);
await checkWalkaroundAllSpp64Status(WALKAROUND_ALL_SPP64_STATUS_PROOF);
await checkWalkaroundPromotionStatus(WALKAROUND_AB_PROMOTION_STATUS_PROOF);

console.log(`[radiometric-ab-proof-check] PASS (${RADIOMETRIC_AB_PROOFS.length} committed radiometric A/B result snapshots, 1 BDPT multi-vertex research guard, 1 ReSTIR-PT specialty fixture, 1 glossy research artifact, pt host status, 1 pt radiometric promotion boundary status, walkaround host status, 4 walkaround A/B cases, 1 high-SPP glossy walkaround status, 1 high-SPP all-cases walkaround status, 1 walkaround promotion boundary status)`);
