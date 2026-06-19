#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
// Verifies that committed radiometric A/B result JSONs still satisfy their proof metadata.

import {
  RADIOMETRIC_AB_PROOFS,
  RESTIR_PT_SPECIALTY_PROOF,
  WALKAROUND_AB_HOST_STATUS_PROOF,
  WALKAROUND_AB_RESULT_PROOF,
} from "./proofs.mjs";

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

/** @param {any} proof */
async function checkRestirPtSpecialty(proof) {
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
  if (status.verdict === "HOST-BLOCKED") {
    if (!proof.blockedReasonCodes.includes(status.reason?.code)) {
      fail(`walkaround-ab: blocked reason code ${status.reason?.code} is not allowed`);
    }
    /** @type {any[]} */
    const nextSteps = status.nextSteps ?? [];
    if (!nextSteps.some((step) => String(step).includes("Do not promote"))) {
      fail("walkaround-ab: HOST-BLOCKED status must preserve the do-not-promote warning");
    }
    return;
  }
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

/**
 * @param {string} label
 * @param {any} proof
 * @param {any} result
 */
function checkWalkaroundCaseCommon(label, proof, result) {
  if (result?.id !== proof.id) fail(`walkaround-ab ${label}: id ${result?.id} differs from ${proof.id}`);
  if (result?.resolution !== WALKAROUND_AB_RESULT_PROOF.resolution) {
    fail(`walkaround-ab ${label}: resolution ${result?.resolution} differs from ${WALKAROUND_AB_RESULT_PROOF.resolution}`);
  }
  if (result?.spp !== WALKAROUND_AB_RESULT_PROOF.spp) {
    fail(`walkaround-ab ${label}: spp ${result?.spp} differs from ${WALKAROUND_AB_RESULT_PROOF.spp}`);
  }
  if (!proof.allowedVerdicts.includes(result?.verdict)) {
    fail(`walkaround-ab ${label}: verdict ${result?.verdict} is outside ${proof.allowedVerdicts.join(", ")}`);
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
  if (result.shadowCorrect !== true) fail("walkaround-ab SUN: shadowCorrect must be true");
  assertFiniteNumber(result.floorRatioToAnalytic, "walkaround-ab SUN: floorRatioToAnalytic");
  if (result.verdict === "PASS" && result.analyticAgreement !== true) {
    fail("walkaround-ab SUN: PASS requires analyticAgreement=true");
  }
  if (result.verdict === "PASS" && Math.abs(result.floorRatioToAnalytic - 1) > proof.maxAnalyticRatioError) {
    fail(`walkaround-ab SUN: analytic ratio ${result.floorRatioToAnalytic} is outside ±${proof.maxAnalyticRatioError}`);
  }
  assertFiniteNumber(result.rendered?.floorLum, "walkaround-ab SUN: rendered.floorLum");
  assertFiniteNumber(result.rendered?.leftWallLum, "walkaround-ab SUN: rendered.leftWallLum");
}

/**
 * @param {any} proof
 * @param {any} result
 */
function checkWalkaroundGlass(proof, result) {
  checkWalkaroundCaseCommon("GLASS", proof, result);
  assertFiniteNumber(result.centreRatio, "walkaround-ab GLASS: centreRatio");
  if (result.centreRatio < proof.minCentreRatio) {
    fail(`walkaround-ab GLASS: centreRatio ${result.centreRatio} is below ${proof.minCentreRatio}`);
  }
  assertFiniteNumber(result.delta?.centreRegionLum, "walkaround-ab GLASS: delta.centreRegionLum");
  assertFiniteNumber(result.delta?.overall, "walkaround-ab GLASS: delta.overall");
  const signal = Math.max(Math.abs(result.delta.centreRegionLum), Math.abs(result.delta.overall));
  if (result.materialEffectObserved !== (signal >= proof.minSignalDeltaForPass)) {
    fail("walkaround-ab GLASS: materialEffectObserved does not match committed deltas");
  }
  if (result.verdict === "PASS" && signal < proof.minSignalDeltaForPass) {
    fail(
      `walkaround-ab GLASS: PASS requires observed material effect; max delta ${signal} ` +
      `is below ${proof.minSignalDeltaForPass}`,
    );
  }
}

/**
 * @param {any} proof
 * @param {any} result
 */
function checkWalkaroundGlossy(proof, result) {
  checkWalkaroundCaseCommon("GLOSSY", proof, result);
  assertFiniteNumber(result.floorRatio, "walkaround-ab GLOSSY: floorRatio");
  if (result.verdict !== "FINDING" && result.floorRatio < proof.minFloorRatio) {
    fail(`walkaround-ab GLOSSY: floorRatio ${result.floorRatio} is below ${proof.minFloorRatio}`);
  }
  assertFiniteNumber(result.delta?.floorLum, "walkaround-ab GLOSSY: delta.floorLum");
  assertFiniteNumber(result.delta?.overall, "walkaround-ab GLOSSY: delta.overall");
  const signal = Math.max(Math.abs(result.delta.floorLum), Math.abs(result.delta.overall));
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

for (const proof of RADIOMETRIC_AB_PROOFS) {
  const resultUrl = new URL(`../../${proof.resultPath}`, import.meta.url);
  const result = JSON.parse(await Deno.readTextFile(resultUrl));
  await assertCommon(proof, result);
  if (proof.id === "sppm") checkSppm(proof, result);
  else if (proof.id === "bdpt") checkBdpt(proof, result);
  else if (proof.id === "restir-pt") checkRestirPt(proof, result);
  else fail(`unknown proof id ${proof.id}`);
}

await checkRestirPtSpecialty(RESTIR_PT_SPECIALTY_PROOF);
await checkWalkaroundHostStatus(WALKAROUND_AB_HOST_STATUS_PROOF);
await checkWalkaroundResults(WALKAROUND_AB_RESULT_PROOF);

console.log(`[radiometric-ab-proof-check] PASS (${RADIOMETRIC_AB_PROOFS.length} committed radiometric A/B result snapshots, 1 specialty fixture, walkaround host status, 4 walkaround A/B cases)`);
