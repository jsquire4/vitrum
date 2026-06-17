#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
// Verifies that committed radiometric A/B result JSONs still satisfy their proof metadata.

import {
  RADIOMETRIC_AB_PROOFS,
  RESTIR_PT_SPECIALTY_PROOF,
  WALKAROUND_AB_HOST_STATUS_PROOF,
} from "./proofs.mjs";

function fail(message) {
  throw new Error(`[radiometric-ab-proof-check] ${message}`);
}

function sameJson(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function assertFiniteNumber(value, label) {
  if (typeof value !== "number" || !Number.isFinite(value)) fail(`${label} must be a finite number`);
}

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

function checkSppm(proof, result) {
  if (result.reference?.strategy !== proof.reference.strategy) fail("sppm: reference strategy differs from proofs.mjs");
  if (result.reference?.frames !== proof.reference.frames) fail("sppm: reference frame count differs from proofs.mjs");
  if (!Array.isArray(result.sppm)) fail("sppm: result.sppm must be an array");
  const frames = result.sppm.map((entry) => entry.frames);
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
  const controls = result.controls?.byMaxLightBounces ?? [];
  const controlDepths = controls.map((entry) => entry.maxLightBounces);
  if (!sameJson(controlDepths, [1, 2, 3])) fail(`bdpt: control depths ${JSON.stringify(controlDepths)} differ from expected [1,2,3]`);
}

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

async function checkWalkaroundHostStatus(proof) {
  const statusUrl = new URL(`../../${proof.statusPath}`, import.meta.url);
  const status = JSON.parse(await Deno.readTextFile(statusUrl));
  if (status.harness !== proof.harness) fail(`walkaround-ab: harness ${status.harness} differs from proofs.mjs`);
  if (status.verdict !== proof.expectedVerdict) fail(`walkaround-ab: expected HOST-BLOCKED status, got ${status.verdict}`);
  if (status.reason?.code !== proof.reasonCode) fail(`walkaround-ab: reason code ${status.reason?.code} differs from proofs.mjs`);
  if (status.preservedResultFile !== proof.preservedResultFile) {
    fail("walkaround-ab: preservedResultFile differs from proofs.mjs");
  }
  const preservedUrl = new URL(`../../${status.preservedResultFile}`, import.meta.url);
  const preservedStat = await Deno.stat(preservedUrl);
  if (!preservedStat.isFile || preservedStat.size <= 2) fail("walkaround-ab: preserved result file is missing or empty");
  const nextSteps = status.nextSteps ?? [];
  if (!nextSteps.some((step) => String(step).includes("Do not promote"))) {
    fail("walkaround-ab: HOST-BLOCKED status must preserve the do-not-promote warning");
  }
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

console.log(`[radiometric-ab-proof-check] PASS (${RADIOMETRIC_AB_PROOFS.length} committed radiometric A/B result snapshots, 1 specialty fixture, 1 host-blocked walkaround status)`);
