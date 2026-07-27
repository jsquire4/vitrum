#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
// Verifies committed path-tracer numerical baselines against their regression
// contracts, then pins named test-source files and stable production-source
// contract text. It does not execute those tests. Historical captures are data, not source-maturity
// authority, so their pre-manifest provenance is accepted explicitly.

import {
  PT_LOCAL_ACCEPTANCE_PROOFS,
  RADIOMETRIC_AB_PROOFS,
  RESTIR_PT_SPECIALTY_PROOF,
} from "./proofs.mjs";
import {
  validateRadiometricResult,
  validateRestirPtSpecialtyResult,
} from "./resultValidation.mjs";

// The retained ReSTIR-PT GPU capture predates its exact seed/capture contract,
// so it is not a current gate. ReSTIR-PT is checked below through the deterministic
// specialty fixture plus named test-source and production-source text pins.
const REQUIRED_RADIOMETRIC_AB_ROWS = Object.freeze([
  ["sppm", "tools/radiometric-ab/ab-sppm.mjs", "tools/radiometric-ab/results-sppm.json"],
  ["bdpt", "tools/radiometric-ab/ab-bdpt.mjs", "tools/radiometric-ab/results-bdpt.json"],
  ["sobol", "tools/radiometric-ab/ab-sobol.mjs", "tools/radiometric-ab/results-sobol.json"],
]);
const REQUIRED_RADIOMETRIC_AB_IDS = new Set(
  REQUIRED_RADIOMETRIC_AB_ROWS.map(([id]) => id),
);
const SOURCE_ONLY = Deno.args.includes("--source-only");

/** @param {string} message @returns {never} */
function fail(message) {
  throw new Error(`[radiometric-ab-proof-check] ${message}`);
}

function assertRequiredRadiometricRows() {
  const byId = new Map();
  for (const proof of RADIOMETRIC_AB_PROOFS) {
    if (byId.has(proof.id)) fail(`duplicate radiometric proof id ${proof.id}`);
    byId.set(proof.id, proof);
  }
  for (const [id, scriptPath, resultPath] of REQUIRED_RADIOMETRIC_AB_ROWS) {
    const proof = byId.get(id);
    if (proof == null) fail(`missing required radiometric proof row ${id}`);
    if (proof.scriptPath !== scriptPath || proof.resultPath !== resultPath) {
      fail(`${id}: script/result paths differ from the stable proof contract`);
    }
  }
}

/** @param {any} proof */
async function checkRadiometricResult(proof) {
  const scriptUrl = new URL(`../../${proof.scriptPath}`, import.meta.url);
  const scriptStat = await Deno.stat(scriptUrl);
  if (!scriptStat.isFile) fail(`${proof.id}: script path is missing`);

  const result = JSON.parse(
    await Deno.readTextFile(new URL(`../../${proof.resultPath}`, import.meta.url)),
  );
  validateRadiometricResult(proof, result, { historicalBaseline: true });
}

/** @param {any} proof */
async function checkRestirPtSpecialty(proof) {
  const scriptStat = await Deno.stat(
    new URL(`../../${proof.scriptPath}`, import.meta.url),
  );
  if (!scriptStat.isFile) fail("restir-pt-specialty: script path is missing");
  const result = JSON.parse(
    await Deno.readTextFile(new URL(`../../${proof.resultPath}`, import.meta.url)),
  );
  validateRestirPtSpecialtyResult(proof, result);
}

/** @param {Record<string, any>} proofs */
async function checkPtSourcePins(proofs) {
  for (const [feature, proof] of Object.entries(proofs)) {
    for (const path of proof.paths) {
      const source = await Deno.readTextFile(
        new URL(`../../${path}`, import.meta.url),
      );
      if (!source.includes("describe(") && !source.includes("it(")) {
        fail(`${feature}: named test-source file ${path} has no test declaration`);
      }
    }
    const source = await Deno.readTextFile(
      new URL(`../../${proof.sourcePath}`, import.meta.url),
    );
    for (const needle of proof.needles) {
      if (!source.includes(needle)) {
        fail(`${feature}: ${proof.sourcePath} is missing stable contract needle ${needle}`);
      }
    }
  }
}

if (!SOURCE_ONLY) {
  assertRequiredRadiometricRows();
  for (const proof of RADIOMETRIC_AB_PROOFS.filter((entry) =>
    REQUIRED_RADIOMETRIC_AB_IDS.has(entry.id)
  )) {
    await checkRadiometricResult(proof);
  }
}
await checkRestirPtSpecialty(RESTIR_PT_SPECIALTY_PROOF);
await checkPtSourcePins(PT_LOCAL_ACCEPTANCE_PROOFS);

console.log(
  `[${SOURCE_ONLY ? "radiometric-ab-source-check" : "radiometric-ab-proof-check"}] PASS ` +
    `(${SOURCE_ONLY ? 0 : REQUIRED_RADIOMETRIC_AB_IDS.size} historical numerical baselines, ` +
    `1 ReSTIR-PT specialty fixture, ${Object.keys(PT_LOCAL_ACCEPTANCE_PROOFS).length} source-pin groups)`,
);
