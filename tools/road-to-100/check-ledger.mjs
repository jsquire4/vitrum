#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
// Checks that the named Road-to-100 source artifacts exist and stay wired into
// the proof umbrella. This prevents handoff/source-of-truth drift where a plan
// names a ledger file that is absent from the repository.

const REQUIRED_SOURCE_FILES = [
  "plan/road-to-100.md",
  "plan/road-to-100-gap-ledger-2026-06-11.md",
  "items_to_fix.md",
];

const LEDGER_PATH = "plan/road-to-100-gap-ledger-2026-06-11.md";
const PACKAGE_PATH = "package.json";
const VALIDATION_QUEUE_PATH = "tools/road-to-100/validation-queue.json";
const EXPECTED_REQUIRED_GREEN_GATES = [
  "npm run typecheck",
  "npm test",
  "npm run shader-gate",
  "npm run proof-check",
  "npm run validate:gpu:smoke after render-changing backend work",
];

/** @param {string} path */
function repoUrl(path) {
  return new URL(`../../${path}`, import.meta.url);
}

/**
 * @param {string} message
 * @returns {never}
 */
function fail(message) {
  throw new Error(`[road-to-100-source-check] ${message}`);
}

/** @param {string} path */
async function readText(path) {
  return await Deno.readTextFile(repoUrl(path));
}

/**
 * @param {string} source
 * @param {string} name
 */
function extractFrozenObjectBlock(source, name) {
  const start = source.indexOf(`const ${name}`);
  if (start < 0) fail(`missing ${name} declaration`);
  const open = source.indexOf("Object.freeze({", start);
  if (open < 0) fail(`${name} must remain an Object.freeze object`);
  const close = source.indexOf("\n});", open);
  if (close < 0) fail(`${name} object block is not closed as expected`);
  return source.slice(open, close + 4);
}

/**
 * @param {string} source
 * @param {string} needle
 */
function extractArrayBlock(source, needle) {
  const start = source.indexOf(needle);
  if (start < 0) fail(`missing array declaration: ${needle}`);
  const open = source.indexOf("[", start);
  if (open < 0) fail(`${needle} array has no opening bracket`);
  const close = source.indexOf("];", open);
  if (close < 0) fail(`${needle} array block is not closed as expected`);
  return source.slice(open, close + 1);
}

/**
 * @param {string} source
 * @param {string} name
 */
function parseStringSupportObject(source, name) {
  const block = extractFrozenObjectBlock(source, name);
  /** @type {Map<string, string>} */
  const rows = new Map();
  for (const match of block.matchAll(/^\s{2}([A-Za-z0-9_]+): '([^']+)'/gm)) {
    rows.set(match[1], match[2]);
  }
  if (rows.size === 0) fail(`${name} parsed zero support rows`);
  return rows;
}

/** @param {string} source */
function parseConsumedMaterialFields(source) {
  const start = source.indexOf("export const CONSUMED_MATERIAL_FIELDS");
  if (start < 0) fail("missing CONSUMED_MATERIAL_FIELDS declaration");
  const open = source.indexOf("[", start);
  if (open < 0) fail("CONSUMED_MATERIAL_FIELDS set has no opening bracket");
  const close = source.indexOf("]);", open);
  if (close < 0) fail("CONSUMED_MATERIAL_FIELDS set is not closed as expected");
  const block = source.slice(open, close + 1);
  return new Set([...block.matchAll(/'([^']+)'/g)].map((match) => match[1]));
}

/** @param {string} source */
function parseAtlasMapFieldUnion(source) {
  const start = source.indexOf("export type AtlasMapField =");
  if (start < 0) fail("missing AtlasMapField union");
  const close = source.indexOf(";", start);
  if (close < 0) fail("AtlasMapField union is not closed as expected");
  return new Set([...source.slice(start, close).matchAll(/'([^']+)'/g)].map((match) => match[1]));
}

/** @param {string} source */
function parseAtlasMapFields(source) {
  const block = extractArrayBlock(source, "const ATLAS_MAP_FIELDS");
  return new Set([...block.matchAll(/field: '([^']+)'/g)].map((match) => match[1]));
}

/** @param {string} source */
function parseMaterialAtlasOffsetNames(source) {
  const start = source.indexOf("export const MATERIAL_MAP_META_TEXEL_OFFSETS = {");
  if (start < 0) fail("missing MATERIAL_MAP_META_TEXEL_OFFSETS declaration");
  const close = source.indexOf("} as const;", start);
  if (close < 0) fail("MATERIAL_MAP_META_TEXEL_OFFSETS block is not closed as expected");
  return new Set([...source.slice(start, close).matchAll(/^\s{2}([A-Z0-9_]+):/gm)].map((match) => match[1]));
}

for (const path of REQUIRED_SOURCE_FILES) {
  try {
    const stat = await Deno.stat(repoUrl(path));
    if (!stat.isFile) fail(`${path} exists but is not a file`);
    if (stat.size <= 0) fail(`${path} is empty`);
  } catch (err) {
    if (err instanceof Deno.errors.NotFound) fail(`${path} is missing`);
    throw err;
  }
}

const ledger = await readText(LEDGER_PATH);
const match = ledger.match(/```json road-to-100-ledger\.v1\n([\s\S]*?)\n```/);
if (!match) fail(`${LEDGER_PATH} must contain a \`\`\`json road-to-100-ledger.v1 block`);

/** @type {{
 *   schema?: string;
 *   ledgerDate?: string;
 *   currentAsOf?: string;
 *   status?: string;
 *   canonicalDetail?: string;
 *   historicalBugLedger?: string;
 *   sourceCheck?: string;
 *   proofUmbrella?: string;
 *   validationQueue?: string;
 *   validationQueueCheck?: string;
 *   closedContractCampaigns?: unknown[];
 *   openPromotionBuckets?: unknown[];
 *   requiredGreenGates?: unknown[];
 * }}
 */
const metadata = JSON.parse(match[1]);
const packageJson = JSON.parse(await readText(PACKAGE_PATH));

if (metadata.schema !== "vitrum.road-to-100.gap-ledger.v1") fail("ledger schema mismatch");
if (metadata.ledgerDate !== "2026-06-11") fail("ledgerDate must remain 2026-06-11 for this named artifact");
if (typeof metadata.currentAsOf !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(metadata.currentAsOf)) {
  fail("ledger currentAsOf must be a YYYY-MM-DD string");
}
if (metadata.currentAsOf < "2026-06-21") {
  fail("ledger currentAsOf predates the historical-provenance reconciliation");
}
const validationQueue = JSON.parse(await readText(VALIDATION_QUEUE_PATH));
if (metadata.currentAsOf !== validationQueue.currentAsOf) {
  fail(`ledger currentAsOf ${metadata.currentAsOf} must match ${VALIDATION_QUEUE_PATH} currentAsOf ${validationQueue.currentAsOf}`);
}
if (metadata.status !== "active") fail("ledger status must be active until Road-to-100 is complete");
if (metadata.canonicalDetail !== "plan/road-to-100.md") fail("canonicalDetail must point at plan/road-to-100.md");
if (metadata.historicalBugLedger !== "items_to_fix.md") fail("historicalBugLedger must point at items_to_fix.md");
if (metadata.sourceCheck !== "npm run road-to-100-source-check") fail("sourceCheck command mismatch");
if (metadata.proofUmbrella !== "npm run proof-check") fail("proofUmbrella command mismatch");
if (metadata.validationQueue !== VALIDATION_QUEUE_PATH) fail(`validationQueue must point at ${VALIDATION_QUEUE_PATH}`);
if (metadata.validationQueueCheck !== "npm run road-to-100-validation-status") fail("validationQueueCheck command mismatch");

if (!Array.isArray(metadata.closedContractCampaigns) || metadata.closedContractCampaigns.length < 5) {
  fail("closedContractCampaigns must summarize the closed implementation campaigns");
}
if (!Array.isArray(metadata.openPromotionBuckets) || metadata.openPromotionBuckets.length < 4) {
  fail("openPromotionBuckets must keep remaining proof/promotion work explicit");
}
if (!Array.isArray(metadata.requiredGreenGates)) {
  fail("requiredGreenGates must be an array");
}
const requiredGreenGates = metadata.requiredGreenGates.map((entry) => String(entry));
if (JSON.stringify(requiredGreenGates) !== JSON.stringify(EXPECTED_REQUIRED_GREEN_GATES)) {
  fail(`requiredGreenGates must exactly match ${JSON.stringify(EXPECTED_REQUIRED_GREEN_GATES)}`);
}
for (const gate of requiredGreenGates) {
  const match = gate.match(/^npm (?:run )?([^ ]+)/);
  if (!match) fail(`required green gate is not an npm command: ${gate}`);
  const scriptName = match[1];
  if (typeof packageJson.scripts?.[scriptName] !== "string") {
    fail(`required green gate ${gate} references missing package script ${scriptName}`);
  }
}
const closedCampaigns = metadata.closedContractCampaigns.map((entry) => String(entry));
if (!closedCampaigns.some((entry) => entry.includes("historical items_to_fix open-heading reconciliation"))) {
  fail("closedContractCampaigns must record the items_to_fix historical-heading reconciliation");
}
const openBuckets = metadata.openPromotionBuckets.map((entry) => String(entry));
for (const [needle, label] of [
  ["gltf-browser-proof-check:required", "required browser glTF proof command"],
  ["HOST-BLOCKED", "browser host-blocked status"],
  ["GRIS/ReSTIR-GI/PPG/NRC/neural/BDPT", "learned/biased-system evidence bucket"],
  ["explicit unsupported/approximate contract rows", "unsupported/approximate contract boundary"],
  ["future contract expands them", "future-contract expansion boundary"],
]) {
  if (!openBuckets.some((entry) => entry.includes(needle))) {
    fail(`openPromotionBuckets must retain ${label}`);
  }
}
if (!ledger.includes("not the active implementation queue")) {
  fail("compact gap ledger must label items_to_fix.md as historical provenance, not the active queue");
}
if (ledger.includes("historical audit provenance plus\n  any explicitly marked open bug rows")) {
  fail("compact gap ledger must not imply items_to_fix.md is a live open-row queue");
}

const road = await readText("plan/road-to-100.md");
const roadmap = await readText("plan/roadmap.md");
const gapExecutionPlan = await readText("plan/gap-closure-execution-plan.md");
const items = await readText("items_to_fix.md");
const shaderGateReadme = await readText("tools/shader-gate/README.md");
const radiometricReadme = await readText("tools/radiometric-ab/README.md");
const walkaroundAbHostStatus = JSON.parse(await readText("tools/radiometric-ab/walkaround-ab-host-status.json"));
const walkaroundAbResults = JSON.parse(await readText("tools/radiometric-ab/walkaround-ab-results.json"));

for (const [docName, docText] of [
  ["plan/road-to-100.md", road],
  ["plan/roadmap.md", roadmap],
  ["plan/gap-closure-execution-plan.md", gapExecutionPlan],
  ["items_to_fix.md", items],
  ["tools/shader-gate/README.md", shaderGateReadme],
]) {
  for (const stalePhrase of [
    "48 WGSL shaders",
    "47→48 WGSL shaders",
    "51 production WGSL modules",
    "compiles **51 shaders**",
    "creates 28 compute/production pipeline variants",
    "**28 pipeline variants**",
    "shader-gate 48/48",
    "6 feature combinations",
    "The 6 combinations above",
    "RANDOM_TYPE=1/2",
    "all 9 rendering rows",
    "All 9 pt-webgpu rows",
    "all 9 pt-webgpu rows",
    "vertex-count / index changes still force full `setScene()` on PT backends",
    "GLASS remains smoke-only",
    "GLASS is smoke-only",
  ]) {
    if (docText.includes(stalePhrase)) {
      fail(`${docName} contains stale hard-coded count claim: ${stalePhrase}`);
    }
  }
}
if (!roadmap.includes("renderer-fidelity-proof-check")) {
  fail("roadmap must point pt-webgpu fidelity-promotion claims at renderer-fidelity-proof-check");
}
if (!roadmap.includes("plan/renderer-fidelity-matrix.md")) {
  fail("roadmap must name plan/renderer-fidelity-matrix.md as the row source of truth");
}
if (!road.includes("shader compile gate discovers the live WGSL inventory")) {
  fail("road-to-100.md must describe shader-gate coverage without stale exact counts");
}
if (!/keep the command output as the count\s+source of truth/.test(shaderGateReadme)) {
  fail("shader-gate README must keep count-source-of-truth wording");
}
if (!shaderGateReadme.includes("| `sobol-on` | false | false | 0 | false | `RANDOM_TYPE=1` Sobol RNG path |")) {
  fail("shader-gate README must include the production Sobol GLSL gate row");
}

if (walkaroundAbHostStatus.verdict === "HOST-BLOCKED") {
  for (const [docName, docText] of [
    ["plan/road-to-100.md", road],
    ["tools/radiometric-ab/README.md", radiometricReadme],
  ]) {
    for (const stalePhrase of [
      "walkaround A/B host now runs to completion",
      "records `PASS-PARTIAL` rather than `HOST-BLOCKED`",
      "the walkaround harness at `PASS-PARTIAL`",
      "radiometric walkaround lane is no longer\nhost-blocked",
      "now records\n`PASS-PARTIAL` in `tools/radiometric-ab/walkaround-ab-host-status.json`",
    ]) {
      if (docText.includes(stalePhrase)) {
        fail(`${docName} contains stale walkaround radiometric host status wording: ${stalePhrase}`);
      }
    }
  }
  if (!/current\s+committed native-Deno radiometric host status is `HOST-BLOCKED`/.test(road)) {
    fail("road-to-100.md must mirror walkaround-ab-host-status.json HOST-BLOCKED verdict");
  }
  if (!/current\s+committed native-Deno host status is `HOST-BLOCKED`/.test(radiometricReadme)) {
    fail("radiometric README must mirror walkaround-ab-host-status.json HOST-BLOCKED verdict");
  }
} else if (walkaroundAbHostStatus.verdict === "PASS-PARTIAL") {
  for (const [docName, docText] of [
    ["plan/road-to-100.md", road],
    ["tools/radiometric-ab/README.md", radiometricReadme],
  ]) {
    for (const stalePhrase of [
      "current committed native-Deno radiometric host status is `HOST-BLOCKED`",
      "current committed native-Deno host status is `HOST-BLOCKED`",
      "is currently host-blocked on this native WSL/Deno path",
      "records the same native-Deno host class",
      "is `HOST-BLOCKED` on the known wgpu-hal panic",
    ]) {
      if (docText.includes(stalePhrase)) {
        fail(`${docName} contains stale walkaround radiometric HOST-BLOCKED wording: ${stalePhrase}`);
      }
    }
  }
  if (!/current\s+committed native-Deno radiometric host status now records `PASS-PARTIAL`/.test(road)) {
    fail("road-to-100.md must mirror walkaround-ab-host-status.json PASS-PARTIAL verdict");
  }
  if (!/latest\s+committed native-Deno\s+status is `PASS-PARTIAL`/.test(radiometricReadme)) {
    fail("radiometric README must mirror walkaround-ab-host-status.json PASS-PARTIAL verdict");
  }
  if (walkaroundAbResults?.sun?.verdict === "PASS") {
    for (const [docName, docText] of [
      ["plan/road-to-100.md", road],
      ["tools/radiometric-ab/README.md", radiometricReadme],
    ]) {
      if (!docText.includes("SUN is `PASS`") && !docText.includes("records SUN as `PASS`")) {
        fail(`${docName} must describe the committed SUN PASS lane when host status is PASS-PARTIAL`);
      }
      if (!docText.includes("receiver ratio = 0.99948")) {
        fail(`${docName} must pin the committed SUN receiver ratio when host status is PASS-PARTIAL`);
      }
    }
  }
}

if (walkaroundAbResults?.glossy?.verdict === "FINDING") {
  for (const [docName, docText] of [
    ["plan/road-to-100.md", road],
    ["tools/radiometric-ab/README.md", radiometricReadme],
  ]) {
    for (const stalePhrase of [
      "GLOSSY `PASS-WEAK`",
      "**Verdict: PASS-WEAK**",
      "| GLOSSY probe | PASS-WEAK |",
      "GLOSSY material probes are currently no-delta checks",
      "no observed\nmaterial-effect delta at 16 spp",
    ]) {
      if (docText.includes(stalePhrase)) {
        fail(`${docName} contains stale GLOSSY PASS-WEAK wording: ${stalePhrase}`);
      }
    }
  }
  if (!road.includes("GLOSSY is a material-effect `FINDING`")) {
    fail("road-to-100.md must mirror walkaround-ab-results.json glossy FINDING verdict");
  }
  if (!radiometricReadme.includes("**Verdict: FINDING**")) {
    fail("radiometric README must mirror walkaround-ab-results.json glossy FINDING verdict");
  }
}

if (walkaroundAbResults?.glass?.verdict === "PASS") {
  for (const [docName, docText] of [
    ["plan/gap-closure-execution-plan.md", gapExecutionPlan],
    ["tools/road-to-100/validation-queue.json", await readText("tools/road-to-100/validation-queue.json")],
  ]) {
    if (!docText.includes("GLASS is `PASS`") && !docText.includes("glass pass")) {
      fail(`${docName} must mirror walkaround-ab-results.json glass PASS verdict`);
    }
  }
}

const walkaroundPromiseLedger = await readText("packages/core/src/engine/promiseLedger.ts");
if (!/row !== 'unsupported'[\s\S]*CONSUMED_MATERIAL_FIELDS/.test(walkaroundPromiseLedger)) {
  fail("walkaround material ledger must keep the consumed-field equivalence comment");
}

const walkaroundMaterialRows = parseStringSupportObject(walkaroundPromiseLedger, "WALKAROUND_MATERIALS");
const walkaroundConsumedFieldsSource = await readText("packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts");
const walkaroundConsumedFields = parseConsumedMaterialFields(walkaroundConsumedFieldsSource);
const walkaroundMaterialAtlas = await readText("packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts");
const atlasMapFieldUnion = parseAtlasMapFieldUnion(walkaroundMaterialAtlas);
const atlasMapFields = parseAtlasMapFields(walkaroundMaterialAtlas);
const atlasOffsetNames = parseMaterialAtlasOffsetNames(walkaroundMaterialAtlas);
const materialAtlasWgsl = await readText("packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts");
const restirPhatWgsl = await readText("packages/walkaround-hybrid/src/shaders/restirPHat.wgsl.ts");
const ddgiProbeUpdateWgsl = await readText("packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateRays.wgsl.ts");

for (const [field, support] of walkaroundMaterialRows) {
  const consumed = walkaroundConsumedFields.has(field);
  const shouldConsume = support !== "unsupported";
  if (consumed !== shouldConsume) {
    fail(
      `walkaround material row drift for ${field}: ledger=${support}, ` +
        `CONSUMED_MATERIAL_FIELDS=${consumed ? "present" : "absent"}`,
    );
  }
}

for (const field of walkaroundConsumedFields) {
  if (!walkaroundMaterialRows.has(field)) {
    fail(`CONSUMED_MATERIAL_FIELDS contains ${field}, but WALKAROUND_MATERIALS has no row for it`);
  }
}

const permanentlyUnsupportedWalkaroundFields = [
  "spectralAttenuation",
  "dispersionAbbeNumber",
  "thinFilmStack",
];

const walkaroundApproximateVolumeScatteringFields = [
  "scatteringCoefficient",
  "scatteringAnisotropy",
  "scatteringCoefficientRGB",
];

const walkaroundApproximateFaceLayerFields = [
  "frontLayer",
  "backLayer",
];

const walkaroundApproximateVertexDisplacementFields = [
  "displacementMap",
  "displacementScale",
  "displacementBias",
];

for (const field of permanentlyUnsupportedWalkaroundFields) {
  if (walkaroundMaterialRows.get(field) !== "unsupported") {
    fail(`walkaround permanent unsupported field was promoted without source-check update: ${field}`);
  }
  if (walkaroundConsumedFields.has(field)) {
    fail(`walkaround permanent unsupported field is present in CONSUMED_MATERIAL_FIELDS: ${field}`);
  }
  if (atlasMapFieldUnion.has(field) || atlasMapFields.has(field)) {
    fail(`walkaround permanent unsupported field is present in material texture atlas fields: ${field}`);
  }
}

for (const field of walkaroundApproximateVertexDisplacementFields) {
  if (walkaroundMaterialRows.get(field) !== "approximate") {
    fail(`walkaround vertex-displacement field must remain approximate, not unsupported/native: ${field}`);
  }
  if (!walkaroundConsumedFields.has(field)) {
    fail(`walkaround approximate vertex-displacement field is absent from CONSUMED_MATERIAL_FIELDS: ${field}`);
  }
  if (atlasMapFieldUnion.has(field) || atlasMapFields.has(field)) {
    fail(`walkaround vertex-displacement field is incorrectly present in the material texture atlas: ${field}`);
  }
}

for (const field of walkaroundApproximateVolumeScatteringFields) {
  if (walkaroundMaterialRows.get(field) !== "approximate") {
    fail(`walkaround volume-scattering field must stay approximate unless volumetric transport proof lands: ${field}`);
  }
  if (!walkaroundConsumedFields.has(field)) {
    fail(`walkaround approximate volume-scattering field is absent from CONSUMED_MATERIAL_FIELDS: ${field}`);
  }
  if (atlasMapFieldUnion.has(field) || atlasMapFields.has(field)) {
    fail(`walkaround volume-scattering field should use scalar metadata, not AtlasMapField map packing: ${field}`);
  }
}

for (const field of walkaroundApproximateFaceLayerFields) {
  if (walkaroundMaterialRows.get(field) !== "approximate") {
    fail(`walkaround face-layer field must stay approximate unless native layered-BSDF proof lands: ${field}`);
  }
  if (!walkaroundConsumedFields.has(field)) {
    fail(`walkaround approximate face-layer field is absent from CONSUMED_MATERIAL_FIELDS: ${field}`);
  }
  if (atlasMapFieldUnion.has(field) || atlasMapFields.has(field)) {
    fail(`walkaround face-layer field should use scalar metadata, not AtlasMapField map packing: ${field}`);
  }
}

for (const atlasField of atlasMapFieldUnion) {
  if (!atlasMapFields.has(atlasField)) {
    fail(`AtlasMapField union includes ${atlasField}, but ATLAS_MAP_FIELDS does not pack it`);
  }
  if (!walkaroundConsumedFields.has(atlasField)) {
    fail(`material atlas exposes ${atlasField}, but CONSUMED_MATERIAL_FIELDS does not include it`);
  }
}
for (const atlasField of atlasMapFields) {
  if (!atlasMapFieldUnion.has(atlasField)) {
    fail(`ATLAS_MAP_FIELDS packs ${atlasField}, but AtlasMapField union does not include it`);
  }
}

for (const forbiddenOffsetNeedle of [
  "DISPLACEMENT",
  "SPECTRAL",
  "DISPERSION",
  "THIN_FILM",
]) {
  for (const offsetName of atlasOffsetNames) {
    if (offsetName.includes(forbiddenOffsetNeedle)) {
      fail(`material atlas offset ${offsetName} appears to pack unsupported walkaround material data`);
    }
  }
  if (materialAtlasWgsl.includes(`MATERIAL_MAP_${forbiddenOffsetNeedle}`)) {
    fail(`materialAtlas.wgsl declares an unsupported walkaround atlas offset: ${forbiddenOffsetNeedle}`);
  }
}

for (const offsetName of ["FRONT_LAYER", "BACK_LAYER"]) {
  if (!atlasOffsetNames.has(offsetName)) {
    fail(`walkaround face-layer metadata offset is missing from host atlas: ${offsetName}`);
  }
  if (!materialAtlasWgsl.includes(`MATERIAL_MAP_${offsetName}_TEXEL_OFFSET`)) {
    fail(`materialAtlas.wgsl is missing face-layer metadata offset: ${offsetName}`);
  }
}
if (!atlasOffsetNames.has("VOLUME_SCATTERING")) {
  fail("walkaround volume-scattering metadata offset is missing from host atlas");
}
if (!materialAtlasWgsl.includes("MATERIAL_MAP_VOLUME_SCATTERING_TEXEL_OFFSET")) {
  fail("materialAtlas.wgsl is missing volume-scattering metadata offset");
}
for (const [sourceName, source, needle] of [
  ["materialAtlas.wgsl.ts", materialAtlasWgsl, "fn sampleFaceLayerControls("],
  ["materialAtlas.wgsl.ts", materialAtlasWgsl, "payload.layerTransmission = faceLayerTransmission(layerControls);"],
  ["restirPHat.wgsl.ts", restirPhatWgsl, "let brdf = surf.layerTransmission * evalGGXWithSpecularClearcoatSheenWithAnisotropyFrame("],
  ["probeUpdateRays.wgsl.ts", ddgiProbeUpdateWgsl, "radiance = radiance * probeMat.layerTransmission;"],
]) {
  if (!source.includes(needle)) {
    fail(`walkaround face-layer implementation proof is missing from ${sourceName}: ${needle}`);
  }
}
for (const [sourceName, source, needle] of [
  ["materialAtlas.wgsl.ts", materialAtlasWgsl, "fn sampleVolumeScatteringControls("],
  ["materialAtlas.wgsl.ts", materialAtlasWgsl, "payload.volumeScattering = sampleVolumeScatteringControls(hit.indices.w);"],
  ["restirPHat.wgsl.ts", restirPhatWgsl, "return applyVolumeScatteringApproximation(brdf, surf.albedo, surf.volumeScattering, surf.normal, surf.wo);"],
  ["probeUpdateRays.wgsl.ts", ddgiProbeUpdateWgsl, "radiance = ddgiApplyVolumeScatteringApproximation("],
]) {
  if (!source.includes(needle)) {
    fail(`walkaround volume-scattering implementation proof is missing from ${sourceName}: ${needle}`);
  }
}

if (!road.includes('For this ledger, "100%" = everything fully implemented')) {
  fail("road-to-100.md must retain the explicit 100% definition");
}
if (!road.includes("Still OPEN for full-path parity")) {
  fail("road-to-100.md must retain explicit open full-path parity language while active");
}
if (!road.includes("Scoped inverse/truthfulness and validation distance remaining")) {
  fail("road-to-100.md must classify inverse residuals as scoped truthfulness/validation distance, not generic implementation distance");
}
if (road.includes("Document in ledger + planner: `displacement*`, `spectralAttenuation`")) {
  fail("road-to-100.md must not classify displacement* as a permanent walkaround unsupported row");
}
if (road.includes("spectral/displacement/thin-film/layers")) {
  fail("road-to-100.md must not include displacement in the permanent walkaround unsupported shorthand");
}
if (!road.includes("scattering and displacement are handled\n> separately as approximate realtime/vertex-level shared-BVH rows")) {
  fail("road-to-100.md must keep the Phase 3 scope note aligned with the scattering/displacement approximate rows");
}
if (!road.includes("`displacementMap` / `displacementScale` / `displacementBias` are not in this unsupported bucket")) {
  fail("road-to-100.md must retain the walkaround displacement approximate-vs-unsupported boundary");
}
if (road.includes("**Implementation distance remaining:** full analytic adjoint replay")) {
  fail("road-to-100.md must not reopen full analytic adjoint replay as a generic implementation-distance blocker");
}
if (road.includes("remaining proof/implementation tail is explicit")) {
  fail("road-to-100.md must not classify guarded multi-vertex BDPT as generic implementation debt");
}
if (!road.includes("The remaining research/promotion tail is explicit\n> multi-vertex BDPT")) {
  fail("road-to-100.md must classify guarded multi-vertex BDPT as research/promotion distance");
}
if (road.includes("blue-noise rotation, broader") || road.includes("blue-noise/per-dimension-audited")) {
  fail("road-to-100.md contains stale Sobol blue-noise-rotation pending wording");
}
if (road.includes("broader bounce/lobe/light\n   dimension assignment audit")) {
  fail("road-to-100.md must not keep the pt-webgpu Sobol dimension audit as pending after the source-level proof landed");
}
if (road.includes("to backend-ready texture payloads")) {
  fail("road-to-100.md contains ambiguous glTF decode target wording");
}
if (!road.includes("pt-webgpu Sobol now carries a binding-free 8x8 ranked tiled")) {
  fail("road-to-100.md must retain the reconciled pt-webgpu Sobol rotation summary");
}
if (!road.includes("source-level\n   dimension audit now pins the monotonic RNG state")) {
  fail("road-to-100.md must retain the pt-webgpu Sobol dimension-audit closure summary");
}
if (!road.includes("tools/radiometric-ab/ab-sobol.mjs")) {
  fail("road-to-100.md must retain the committed pt-webgpu Sobol RMSE A/B harness summary");
}
if (!road.includes("full-tier/real-adapter equal-time evidence before changing defaults")) {
  fail("road-to-100.md must keep the Sobol default-promotion boundary precise");
}
if (!road.includes("to backend-upload-ready CPU/data texture payloads, not live `GPUTexture`")) {
  fail("road-to-100.md must retain the precise glTF decode target boundary");
}
if (road.includes("Remaining work is backend opt-in capability flags, binary-BVH fallback policy")) {
  fail("road-to-100.md contains stale CWBVH opt-in/fallback/parity pending wording");
}
if (road.includes("the fidelity matrix's `pt-webgl`\n  column describes a deleted package and omits pt-webgl2")) {
  fail("road-to-100.md contains stale C5 fidelity-matrix contradiction wording");
}
if (road.includes("| `plan/renderer-fidelity-matrix.md` | Remove deleted `pt-webgl` column; add pt-webgl2 |")) {
  fail("road-to-100.md contains stale pending 5D renderer-matrix action");
}
if (!road.includes("pt-webgpu` exposes\n   the explicit full-tier `bvhTraversal:'cwbvh-closest-experimental'` opt-in")) {
  fail("road-to-100.md must retain the reconciled CWBVH opt-in summary");
}
if (!road.includes("zero-delta readback against binary traversal plus same-scene timing/memory\n   evidence on both the simple Cornell lane and a 144-primitive complex lane")) {
  fail("road-to-100.md must retain the reconciled CWBVH timing/memory proof summary");
}
if (!road.includes("dzn shards\n   prove exact full-tier parity on both original CWBVH lanes")) {
  fail("road-to-100.md must retain the dzn CWBVH cross-adapter parity summary");
}
if (!road.includes("2026-06-22 broader dzn shard proves exact parity on material-lobes")) {
  fail("road-to-100.md must retain the broader CWBVH material/glTF dzn parity summary");
}
if (!road.includes("Timing is mixed on dzn")) {
  fail("road-to-100.md must retain the mixed CWBVH timing/default-promotion boundary");
}
if (!road.includes("browser/real-adapter default-promotion throughput evidence")) {
  fail("road-to-100.md must retain the narrowed CWBVH promotion-only residual");
}
if (!road.includes("fidelity matrix tracks the active `pt-webgl2` / `pt-webgpu` columns and records")) {
  fail("road-to-100.md must retain the reconciled C5 renderer-matrix summary");
}
if (!road.includes("attachVitrum.auto-recreate-scene-snapshot-unavailable")) {
  fail("road-to-100.md must retain the attachVitrum no-live-scene-snapshot warning follow-up");
}
if (!road.includes("`npm run gltf-browser-proof-check:required` is the promotion gate and fails on")) {
  fail("road-to-100.md must retain strict browser glTF promotion-gate wording");
}
if (road.includes("status records each row as `HOST-BLOCKED` at `engine-captureFrame-output`")) {
  fail("road-to-100.md contains stale pt-webgl2 browser proof final-step wording");
}
if (!road.includes("status first records the `engine-captureFrame-output` timeout and then the")) {
  fail("road-to-100.md must retain the current engine-first plus browser-fallback proof summary");
}
if (gapExecutionPlan.includes("now uses `canvas-first` mode")) {
  fail("gap-closure-execution-plan.md contains stale pt-webgl2 browser artifact capture mode wording");
}
if (!gapExecutionPlan.includes("now uses the default `engine-first`")) {
  fail("gap-closure-execution-plan.md must retain the committed engine-first browser proof artifact wording");
}
if (packageJson.scripts?.["gltf-browser-proof-check:required"] !== "deno run --sloppy-imports --allow-read tools/gltf-browser-proof/check-status.mjs --require-pass") {
  fail("package.json must retain the strict browser glTF required proof script");
}

const gltfBrowserProofCheck = await readText("tools/gltf-browser-proof/check-status.mjs");
for (const needle of [
  "const requirePass = Deno.args.includes(\"--require-pass\");",
  "require-pass mode needs browser real glTF PASS; current status is HOST-BLOCKED",
  "fail-closed HOST-BLOCKED on this WSL Playwright host",
]) {
  if (!gltfBrowserProofCheck.includes(needle)) {
    fail(`gltf browser proof checker must retain fail-closed/default plus required-promotion mode: ${needle}`);
  }
}

const gltfBrowserCapture = await readText("tools/gltf-browser-proof/capture-pt-webgl2-real.mjs");
for (const needle of [
  "VITRUM_GLTF_BROWSER_STATUS_PATH",
  "async function captureCanvasPng(page)",
  "method: 'canvas-data-url'",
  "canvas.toDataURL('image/png')",
]) {
  if (!gltfBrowserCapture.includes(needle)) {
    fail(`gltf browser capture harness must retain temp status output and canvas data-url fallback: ${needle}`);
  }
}

const ptWebgpuSource = await readText("packages/pt-webgpu/src/index.ts");
if (ptWebgpuSource.includes("blue-noise rotation, broader dimension audits")) {
  fail("pt-webgpu sampling option docs contain stale Sobol blue-noise pending wording");
}
if (ptWebgpuSource.includes("broader dimension audits and RMSE promotion")) {
  fail("pt-webgpu sampling option docs must not keep the Sobol dimension audit as pending");
}
if (!ptWebgpuSource.includes("with a tiled ranked rotation; the dimension-assignment audit is pinned")) {
  fail("pt-webgpu sampling option docs must retain the Sobol rotation and dimension-audit boundary");
}
if (!ptWebgpuSource.includes("WSL-lite RMSE evidence is bounded")) {
  fail("pt-webgpu sampling option docs must retain the bounded-RMSE/default-promotion boundary");
}
const ptWebgpuSamplingOptionsTest = await readText("packages/pt-webgpu/src/__tests__/samplingOptions.test.ts");
for (const needle of [
  "SOBOL_DIMENSION_AUDIT_2026_06_21",
  "monotonic frame sample key",
  "per-pixel scramble slot",
  "monotonic dimension increment",
  "area-light surface pair consumes adjacent dimensions",
  "photon stream seed",
  "source lobe selection uses stream",
]) {
  if (!ptWebgpuSamplingOptionsTest.includes(needle)) {
    fail(`pt-webgpu Sobol dimension audit test must retain source-level draw-order proof: ${needle}`);
  }
}
const radiometricProofs = await readText("tools/radiometric-ab/proofs.mjs");
for (const needle of [
  'id: "sobol"',
  'resultPath: "tools/radiometric-ab/results-sobol.json"',
  'defaultReady: false',
]) {
  if (!radiometricProofs.includes(needle)) {
    fail(`radiometric A/B proofs must retain Sobol bounded-evidence metadata: ${needle}`);
  }
}

const ptWebgl2ConstructionSourceForBdpt = await readText("packages/pt-webgl2/src/index.ts");
const bdptResearchGateSources = [
  ["pt-webgpu constructor", ptWebgpuSource],
  ["pt-webgl2 constructor", ptWebgl2ConstructionSourceForBdpt],
];
for (const [label, source] of bdptResearchGateSources) {
  for (const needle of [
    "opts.bdptOptions?.experimentalMultiVertex !== true",
    "bdptOptions.maxLightBounces > 1 activates the multi-vertex BDPT research path",
    "bdpt-multivertex-research-mode",
  ]) {
    if (!source.includes(needle)) {
      fail(`${label} must retain the explicit multi-vertex BDPT research opt-in gate: ${needle}`);
    }
  }
}
const ptWebgpuBdptGateTest = await readText("packages/pt-webgpu/src/__tests__/h51WarnCoercions.test.ts");
const ptWebgl2BdptGateTest = await readText("packages/pt-webgl2/src/__tests__/bdptDriver.test.ts");
for (const [label, source] of [
  ["pt-webgpu BDPT gate test", ptWebgpuBdptGateTest],
  ["pt-webgl2 BDPT gate test", ptWebgl2BdptGateTest],
]) {
  if (!source.includes("rejects multi-vertex BDPT unless the research flag is explicit")) {
    fail(`${label} must pin rejection of multi-vertex BDPT without experimentalMultiVertex`);
  }
}

const behavioralGateReadme = await readText("tools/behavioral-gate/README.md");
if (behavioralGateReadme.includes("blue-noise rotation, dimension-assignment audit")) {
  fail("behavioral-gate README contains stale Sobol blue-noise pending wording");
}
if (behavioralGateReadme.includes("dimension-assignment\naudit and equal-time RMSE promotion remain")) {
  fail("behavioral-gate README must not keep the Sobol dimension audit as pending");
}
if (!behavioralGateReadme.includes("variants with the tiled ranked Sobol rotation")) {
  fail("behavioral-gate README must retain the Sobol rotation proof boundary");
}

if (!items.includes("HISTORICAL AUDIT ITEMS") || !items.includes("G-P2.6 PERF-HYGIENE RECONCILIATION")) {
  fail("items_to_fix.md must retain the current historical-audit/provenance markers");
}
if (items.includes("**OPEN ITEMS (2026-06-06):**")) {
  fail("items_to_fix.md must not revive the stale 2026-06-06 OPEN ITEMS heading");
}
if (items.includes("## Section E — Open items discovered 2026-05-19")) {
  fail("items_to_fix.md must keep Section E labeled as historical/closed");
}
if (items.includes("## Section F — Open follow-ups (post-2026-05-30 wave)")) {
  fail("items_to_fix.md must keep Section F labeled as historical/closed-or-proof-tail");
}
for (const [stalePhrase, message] of [
  ["### A1. `createEngine` proxy drops `updateEnvironment` pass-through", "stale A1 updateEnvironment facade heading"],
  ["### A2. `attachVitrum` never plumbs `swapChainView` into `FrameInput`", "stale A2 swapChainView heading"],
  ["### A3. `HybridEngine.updatePrimitive` / `updateEmitter` are `never`", "stale A3 mutation-never heading"],
  ["### A4. `HybridEngine` ignores `FrameInput.viewport`", "stale A4 viewport heading"],
  ["### B1. PPG GPU dispatch is `dispatchWorkgroups(0, 0, 0)`", "stale B1 PPG zero-dispatch heading"],
  ["stubPass.dispatchWorkgroups(0, 0, 0)", "stale B1 PPG stub snippet"],
  ["### B2. RC subsystem ships 1500+ LOC but is unwired", "stale B2 RC unwired heading"],
  ["### B4. OIDN bridge has zero non-test consumers", "stale B4 OIDN zero-consumer heading"],
  ["PPGGuidePass", "stale PPGGuidePass closure citation"],
  ["What may still be broken is the input-packing layout", "stale B3 neural input-packing uncertainty"],
  ["runtime layout \"is not the interleaved per-pixel layout", "stale B3 legacy planar-layout quote"],
  ["dispatches 0 workgroups", "stale C1 PPG no-op dispatch text"],
  ["pre-alpha prototype", "stale C1 pt-webgpu maturity text"],
  ["### C1. CLAUDE.md \"What's done\" section is at least one major work-package behind", "stale C1 CLAUDE.md heading"],
  ["### C2. `memory/in-flight-sweep.md` is mostly stale and misleading", "stale C2 memory heading"],
  ["### C3. CHANGELOG.md likely missing recent entries", "stale C3 changelog heading"],
  ["### C4. Per-package READMEs may overclaim", "stale C4 README heading"],
  ["zero footprint in the\ncurrent code", "stale Section E zero-footprint contradiction"],
  ["validateBvhEncoding` leaves the public surface", "stale E7 validateBvhEncoding un-export claim"],
  ["un-exported `validateBvhEncoding` from `shared-bvh/index.ts`", "stale AGENTS/shared-bvh validateBvhEncoding export claim"],
  ["have not been comprehensively audited", "stale Section E unaudited-wave note"],
  ["the `FrameOutput` contract is back to", "stale E1 live FrameOutput regression wording"],
  ["any `Float32Array` of any length silently satisfies the `Mat4` type", "stale E2 live Mat4 regression wording"],
  ["`packages/pt-webgl/src/forkAccess.ts` is missing", "stale E6 retired pt-webgl missing-file wording"],
  ["`packages/pt-webgl/src/forkAccess.ts` exists", "stale E6 retired pt-webgl live-file wording"],
  ["the pre-E1 implicit-singleton behaviour is back", "stale E5 live shared-device regression wording"],
  ["Neither file is in HEAD; both local copies are back", "stale E4 live shared-WGSL regression wording"],
  ["`packages/core/src/frame.ts:193` reads `export type BackendTexture = unknown;`", "stale E3 live backend-texture regression wording"],
]) {
  if (items.includes(stalePhrase)) {
    fail(`items_to_fix.md contains ${message}`);
  }
}
if (!items.includes("### C1-C4. Documentation truthfulness sweep — closed/source-verified")) {
  fail("items_to_fix.md must retain the reconciled Section C source summary");
}
for (const needle of [
  "### E1. W3-D7 FrameOutput discriminated union — closed (re-landed)",
  "`packages/core/src/frame.ts` now exports `FrameSkipped`, `FrameRendered`, and `FrameOutput = FrameSkipped | FrameRendered`",
  "### E2. W3-D6 Mat4 brand — closed (re-landed)",
  "`packages/core/src/scene/math.ts` declares `MAT4_BRAND`",
  "### E6. W6-E6 ForkAccess indirection — closed (re-landed)",
  "the fork-backed package is no longer present; `packages/pt-webgl2` is the native WebGL2 backend",
  "### E5. W6-E1 reuseSharedWebGpuDevice default flip — closed (re-landed)",
  "gates the process-shared fallback with `opts.reuseSharedWebGpuDevice === true && opts.device == null`",
  "### E4. W2-C6 shared-samplers WGSL primitives — closed (re-landed)",
  "`packages/shared-samplers/src/wgsl/pcg.wgsl.ts` and `packages/shared-samplers/src/wgsl/bsdfPrimitives.wgsl.ts` both exist",
  "### E3. W3-D19 BackendTexture brand — closed (re-landed)",
  "`packages/core/src/frame.ts` declares `BACKEND_TEXTURE_BRAND` / `BACKEND_TEXTURE_FORMAT_BRAND`",
]) {
  if (!items.includes(needle)) {
    fail(`items_to_fix.md must retain reconciled Section E source summary: ${needle}`);
  }
}

const agentBrief = await readText("AGENTS.md");
const claudeBrief = await readText("CLAUDE.md");
for (const [docName, docText] of [
  ["AGENTS.md", agentBrief],
  ["CLAUDE.md", claudeBrief],
]) {
  if (!docText.includes("GitNexus is intentionally not part of the operating workflow for this repo right now.")) {
    fail(`${docName} must retain the GitNexus-disabled operating note`);
  }
  for (const stalePhrase of [
    "MUST run impact analysis before editing any symbol",
    "NEVER edit a function, class, or method without first running `impact`",
    "`packages/pt-webgl/src/forkAccess.ts` exists",
    "walkaround + pt-webgl + pt-webgpu",
    "pt-webgl/pt-webgpu also absorb",
    "A4-progressive (true Hachisuka SPPM — current is streaming-window)",
    "`TextureRef.texCoord` on pt-webgl2 (documented unkept promise)",
    "H-residue (H5 BDPT host driver",
    "/home/jsquire4/.claude/projects/-home-jsquire4-projects-vitrum/memory/",
    "un-exported `validateBvhEncoding` from `shared-bvh/index.ts`",
  ]) {
    if (docText.includes(stalePhrase)) {
      fail(`${docName} contains stale agent-brief text: ${stalePhrase}`);
    }
  }
}
if (!items.includes("### B3. Neural denoiser layout — closed/source-verified")) {
  fail("items_to_fix.md must retain the reconciled B3 neural input-pack heading");
}
if (!items.includes("Remaining neural Road work is production checkpoint + quality A/B, not input layout.")) {
  fail("items_to_fix.md must retain the reconciled B3 residual boundary");
}
for (const needle of [
  "### A1. `createEngine` proxy optional-method forwarding — closed/source-verified",
  "The old A1 missing-`updateEnvironment` facade gap is closed.",
  "### A2. `attachVitrum` WebGPU swap-chain plumbing — closed/source-verified",
  "The old WebGPU black-canvas `swapChainView` omission is closed.",
  "### A3. `HybridEngine.updatePrimitive` / `updateEmitter` — closed/source-verified",
  "not the old `never` API gap.",
  "### A4. `HybridEngine` viewport/setSize contract — closed/source-verified",
  "The old silent viewport ambiguity is closed",
]) {
  if (!items.includes(needle)) {
    fail(`items_to_fix.md must retain reconciled Section A source summary: ${needle}`);
  }
}
for (const needle of [
  "### B1. PPG GPU dispatch — closed/source-verified",
  "Remaining PPG Road work is broad favorable-scene A/B and tuning, not no-op dispatch.",
  "### B2. RC subsystem wiring — closed/source-verified",
  "Remaining RC Road work is promotion/evidence and scene-tuning, not an unwired subsystem.",
  "### B4. OIDN bridge consumers — closed/source-verified",
  "Remaining OIDN posture is host provisioning/quality evidence, not zero consumers.",
]) {
  if (!items.includes(needle)) {
    fail(`items_to_fix.md must retain reconciled Section B source summary: ${needle}`);
  }
}
if (items.includes("point+rect direct lighting with no spot/mesh-area/env/indirect terms")) {
  fail("items_to_fix.md contains the stale H14 adjoint-lighting residual");
}
if (!items.includes("point, spot, rect/disc, mesh-area, and HDRI/environment direct-light replay")) {
  fail("items_to_fix.md must retain the reconciled H14 adjoint-lighting source summary");
}
if (!items.includes("pt-webgpu.hdri-unreadable") || !items.includes("pt-webgpu.hdri-zero-luminance")) {
  fail("items_to_fix.md must retain the pt-webgpu structured HDRI fallback boundary");
}
if (items.includes("V25 BDPT (pt-webgl) root-cause")) {
  fail("items_to_fix.md contains stale retired-pt-webgl V25 BDPT wording");
}
if (!items.includes("pt-webgl2 BDPT browser/visual A/B promotion remains in `plan/renderer-fidelity-matrix.md`")) {
  fail("items_to_fix.md must retain the reconciled pt-webgl2 BDPT promotion boundary");
}

const idempotentDispose = await readText("packages/engine/src/idempotentDispose.ts");
for (const needle of [
  "{ method: 'updateEnvironment', disposedBehavior: 'noop' }",
  "{ method: 'setSize', disposedBehavior: 'noop' }",
  "{ method: 'updateLighting', disposedBehavior: 'noop' }",
]) {
  if (!idempotentDispose.includes(needle)) {
    fail(`engine facade must retain optional-method forwarding row: ${needle}`);
  }
}

const swapChainVanilla = await readText("packages/engine/src/lifecycle/vanilla.ts");
for (const needle of [
  "const swapChainView = acquireSwapChainView(webgpuSwapChain.context",
  "swapChainView: asBackendTexture<'webgpu', GPUTextureView>(opts.swapChainView)",
  "swapChainFormat: asBackendTextureFormat<'webgpu', GPUTextureFormat>",
  "swapChainView,",
  "swapChainFormat: webgpuSwapChain.format",
  "engine.setSize?.(viewportW, viewportH);",
]) {
  if (!swapChainVanilla.includes(needle)) {
    fail(`attachVitrum must retain swap-chain/setSize plumbing: ${needle}`);
  }
}
for (const needle of [
  "attachVitrum.auto-recreate-scene-snapshot-unavailable",
  "the current engine does not implement getScene()",
  "fallback: 'tracked-scene'",
]) {
  if (!swapChainVanilla.includes(needle)) {
    fail(`attachVitrum must retain auto-recreate no-snapshot warning: ${needle}`);
  }
}

const attachVitrumAutoRecreateTest = await readText("packages/engine/src/__tests__/attachVitrumAutoRecreate.test.ts");
for (const needle of [
  "recreates with the backend-retained live scene when fast paths bypass lifecycle setScene tracking",
  "warns and falls back to the tracked scene when backend scene snapshot throws",
  "warns and falls back to the tracked scene when a supplied engine cannot expose a live scene snapshot",
  "attachVitrum.auto-recreate-scene-snapshot-unavailable",
]) {
  if (!attachVitrumAutoRecreateTest.includes(needle)) {
    fail(`attachVitrum auto-recreate tests must retain scene-retention regressions: ${needle}`);
  }
}

const frameContract = await readText("packages/core/src/frame.ts");
for (const needle of [
  "Generic PT engines",
  "`HybridEngine` (`@vitrum/walkaround-hybrid`) does NOT honour",
  "Hosts driving `HybridEngine` directly MUST call `engine.setSize()`",
]) {
  if (!frameContract.includes(needle)) {
    fail(`FrameInput viewport contract must retain HybridEngine setSize wording: ${needle}`);
  }
}

const createEngineInternals = await readText("packages/engine/src/createEngineInternals.ts");
if (!createEngineInternals.includes("| 'attach:initial'")) {
  fail("CreateEngineErrorPhase must retain the React initial attach failure phase");
}

const vitrumCanvasSource = await readText("packages/engine/src/react/VitrumCanvas.tsx");
for (const needle of [
  "try { onAttachErrorRef.current?.(err); } catch",
  "onErrorRef.current?.(err, { phase: 'attach:initial', recoverable: false });",
  "advancedBackend?: CreateEngineOptions['advancedBackend'];",
  "advancedByBackend?: CreateEngineOptions['advancedByBackend'];",
  "onWarning?: (warning: EngineWarning) => void;",
  "onAdapterProfile?: (profile: AdapterProfile) => void;",
  "onWarning: (warning) => { try { onWarningRef.current?.(warning); } catch",
  "onAdapterProfile: (profile) => { try { onAdapterProfileRef.current?.(profile); } catch",
]) {
  if (!vitrumCanvasSource.includes(needle)) {
    fail(`VitrumCanvas must retain guarded initial attach error routing: ${needle}`);
  }
}

const vitrumCanvasTest = await readText("packages/engine/__tests__/vitrumCanvasMount.test.tsx");
for (const needle of [
  "reports initial attach failures through guarded React callbacks",
  "phase: 'attach:initial'",
  "host structured callback failed",
  "advancedBackend: 'pt-webgl2'",
  "host adapter-profile callback failed",
]) {
  if (!vitrumCanvasTest.includes(needle)) {
    fail(`VitrumCanvas attach-error regression test must retain source proof: ${needle}`);
  }
}

const engineGltfBridge = await readText("packages/engine/src/gltf.ts");
for (const needle of [
  "export { GltfCompatibilityError, loadGltfForEngine }",
  "if (backend !== 'pt-webgpu' && engine.backendId === 'pt-webgpu')",
  "disposeEngineAfterRejectedGltfRuntimeProfile(engine);",
  "if (options.runtimeProfile !== undefined)",
  "throw new GltfCompatibilityError({",
]) {
  if (!engineGltfBridge.includes(needle)) {
    fail(`engine glTF bridge must retain runtime-profile/fallback compatibility guard: ${needle}`);
  }
}

const engineGltfTierTest = await readText("packages/engine/src/__tests__/gltfStrictPtWebgpuTier.test.ts");
for (const needle of [
  "honors explicit pt-webgpu-lite runtimeProfile without probing",
  "reports explicit pt-webgpu-lite runtimeProfile on best-effort loads without probing",
  "revalidates actual pt-webgpu fallback engines against the runtime lite profile",
]) {
  if (!engineGltfTierTest.includes(needle)) {
    fail(`engine glTF strict-tier tests must retain runtime-profile/fallback regression: ${needle}`);
  }
}

const gltfErrors = await readText("packages/gltf-adapter/src/errors.ts");
for (const needle of [
  "export class GltfCompatibilityError extends GltfAdapterError",
  "readonly compatibilityMode?: string;",
  "readonly failures: readonly string[];",
  "readonly failureDetails: readonly GltfCompatibilityFailureDetail[];",
]) {
  if (!gltfErrors.includes(needle)) {
    fail(`gltf-adapter must retain structured compatibility error surface: ${needle}`);
  }
}

const gltfEngineBridge = await readText("packages/gltf-adapter/src/engineBridge.ts");
for (const needle of [
  "GltfCompatibilityError,",
  "type GltfCompatibilityFailureDetail,",
  "code: 'GLTF_COMPATIBILITY_REJECTED'",
  "code: 'GLTF_COMPATIBILITY_PROFILE_MISSING'",
  "code: 'GLTF_RUNTIME_PROFILE_MISMATCH'",
]) {
  if (!gltfEngineBridge.includes(needle)) {
    fail(`gltf-adapter engine bridge must throw structured compatibility errors: ${needle}`);
  }
}

const gltfIndex = await readText("packages/gltf-adapter/src/index.ts");
if (!gltfIndex.includes("GltfCompatibilityError")) {
  fail("gltf-adapter package root must export GltfCompatibilityError");
}

const gltfAssetApiTest = await readText("packages/gltf-adapter/src/gltfAssetApi.test.ts");
for (const needle of [
  "GltfCompatibilityError",
  "code: 'GLTF_COMPATIBILITY_REJECTED'",
  "code: 'GLTF_RUNTIME_PROFILE_MISMATCH'",
]) {
  if (!gltfAssetApiTest.includes(needle)) {
    fail(`gltf-adapter tests must retain structured compatibility error assertions: ${needle}`);
  }
}

const gltfTexturePipeline = await readText("packages/gltf-adapter/src/texturePipeline.ts");
for (const needle of [
  "data length ${pixels.data.length} is too short",
  "expected at least ${requiredLength}",
  "code: 'decode-pixels-invalid'",
]) {
  if (!gltfTexturePipeline.includes(needle)) {
    fail(`gltf texture decode validation must fail closed on undersized decoder payloads: ${needle}`);
  }
}
for (const needle of [
  "reports too-short decodePixels payloads as texture diagnostics instead of padding missing texels",
  "data length 11 is too short for 2x2x4",
]) {
  if (!gltfAssetApiTest.includes(needle)) {
    fail(`gltf asset API tests must pin short decoded-payload diagnostics: ${needle}`);
  }
}

const gltfAccessors = await readText("packages/gltf-adapter/src/accessors.ts");
for (const needle of [
  "If bufferView is absent, result stays zero-initialized before any sparse patch",
  "if (accessor.sparse) {",
  "const sv = _resolveSparseViews(gltf, buffers, accessorIndex, accessor, 1, warnings, onDiagnostic);",
  "validateBufferViewAccess(buf, bvIdx, bv, range.requiredByteLength, 'index accessor');",
]) {
  if (!gltfAccessors.includes(needle)) {
    fail(`gltf accessor unpacking must apply pure-sparse index patches instead of returning early: ${needle}`);
  }
}
const gltfAccessorTest = await readText("packages/gltf-adapter/src/accessors.test.ts");
for (const needle of [
  "applies pure-sparse index accessors on top of the implicit zero base",
  "expect(Array.from(out)).toEqual([5, 0, 9, 0]);",
  "rejects base accessors that read past the declared bufferView byteLength",
  "rejects index accessors that read past the declared bufferView byteLength",
  "warns and skips sparse patches that read past declared bufferView byteLength",
]) {
  if (!gltfAccessorTest.includes(needle)) {
    fail(`gltf accessor tests must pin pure-sparse index accessors: ${needle}`);
  }
}

const gltfReadme = await readText("packages/gltf-adapter/README.md");
for (const needle of [
  "Skinned/morphed instancing is supported as a renderable `fallback-generated-mesh` route",
  "the importer expands it to one `SkinnedMeshPrimitive` per authored instance",
  "`reject-unsupported` accepts it",
  "`reject-degraded` rejects it as a non-native approximation",
  "pt-webgpu lite accepts the primitive-constant opaque RGB case by baking the tint into `material.baseColor`",
  "reports a structured unsupported issue for non-constant colors, alpha-bearing colors, or material-variant scenes",
]) {
  if (!gltfReadme.includes(needle)) {
    fail(`gltf-adapter README must describe current compatibility fallback behavior: ${needle}`);
  }
}
if (gltfReadme.includes("imports the skinned/morphed primitive once")) {
  fail("gltf-adapter README must not revive the stale instanced skinned/morphed single-import wording");
}
if (gltfReadme.includes("pt-webgpu lite reports a structured unsupported issue. Secondary vertex color sets")) {
  fail("gltf-adapter README must not revive the stale blanket pt-webgpu-lite COLOR_0 unsupported wording");
}

const gltfFeatureReport = await readText("packages/gltf-adapter/src/featureReport.ts");
const gltfToScene = await readText("packages/gltf-adapter/src/gltfToScene.ts");
for (const needle of [
  "name: 'EXT_mesh_gpu_instancing.skinnedOrMorphed'",
  "support: 'fallback-generated-mesh'",
  "fallback-expanded into one SkinnedMeshPrimitive",
]) {
  if (!gltfFeatureReport.includes(needle)) {
    fail(`gltf feature report must keep instanced skinned/morphed fallback classification: ${needle}`);
  }
}
for (const needle of [
  "code: 'fallback-expanded-gpu-instancing'",
  "per authored instance so every instance remains renderable",
  "const instanceId = `${id}-instance-${instanceIndex}`;",
]) {
  if (!gltfToScene.includes(needle)) {
    fail(`gltf importer must keep instanced skinned/morphed fallback expansion: ${needle}`);
  }
}
for (const needle of [
  "allows fallback-expanded instanced morphed meshes in reject-unsupported mode before constructing an engine",
  "expect(result.asset.scene.primitives).toHaveLength(2);",
  "rejects fallback-expanded instanced morphed meshes in reject-degraded mode before constructing an engine",
  "primitive:EXT_mesh_gpu_instancing.skinnedOrMorphed=fallback-generated-mesh",
]) {
  if (!gltfAssetApiTest.includes(needle)) {
    fail(`gltf asset API tests must pin fallback-expanded instanced morph/skinning compatibility: ${needle}`);
  }
}

const gltfTextures = await readText("packages/gltf-adapter/src/textures.ts");
for (const needle of [
  "interface SelectedTextureImageSource",
  "path: `textures[${textureIndex}].extensions.${extName}.source`",
  "path: selectedSource?.path ?? `textures[${textureIndex}].source`",
]) {
  if (!gltfTextures.includes(needle)) {
    fail(`gltf texture acquisition diagnostics must retain selected source-extension paths: ${needle}`);
  }
}

const gltfTextureSweepTest = await readText("packages/gltf-adapter/src/gltfTextureSweep.test.ts");
for (const needle of [
  "reports selected source-extension missing images at the extension source path",
  "path: 'textures[0].extensions.MSFT_texture_dds.source'",
]) {
  if (!gltfTextureSweepTest.includes(needle)) {
    fail(`gltf texture sweep must pin selected source-extension missing-image diagnostics: ${needle}`);
  }
}

const gltfAssetLoader = await readText("packages/gltf-adapter/src/assetLoader.ts");
for (const needle of [
  "const TEXTURE_DECODE_DIAGNOSTIC_ISSUE_PREFIX = 'texture-decode:';",
  "'decoded-texture-exceeds-max-size'",
  "'decoded-texture-npot-repeat-wrap'",
  "textureDecodeDiagnosticIssuesForCandidate(",
]) {
  if (!gltfAssetLoader.includes(needle)) {
    fail(`gltf decoded texture diagnostics must feed backend compatibility: ${needle}`);
  }
}

for (const needle of [
  "rejects degraded texture decode diagnostics before constructing an engine",
  "rejects missing base primitive material indices in reject-degraded mode",
  "primitive.material.missing-material",
  "import:material-not-found=approximate at meshes[0].primitives[0].material",
  "promotes NPOT repeat-wrap decode diagnostics into degraded compatibility issues",
  "texture-decode:decoded-texture-exceeds-max-size:baseColorMap",
  "texture-decode:decoded-texture-npot-repeat-wrap:baseColorMap",
]) {
  if (!gltfAssetApiTest.includes(needle)) {
    fail(`gltf asset API tests must pin decode diagnostics as degraded compatibility: ${needle}`);
  }
}

const ptWebgl2TexturesArray = await readText("packages/pt-webgl2/src/scene/texturesArray.ts");
for (const needle of [
  "readonly mipLevels: readonly TextureAtlasMipLevel[];",
  "function buildAtlasMipLevels(",
  "atlas.mipLevels.forEach((level, lod) =>",
  "gl.TEXTURE_MAX_LEVEL, atlas.mipLevels.length - 1",
  "updateTextureAtlasLayers(",
]) {
  if (!ptWebgl2TexturesArray.includes(needle)) {
    fail(`pt-webgl2 texture atlas must retain exact mipmapped sampler-policy support: ${needle}`);
  }
}

const ptWebgl2MaterialsTexture = await readText("packages/pt-webgl2/src/scene/materialsTexture.ts");
for (const needle of [
  "const FILTER_MODE_INDEX",
  "const MIP_FILTER_INDEX",
  "function writeSamplerPolicy(",
  "FILTER_MODE_INDEX[ref?.magFilter ?? 'nearest']",
  "FILTER_MODE_INDEX[ref?.minFilter ?? 'nearest'] * 2",
]) {
  if (!ptWebgl2MaterialsTexture.includes(needle)) {
    fail(`pt-webgl2 material texture packer must retain per-map sampler-policy metadata: ${needle}`);
  }
}

const ptWebgl2MaterialStruct = await readText("packages/pt-webgl2/src/glsl/shader/structs/material_struct.glsl.js");
for (const needle of [
  "vec4 sampleMaterialTexture( sampler2DArray tex, vec2 uv, int layer, vec4 samplerPolicy )",
  "float materialTextureRawLod(",
  "vec4 sampleMaterialTextureLinearLevel(",
  "textureSize( tex, level ).xy",
  "texelFetch( tex, ivec3( x0, y0, layer ), level )",
  "m.mapWrap = texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 0}u );",
  "m.thicknessMapWrap = texelFetch1D( tex, i + ${MATERIAL_WRAP_TEXEL_OFFSET + 20}u );",
]) {
  if (!ptWebgl2MaterialStruct.includes(needle)) {
    fail(`pt-webgl2 GLSL material sampler must consume per-map sampler policy: ${needle}`);
  }
}

const walkaroundMaterialTextureAtlas = await readText("packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts");
for (const needle of [
  "'material-texture-sampler-policy-approximation'",
  "const FILTER_MODE_INDEX",
  "function samplerPolicyPacked(",
  "function hasAuthoredSamplerPolicy(",
  "map remains atlas-backed with approximate mip/footprint filtering",
]) {
  if (!walkaroundMaterialTextureAtlas.includes(needle)) {
    fail(`walkaround material texture atlas must retain scoped sampler policy handling: ${needle}`);
  }
}

const ptWebgpuMaterialTextures = await readText("packages/pt-webgpu/src/scene/materialTextures.ts");
for (const needle of [
  "readonly magFilter?: TextureFilterMode;",
  "readonly minFilter?: TextureFilterMode;",
  "readonly mipFilter?: TextureMipFilterMode;",
  "export const MATERIAL_TEX_MIP_POLICY_VEC4_OFFSET",
  "export const MATERIAL_TEX_FILTER_POLICY_VEC4_OFFSET",
  "function writeMipPolicy(",
  "function writeFilterPolicy(",
]) {
  if (!ptWebgpuMaterialTextures.includes(needle)) {
    fail(`pt-webgpu material texture layer uses must retain sampler policy metadata: ${needle}`);
  }
}

const ptWebgpuMaterialTextureArray = await readText("packages/pt-webgpu/src/scene/materialTextureArray.ts");
for (const needle of [
  "'texture-sampler-policy-approximation'",
  "function samplerPolicyIsNativeForPtWebgpu(_use: MaterialTextureLayerUse): boolean",
  "Bump maps still finite-difference in raw UV",
  "fallback: 'nearest-native-sampler-policy'",
  "appendSamplerPolicyWarnings(warnings, structuredWarnings, layerInfos);",
]) {
  if (!ptWebgpuMaterialTextureArray.includes(needle)) {
    fail(`pt-webgpu material texture array must retain native sampler-policy diagnostics: ${needle}`);
  }
}
if (ptWebgpuMaterialTextureArray.includes("regular-map-policy-sampler-with-bump-base-texel-gradient")) {
  fail("pt-webgpu material texture array must not retain the old bump sampler-policy approximation fallback");
}

const ptWebgpuMaterialWgsl = await readText("packages/pt-webgpu/src/wgsl/pathTrace/material.wgsl.ts");
for (const needle of [
  "const MATERIAL_TEX_MIP_POLICY = ${MATERIAL_TEX_MIP_POLICY_VEC4_OFFSET}u;",
  "const MATERIAL_TEX_FILTER_POLICY = ${MATERIAL_TEX_FILTER_POLICY_VEC4_OFFSET}u;",
  "fn materialTextureMipPolicy(base: u32, slot: u32) -> f32",
  "fn materialTexturePolicyLod(lod: f32, mipCount: f32, mipPolicy: f32) -> f32",
  "fn materialTextureFilterPolicy(base: u32, slot: u32) -> vec2f",
  "textureLoad(${texArray}, coord0, layerIdx, lod0u)",
  "textureSampleLevel(${texArray}, materialTexSampler, fittedUv, layerIdx, policyLod)",
  "fn sampleMaterialLayerLinearRawUvPolicy(",
  "MATERIAL_TEX_MIP_BUMP",
]) {
  if (!ptWebgpuMaterialWgsl.includes(needle)) {
    fail(`pt-webgpu WGSL material sampler must consume per-map mip policy: ${needle}`);
  }
}

const ptWebgpuAdjointWgsl = await readText("packages/pt-webgpu/src/wgsl/pathTrace/adjointPass.wgsl.ts");
for (const needle of [
  "fn sampleAdjointMaterialLayerLinearRawUvPolicy(",
  "adjointMaterialTextureMipPolicy(base, mipPolicySlot)",
  "adjointMaterialTextureFilterPolicy(base, mipPolicySlot)",
  "ADJOINT_MATERIAL_TEX_MIP_BUMP",
]) {
  if (!ptWebgpuAdjointWgsl.includes(needle)) {
    fail(`pt-webgpu adjoint material sampler must mirror bump sampler-policy replay: ${needle}`);
  }
}

const gltfMaterialSweepFixture = await readText("tools/gltf-material-sweep/fixture.mjs");
for (const needle of [
  'if (backend === "walkaround-hybrid")',
  'if (backend === "pt-webgl2") return true;',
  'if (backend === "pt-webgpu") return true;',
]) {
  if (!gltfMaterialSweepFixture.includes(needle)) {
    fail(`glTF material sweep sampler-policy fixture must classify backend-native policy consistently: ${needle}`);
  }
}
if (gltfMaterialSweepFixture.includes('backend === "pt-webgpu" && field === "bumpMap"')) {
  fail("glTF material sweep fixture must not retain the old pt-webgpu bump-only sampler-policy exception");
}

const ptWebgpuBsdfWgsl = await readText("packages/pt-webgpu/src/wgsl/pathTrace/bsdf.wgsl.ts");
for (const needle of [
  "fn brdfDirectionalPdfWithIridescence(",
  "let f0Base = materialSpecularF0(baseColor, metallic, specularColor, specularIntensity);",
  "let f0 = iridescenceModifiedF0(",
  "iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,",
]) {
  if (!ptWebgpuBsdfWgsl.includes(needle)) {
    fail(`pt-webgpu sampled BRDF PDFs must retain iridescence-modified F0 parity: ${needle}`);
  }
}
for (const [label, path] of [
  ["full PT sampled Fresnel", "packages/pt-webgpu/src/wgsl/pathTrace/kernel.wgsl.ts"],
  ["lite PT sampled Fresnel", "packages/pt-webgpu/src/wgsl/pathTrace/kernelLite.wgsl.ts"],
  ["ReSTIR-PT source sampling", "packages/pt-webgpu/src/wgsl/pathTrace/restirPtProducer.wgsl.ts"],
  ["BDPT light-subpath sampling", "packages/pt-webgpu/src/wgsl/bdpt/bdptLightSubpath.wgsl.ts"],
]) {
  const source = await readText(path);
  if (!source.includes("iridescenceModifiedF0(")) {
    fail(`${label} must use iridescence-modified F0 for sampled Fresnel/lobe selection`);
  }
}
const ptWebgpuWgslContractTest = await readText("packages/pt-webgpu/src/__tests__/wgslContract.test.ts");
if (!ptWebgpuWgslContractTest.includes("uses iridescence-modified F0 for sampled Fresnel and the full sampled PDF")) {
  fail("pt-webgpu WGSL contract tests must pin sampled iridescence Fresnel/PDF parity");
}

const ptWebgpuUploadSceneBuffersSamplerPolicy = await readText("packages/pt-webgpu/src/scene/uploadSceneBuffers.ts");
for (const needle of [
  "return 'pt-webgpu.material-texture-sampler-policy-approximation';",
  "requestedSamplerPolicies: warning.requestedSamplerPolicies",
]) {
  if (!ptWebgpuUploadSceneBuffersSamplerPolicy.includes(needle)) {
    fail(`pt-webgpu upload warnings must surface sampler-policy approximations: ${needle}`);
  }
}

const walkaroundBackendConstructor = await readText("packages/engine/src/backends/walkaround.ts");
for (const needle of [
  "opts.onAdapterProfile?.(profile);",
  "Host telemetry callbacks must not break backend construction.",
]) {
  if (!walkaroundBackendConstructor.includes(needle)) {
    fail(`walkaround backend must retain guarded adapter-profile callback routing: ${needle}`);
  }
}

const canvasConfigureHelper = await readText("packages/engine/src/configureWebGpuCanvas.ts");
if (!canvasConfigureHelper.includes("Host error callbacks must not break best-effort canvas configuration.")) {
  fail("configureWebGpuCanvas must retain guarded optional error callback handling");
}

const vanillaLifecycle = await readText("packages/engine/src/lifecycle/vanilla.ts");
if (!vanillaLifecycle.includes("Host error callbacks must not break best-effort swap-chain acquisition.")) {
  fail("acquireSwapChainView must retain guarded optional error callback handling");
}

const createEngineConstructionTest = await readText("packages/engine/src/__tests__/createEngineConstruction.test.ts");
if (!createEngineConstructionTest.includes("guards throwing onAdapterProfile callbacks during walkaround construction")) {
  fail("createEngineConstruction must retain adapter-profile callback guard regression test");
}

const configureWebGpuCanvasTest = await readText("packages/engine/src/__tests__/configureWebGpuCanvas.test.ts");
if (!configureWebGpuCanvasTest.includes("guards throwing optional error callbacks")) {
  fail("configureWebGpuCanvas test must retain callback guard regression test");
}

const swapChainPlumbingTest = await readText("packages/engine/__tests__/swapChainPlumbing.test.ts");
if (!swapChainPlumbingTest.includes("guards throwing optional error callbacks on getCurrentTexture failure")) {
  fail("swapChainPlumbing test must retain callback guard regression test");
}

const hybridEngineSource = await readText("packages/walkaround-hybrid/src/HybridEngine.ts");
for (const needle of [
  "updatePrimitive(id: string, patch: Partial<ScenePrimitive>): void",
  "updateEmitter(id: string, patch: Partial<SceneEmitter>): void",
  "this._pipeline?.updateEmitters(this._bvhBuffers);",
  "this._rc?.invalidateBindings();",
  "setSize(width: number, height: number): void",
  "code: 'walkaround-hybrid.viewport-ignored'",
]) {
  if (!hybridEngineSource.includes(needle)) {
    fail(`HybridEngine must retain A3/A4 public API closure seam: ${needle}`);
  }
}

const ppgUpdatePass = await readText("packages/walkaround-hybrid/src/pipeline/passes/PPGUpdatePass.ts");
for (const needle of [
  "const sampleCount = Math.max(1, Math.floor(width / 2)) * Math.max(1, Math.floor(height / 2));",
  "const wgCount = Math.max(1, Math.ceil(sampleCount / 64));",
  "pass.dispatchWorkgroups(wgCount, 1, 1);",
]) {
  if (!ppgUpdatePass.includes(needle)) {
    fail(`walkaround PPGUpdatePass must retain positive dispatch wiring: ${needle}`);
  }
}

const ppgDispatchTest = await readText("packages/walkaround-hybrid/__tests__/ppg-dispatch.test.ts");
for (const needle of [
  "PPGUpdatePass dispatches a positive 1-D workgroup count",
  "expect(captured.length).toBe(1)",
]) {
  if (!ppgDispatchTest.includes(needle)) {
    fail(`walkaround PPG dispatch test must retain positive-dispatch proof: ${needle}`);
  }
}

const hybridEngine = await readText("packages/walkaround-hybrid/src/HybridEngine.ts");
for (const needle of [
  "if (opts.rcEnabled === true)",
  "new RCSubsystem(this._device",
  "this._rcWeight = Math.max(0, Math.min(1, opts.rcWeight ?? 0.5));",
]) {
  if (!hybridEngine.includes(needle)) {
    fail(`HybridEngine must retain RCSubsystem opt-in wiring: ${needle}`);
  }
}

const hybridEngineConfig = await readText("packages/walkaround-hybrid/src/HybridEngineConfig.ts");
for (const needle of [
  "function warnLiteBvhModeOverride",
  "walkaround-hybrid.lite-bvh-mode-overridden",
  "requestedBvhMode: 'tlas'",
  "effectiveBvhMode: 'merged'",
  "opts.onWarning(warning)",
]) {
  if (!hybridEngineConfig.includes(needle)) {
    fail(`HybridEngineConfig must retain structured lite-tier bvh-mode override warning: ${needle}`);
  }
}

const hybridLiteTierTest = await readText("packages/walkaround-hybrid/src/__tests__/hybridLiteTier.test.ts");
for (const needle of [
  "walkaround-hybrid.lite-bvh-mode-overridden",
  "requestedBvhMode: 'tlas'",
  "effectiveBvhMode: 'merged'",
  "fallback: 'merged-bvh'",
]) {
  if (!hybridLiteTierTest.includes(needle)) {
    fail(`hybrid lite-tier tests must pin structured bvh-mode override warning: ${needle}`);
  }
}

const ddgiSource = await readText("packages/walkaround-hybrid/src/ddgi/DDGI.ts");
for (const needle of [
  "onWarning: (warning) => this._warn(warning),",
  "private _warn(warning: EngineWarning): void",
]) {
  if (!ddgiSource.includes(needle)) {
    fail(`DDGI must retain guarded ProbeUpdatePass warning routing: ${needle}`);
  }
}

const ddgiErrorReportingTest = await readText("packages/walkaround-hybrid/src/ddgi/__tests__/ddgiErrorReporting.test.ts");
for (const needle of [
  "guards ProbeUpdatePass construction warnings from throwing host warning callbacks",
  "walkaround-hybrid.ddgi-invalid-max-materials",
  "expect(() => new DDGI({ maxMaterials: 0, onWarning }).dispose()).not.toThrow();",
  "expect(warnSpy).not.toHaveBeenCalled();",
]) {
  if (!ddgiErrorReportingTest.includes(needle)) {
    fail(`DDGI warning-routing tests must pin guarded sub-pass warning behavior: ${needle}`);
  }
}

const hybridFrameOrchestrator = await readText("packages/walkaround-hybrid/src/HybridEngineFrameOrchestrator.ts");
for (const needle of [
  "deps.subsystems.rc.dispatchFrame({",
  "pipeline.setRCInputs(deps.subsystems.rc.buildRCInputs(deps.flags.rcWeight));",
  "pipeline.setRCInputs(null);",
]) {
  if (!hybridFrameOrchestrator.includes(needle)) {
    fail(`HybridEngineFrameOrchestrator must retain RC frame wiring: ${needle}`);
  }
}

const walkaroundPipeline = await readText("packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts");
for (const needle of [
  "registry.register(new PPGUpdatePass(compiled.ppgUpdatePipeline));",
  "registerBuiltinDenoisers(this._denoiserRegistry",
  "this._ddgi.setRCInputs(inputs);",
]) {
  if (!walkaroundPipeline.includes(needle)) {
    fail(`WalkaroundGPUPipeline must retain PPG/RC/OIDN registry wiring: ${needle}`);
  }
}

const walkaroundOidn = await readText("packages/walkaround-hybrid/src/pipeline/denoisers/oidnFinal.ts");
for (const needle of [
  "readonly id = 'oidn-final' as const;",
  "const denoised = await denoiseFinal(inputs, {",
  "modelUrl: this._modelUrl",
]) {
  if (!walkaroundOidn.includes(needle)) {
    fail(`walkaround OIDN final denoiser must retain bridge consumption: ${needle}`);
  }
}

const ptWebgl2DenoiserIndex = await readText("packages/pt-webgl2/src/index.ts");
for (const needle of [
  "if (opts.denoiser === 'oidn-final')",
  "const modelUrl = opts.oidn?.modelUrl;",
  "this.#postDenoiser = new OIDNFinalDispatcher(",
  "pt-webgl2.caustic-strategy-approximation",
  "deterministic refraction-walk heuristic",
  "deterministic cone-traced photon estimate",
]) {
  if (!ptWebgl2DenoiserIndex.includes(needle)) {
    fail(`pt-webgl2 must retain OIDN and caustic-approximation runtime warnings: ${needle}`);
  }
}

const ptWebgl2Mutations = await readText("packages/pt-webgl2/src/scene/mutateSceneTextures.ts");
const ptWebgl2Capabilities = await readText("packages/pt-webgl2/src/capabilities.ts");
for (const needle of [
  "materialAtlasLayerCapacity = refreshTextureAtlasStorage(",
  "textures2DArray = current.textures2DArray;",
  "pushMaterialAtlasRefreshWarning(",
  "textureRefreshMode: 'resident-storage-respecify'",
  "nativePatchMissing: 'targeted-primitive-geometry-splice'",
  "const built = geometryRefreshBuild ?? buildSceneGeometryTextureData(nextScene, {",
  "key === 'material' || key === 'castShadow'",
  "materialWithCastShadow(primitive)",
]) {
  if (!ptWebgl2Mutations.includes(needle)) {
    fail(`pt-webgl2 primitive-list fallbacks must retain resident atlas/storage refresh: ${needle}`);
  }
}
for (const needle of [
  "topology: 'fallback-rebuild'",
  "addPrimitive: 'fallback-rebuild'",
  "removePrimitive: 'fallback-rebuild'",
]) {
  if (!ptWebgl2Capabilities.includes(needle)) {
    fail(`pt-webgl2 capabilities must keep topology/list mutation supportDetails truthful: ${needle}`);
  }
  if (!walkaroundPromiseLedger.includes(needle)) {
    fail(`core promise ledger must keep topology/list mutation supportDetails truthful: ${needle}`);
  }
}
for (const needle of [
  "Remaining tail is performance/promotion only",
  "current predictable API truthfully uses bounded rebuild/repack with structured diagnostics",
]) {
  if (!road.includes(needle)) {
    fail(`road-to-100.md must classify pt-webgl2 topology/list splice as promotion/performance tail: ${needle}`);
  }
}
if (road.includes("Remaining implementation tail before full mutation promotion")) {
  fail("road-to-100.md must not describe truthful topology/list fallback as an implementation blocker");
}

const ptWebgl2DenoiserEngineContractTest = await readText("packages/pt-webgl2/src/__tests__/engineContract.test.ts");
for (const needle of [
  "keeps the atlas texture object resident during dimension-changing primitive list fallbacks",
  "updatePrimitive castShadow patches update the material lane without rebuilding BVH geometry",
  "nativePatchMissing: 'targeted-primitive-geometry-splice'",
  "expect(createTexture.mock.calls.length - initialTextureUploads).toBe(0);",
  "reason: 'capacity-exhausted'",
  "nextLayerCapacity: 8",
  "expect(prim.castShadow).toBe(true);",
]) {
  if (!ptWebgl2DenoiserEngineContractTest.includes(needle)) {
    fail(`pt-webgl2 engine contract tests must pin atlas/castShadow mutation residency: ${needle}`);
  }
}

const walkaroundConsumedMaterialFields = await readText("packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts");
for (const needle of [
  "baseColorMapCanReduceAlpha(mat.baseColorMap) || mat.alphaMap != null",
  "function baseColorMapCanReduceAlpha",
  "if (stride < 4) return false;",
  "collectApproximateRichMaterialPrimitiveFields",
  "RICH_MATERIAL_GI_APPROXIMATION_DETAILS",
]) {
  if (!walkaroundConsumedMaterialFields.includes(needle)) {
    fail(`walkaround alpha-blend warning collector must stay alpha-channel aware: ${needle}`);
  }
}

const walkaroundHybridEngine = await readText("packages/walkaround-hybrid/src/HybridEngine.ts");
for (const needle of [
  "walkaround-hybrid.rich-material-gi-approximation",
  "collectApproximateRichMaterialPrimitiveFields",
  "_warnApproximateRichMaterialPrimitiveFields",
]) {
  if (!walkaroundHybridEngine.includes(needle)) {
    fail(`walkaround rich-material approximation warning must stay wired through HybridEngine: ${needle}`);
  }
}

const walkaroundPrimitiveUpdates = await readText("packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts");
for (const needle of [
  "warnApproximateRichMaterialPrimitiveFields",
  "collectApproximateRichMaterialPrimitiveFields",
]) {
  if (!walkaroundPrimitiveUpdates.includes(needle)) {
    fail(`walkaround rich-material approximation warning must stay wired through material patch fast paths: ${needle}`);
  }
}

const walkaroundConsumedMaterialFieldsTest = await readText("packages/walkaround-hybrid/src/__tests__/consumedMaterialFields.test.ts");
for (const needle of [
  "id: 'base-map-rgb'",
  "id: 'base-map-alpha'",
  "does not emit alpha approximation warning for RGB-only baseColorMap blend coverage",
  "collectApproximateRichMaterialPrimitiveFields",
  "emits a structured warning for rich-material GI approximation",
]) {
  if (!walkaroundConsumedMaterialFieldsTest.includes(needle)) {
    fail(`walkaround alpha-blend tests must pin RGB-vs-RGBA warning precision: ${needle}`);
  }
}

const walkaroundMutationMatrixTest = await readText("packages/walkaround-hybrid/src/__tests__/mutationMatrix.test.ts");
for (const needle of [
  "updatePrimitive(material) emits a structured warning for rich-material GI approximation",
  "walkaround-hybrid.rich-material-gi-approximation",
]) {
  if (!walkaroundMutationMatrixTest.includes(needle)) {
    fail(`walkaround mutation matrix must pin rich-material approximation warnings: ${needle}`);
  }
}

const adjointPass = await readText("packages/pt-webgpu/src/wgsl/pathTrace/adjointPass.wgsl.ts");
for (const needle of [
  "spotLights",
  "meshAreaLights",
  "rectAreaLights",
  "sampleAdjointEnvironmentImportance",
  "ADJOINT_FIELD_ENV_MAP_INTENSITY",
  "ADJOINT_EMITTER_TARGET_MESH",
]) {
  if (!adjointPass.includes(needle)) {
    fail(`pt-webgpu adjoint pass must retain ${needle} while H14 is reconciled`);
  }
}
const pathTraceAdjoint = await readText("packages/pt-webgpu/src/wgsl/pathTrace/pathTraceAdjoint.wgsl.ts");
for (const needle of [
  "dBrdf_dBaseColorWithAnisotropyAndIridescence",
  "dBrdf_dRoughnessWithAnisotropyAndIridescence",
  "dBrdf_dMetallicWithAnisotropyAndIridescence",
  "dBrdf_dSpecularColorWithAnisotropyAndIridescence",
  "dBrdf_dSpecularIntensityWithAnisotropyAndIridescence",
  "adjointEvaluateBrdfWithAnisotropyAndIridescence",
]) {
  if (!pathTraceAdjoint.includes(needle)) {
    fail(`pt-webgpu adjoint BRDF partials must retain fixed-iridescence path replay support: ${needle}`);
  }
}
for (const needle of [
  "dBrdf_dBaseColorWithAnisotropyAndIridescence(",
  "dBrdf_dRoughnessWithAnisotropyAndIridescence(",
  "dBrdf_dMetallicWithAnisotropyAndIridescence(",
  "dBrdf_dSpecularColorWithAnisotropyAndIridescence(",
  "dBrdf_dSpecularIntensityWithAnisotropyAndIridescence(",
]) {
  if (!adjointPass.includes(needle)) {
    fail(`pt-webgpu adjoint direct-light replay must call fixed-iridescence BRDF partial: ${needle}`);
  }
}

const adjointHarnessTest = await readText("packages/pt-webgpu/src/__tests__/adjointHarness.test.ts");
for (const needle of [
  "spotLights",
  "meshAreaLights",
  "sampleAdjointEnvironmentImportance",
  "envMapIntensity",
  "ADJOINT_EMITTER_TARGET_MESH",
]) {
  if (!adjointHarnessTest.includes(needle)) {
    fail(`pt-webgpu adjoint harness test must pin ${needle}`);
  }
}

const ptWebgpuInverseSessionTest = await readText("packages/pt-webgpu/src/__tests__/inverseSession.test.ts");
for (const needle of [
  "keeps envMapIntensity on scoped path-replay for procedural sky baked through the environment CDF",
  "keeps path-replay when active alphaMode uses an RGB-only baseColor texture",
  "environment: {",
  "kind: 'procedural-sky'",
  "code: 'path-replay-unsupported-environment'",
]) {
  if (!ptWebgpuInverseSessionTest.includes(needle)) {
    fail(`pt-webgpu inverse tests must pin procedural-sky environment-CDF path replay: ${needle}`);
  }
}

const ptWebgpuInverseSession = await readText("packages/pt-webgpu/src/inverse/inverseSession.ts");
for (const needle of [
  "const coverage = pathReplayAlphaCoverage(material, primitive);",
  "function pathReplayAlphaCoverage",
  "textureChannelMinimum(material.baseColorMap, 3, 'baseColorMap.a')",
  "textureChannelMinimum(material.alphaMap, 0, 'alphaMap')",
  "function textureChannelStats",
  "if (channel >= stride) return { known: true, min: 1, max: 1, affectedInputs: [] };",
]) {
  if (!ptWebgpuInverseSession.includes(needle)) {
    fail(`pt-webgpu inverse alpha-visibility diagnostics must stay baseColor-alpha aware: ${needle}`);
  }
}
for (const needle of [
  "const PATH_REPLAY_TRANSPORT_ONLY_FIELDS = new Set([",
  "'transmission'",
  "'dispersionAbbeNumber'",
  "'scatteringCoefficientRGB'",
  "const PATH_REPLAY_VISIBILITY_ONLY_FIELDS = new Set(['opacity', 'alphaCutoff']);",
  "const PATH_REPLAY_GEOMETRY_ONLY_FIELDS = new Set(['displacementScale', 'displacementBias']);",
  "message: 'transmission transport is not replayed'",
  "message: 'layered/thin-film material stacks are not replayed'",
  "message: 'spectral/dispersion material transport is not replayed'",
  "message: 'volume/scattering material transport is not replayed'",
  "code: 'path-replay-unsupported-visibility'",
  "finiteDifferenceReason: 'visibility'",
  "code: 'path-replay-unsupported-geometry'",
  "finiteDifferenceReason: 'geometry'",
  "code: 'path-replay-unsupported-normal'",
  "finiteDifferenceReason = 'normal'",
  "function pathReplayEnvironmentIssue",
  "supportedEnvironmentKinds: ['none', 'hdri', 'procedural-sky']",
  "code: 'path-replay-unsupported-light-selection'",
  "directLighting: context.directLighting ?? 'sampled-selection'",
]) {
  if (!ptWebgpuInverseSession.includes(needle)) {
    fail(`pt-webgpu inverse session must keep scoped path-replay finite-difference fallback taxonomy: ${needle}`);
  }
}
for (const needle of [
  "code: 'path-replay-unsupported-render-regime'",
  "code: 'path-replay-unsupported-transport'",
  "code: 'path-replay-unsupported-visibility'",
  "code: 'path-replay-unsupported-geometry'",
  "path-replay-unsupported-light-selection",
  "finiteDifferenceReason",
  "keeps emitter path-replay when receiver materials use replayed top-level normal maps",
  "keeps base BRDF controls on path-replay when a clearcoat-normal map is present",
]) {
  if (!ptWebgpuInverseSessionTest.includes(needle)) {
    fail(`pt-webgpu inverse tests must pin scoped path-replay fallback/replay behavior: ${needle}`);
  }
}
const brdfIssueStart = ptWebgpuInverseSession.indexOf("function materialIssueForBrdf");
const brdfIssueEnd = ptWebgpuInverseSession.indexOf("function materialIssueForAdditiveLobe", brdfIssueStart);
if (brdfIssueStart < 0 || brdfIssueEnd < 0) {
  fail("pt-webgpu inverse session must retain materialIssueForBrdf before materialIssueForAdditiveLobe");
}
const brdfIssueBlock = ptWebgpuInverseSession.slice(brdfIssueStart, brdfIssueEnd);
for (const needle of [
  "allowIridescence: true",
  "allowAnisotropy: true",
]) {
  if (!brdfIssueBlock.includes(needle)) {
    fail(`pt-webgpu inverse base BRDF classifier must retain fixed iridescence/anisotropy replay support: ${needle}`);
  }
}
for (const needle of [
  "keeps AO map intensity on path-replay when fixed anisotropy is active",
  "keeps base-BRDF field %s on path-replay when fixed iridescence is active",
]) {
  if (!ptWebgpuInverseSessionTest.includes(needle)) {
    fail(`pt-webgpu inverse tests must pin fixed-lobe path replay support: ${needle}`);
  }
}
const ptWebgpuBrdfAdjointEmissiveIorTest = await readText("packages/pt-webgpu/src/__tests__/brdfAdjointEmissiveIor.test.ts");
for (const needle of [
  "keeps base/specular BRDF params on path-replay when fixed iridescence is present",
  "keeps additive clearcoat params on path-replay when fixed iridescence is present",
  "keeps clean extension-lobe BRDF params on path-replay when no coupled iridescence lobe is present",
]) {
  if (!ptWebgpuBrdfAdjointEmissiveIorTest.includes(needle)) {
    fail(`pt-webgpu BRDF adjoint tests must pin fixed-iridescence routing: ${needle}`);
  }
}

const walkaroundRisGi = await readText("packages/walkaround-hybrid/src/shaders/risGi.wgsl.ts");
const walkaroundRisGiNrc = await readText("packages/walkaround-hybrid/src/shaders/risGiNrc.wgsl.ts");
for (const [name, source] of [
  ["risGi", walkaroundRisGi],
  ["risGiNrc", walkaroundRisGiNrc],
]) {
  for (const needle of [
    "let ppgGuidedOn_g = (ubo.ppgEnabled == 1u);",
    "let alpha_g = select(0.0, ubo.ppgMixAlpha, ppgGuidedOn_g);",
    "wi = ppgSampleGuidedDir(walkHitPos, &rng);",
    "let pGuide_g = ppgEvalPdf(walkHitPos, wi);",
    "let pSrc_g = alpha_g * pGuide_g + (1.0 - alpha_g) * pCos_g;",
  ]) {
    if (!source.includes(needle)) {
      fail(`walkaround ${name} primary-glass GI branch must retain PPG defensive mixture: ${needle}`);
    }
  }
  if (source.includes("PPG is off for glass pixels")) {
    fail(`walkaround ${name} must not claim PPG is off for glass pixels after primary-glass PPG parity`);
  }
}
const walkaroundRestirGiMaterialParityTest = await readText("packages/walkaround-hybrid/src/__tests__/restirGiMaterialParity.test.ts");
if (!walkaroundRestirGiMaterialParityTest.includes("keeps primary-glass GI branches on the same PPG defensive mixture as opaque GI")) {
  fail("walkaround ReSTIR-GI material parity tests must pin primary-glass PPG parity");
}

const walkaroundBvhCore = await readText("packages/walkaround-hybrid/src/restir/bvhCore.ts");
for (const needle of [
  "function warnScenePackWarnings",
  "walkaround-hybrid.vertex-displacement-skipped",
  "walkaround-hybrid.scene-pack-warning",
  "source: 'shared-bvh'",
]) {
  if (!walkaroundBvhCore.includes(needle)) {
    fail(`walkaround BVH packer warnings must surface structured shared-BVH warnings: ${needle}`);
  }
}
const walkaroundBvhCoreMaterialResolverTest = await readText("packages/walkaround-hybrid/src/restir/__tests__/bvhCoreMaterialResolver.test.ts");
for (const needle of [
  "routes scene-pack vertex-displacement skips through structured warnings in %s mode",
  "walkaround-hybrid.vertex-displacement-skipped",
  "fallback: 'vertex displacement skipped'",
]) {
  if (!walkaroundBvhCoreMaterialResolverTest.includes(needle)) {
    fail(`walkaround BVH core tests must pin structured displacement warning surfacing: ${needle}`);
  }
}

const ptWebgpuEnvironmentPacking = await readText("packages/pt-webgpu/src/scene/environmentPacking.ts");
for (const needle of [
  "bakePreethamSkyEquirect({",
  "hasHdri: true",
  "hdriTexels: texels",
  "hdriCdf: cdf",
]) {
  if (!ptWebgpuEnvironmentPacking.includes(needle)) {
    fail(`pt-webgpu procedural-sky must remain baked into environment-map CDF for adjoint replay: ${needle}`);
  }
}

const ptWebgpuUploadSceneBuffers = await readText("packages/pt-webgpu/src/scene/uploadSceneBuffers.ts");
for (const needle of [
  "structuredWarnings: readonly EngineWarning[]",
  "function structuredEnvironmentWarnings",
  "pt-webgpu.hdri-unreadable",
  "pt-webgpu.hdri-zero-luminance",
  "fallback: 'no-environment'",
]) {
  if (!ptWebgpuUploadSceneBuffers.includes(needle)) {
    fail(`pt-webgpu scene pack must retain structured HDRI fallback diagnostics: ${needle}`);
  }
}

const ptWebgpuIndex = await readText("packages/pt-webgpu/src/index.ts");
for (const needle of [
  "for (const warning of uploadedScene.structuredWarnings)",
  "const structuredScenePackWarnings = new Set",
  "if (structuredScenePackWarnings.has(warning)) continue;",
]) {
  if (!ptWebgpuIndex.includes(needle)) {
    fail(`pt-webgpu engine must drain structured scene-pack warnings without generic duplicates: ${needle}`);
  }
}

const ptWebgpuScenePackEmittersTest = await readText("packages/pt-webgpu/src/__tests__/scenePack.emitters.test.ts");
for (const needle of [
  "pt-webgpu.hdri-unreadable",
  "pt-webgpu.hdri-zero-luminance",
  "fallback: 'no-environment'",
]) {
  if (!ptWebgpuScenePackEmittersTest.includes(needle)) {
    fail(`pt-webgpu scene-pack tests must pin structured HDRI warnings: ${needle}`);
  }
}

const inferenceGraph = await readText("packages/walkaround-hybrid/src/neural/InferenceGraph.ts");
for (const needle of [
  "this._runInputPack(enc, noisyColorBuf, albedoBuf, normalsBuf);",
  "per-pixel INTERLEAVED layout",
  "{ binding: 0, resource: { buffer: noisyColorBuf } }",
  "{ binding: 1, resource: { buffer: albedoBuf } }",
  "{ binding: 2, resource: { buffer: normalsBuf } }",
  "{ binding: 3, resource: { buffer: encInputTensor.buf } }",
  "{ binding: 4, resource: { buffer: this._inputPackUniformBuf } }",
  "pass.dispatchWorkgroups(Math.ceil(pixelCount / 256), 1, 1);",
]) {
  if (!inferenceGraph.includes(needle)) {
    fail(`walkaround neural InferenceGraph must retain input-pack wiring: ${needle}`);
  }
}

const inputPacker = await readText("packages/walkaround-hybrid/src/neural/inputPacker.ts");
for (const needle of [
  "export const INPUT_PACKER_WGSL",
  "encInput[outBase + 0u] = noisyColor[inBase + 0u];",
  "encInput[outBase + 3u] = albedo[inBase + 0u];",
  "encInput[outBase + 6u] = normals[inBase + 0u];",
  "export const INPUT_PACKER_ENTRY = 'inputPackMain';",
]) {
  if (!inputPacker.includes(needle)) {
    fail(`walkaround neural input packer must retain interleaved shader wiring: ${needle}`);
  }
}

const layerResourceAllocator = await readText("packages/walkaround-hybrid/src/neural/layerResourceAllocator.ts");
for (const needle of [
  "import { INPUT_PACKER_WGSL, INPUT_PACKER_ENTRY } from './inputPacker.js';",
  "code: INPUT_PACKER_WGSL",
  "label: 'neural-pipeline-inputPack'",
  "entryPoint: INPUT_PACKER_ENTRY",
  "label: 'neural-uniform-inputPack'",
]) {
  if (!layerResourceAllocator.includes(needle)) {
    fail(`walkaround neural allocator must retain input-pack pipeline wiring: ${needle}`);
  }
}

const hybridEngineDenoiserConfig = await readText("packages/walkaround-hybrid/src/HybridEngineConfig.ts");
for (const needle of [
  "export type ResolvedHybridDenoiser = Exclude<HybridDenoiser, 'auto'>;",
  "function resolveHybridDenoiser",
  "opts.denoiser !== 'auto'",
  "reason = 'host-neural-weights'",
  "reason = 'host-oidn-model-url'",
  "reason = 'lite-neural-unavailable'",
  "packageProvidesProductionWeights: false",
]) {
  if (!hybridEngineDenoiserConfig.includes(needle)) {
    fail(`walkaround denoiser:auto resolver must stay truthful: ${needle}`);
  }
}

const hybridEngineDenoiserWarnings = await readText("packages/walkaround-hybrid/src/HybridEngine.ts");
for (const needle of [
  "walkaround-hybrid.denoiser-auto-resolved",
  "denoiser:'auto' resolved",
  "does not ship production neural weights",
  "walkaround-hybrid.neural-host-weights-required",
]) {
  if (!hybridEngineDenoiserWarnings.includes(needle)) {
    fail(`walkaround denoiser:auto/neural warnings must stay structured: ${needle}`);
  }
}

const walkaroundCapabilitiesPartitionTest = await readText("packages/walkaround-hybrid/src/__tests__/capabilitiesPartition.test.ts");
for (const needle of [
  "resolves denoiser:'auto' to the default when no host model assets exist",
  "reason: 'no-host-model-assets'",
  "resolves denoiser:'auto' to neural only when full-tier host weights are supplied",
  "reason: 'host-neural-weights'",
  "resolves denoiser:'auto' away from neural on lite even if weights are present",
  "reason: 'lite-neural-unavailable'",
]) {
  if (!walkaroundCapabilitiesPartitionTest.includes(needle)) {
    fail(`walkaround denoiser:auto tests must pin resolver cases: ${needle}`);
  }
}

const ptWebgl2Index = await readText("packages/pt-webgl2/src/index.ts");
for (const needle of [
  "function resolveWebgl2AutoDenoiser",
  "pt-webgl2.denoiser-auto-resolved",
  "provide oidn.modelUrl",
  "return { ...opts, denoiser: resolved };",
]) {
  if (!ptWebgl2Index.includes(needle)) {
    fail(`pt-webgl2 denoiser:auto resolver must stay wired: ${needle}`);
  }
}

const ptWebgpuIndexForDenoiser = await readText("packages/pt-webgpu/src/index.ts");
for (const needle of [
  "function resolvePtWebgpuAutoDenoiser",
  "pt-webgpu.denoiser-auto-resolved",
  "provide oidn.modelUrl",
  "new PTEngineWebGPU(effectiveOpts, slot, traceTier)",
]) {
  if (!ptWebgpuIndexForDenoiser.includes(needle)) {
    fail(`pt-webgpu denoiser:auto resolver must stay wired: ${needle}`);
  }
}

const ptWebgpuReadmeForDenoiser = await readText("packages/pt-webgpu/README.md");
for (const needle of [
  "`denoiser: 'auto'` / `'oidn-final'` with aux readback",
  "resolves at construction to host OIDN",
  "`oidn: { modelUrl }` exists",
  "structured",
  "`pt-webgpu.denoiser-auto-resolved` warning",
  "Missing `modelUrl` throws at engine",
  "reported through the denoiser error state",
]) {
  if (!ptWebgpuReadmeForDenoiser.includes(needle)) {
    fail(`pt-webgpu README must document denoiser:auto resolver truthfully: ${needle}`);
  }
}

const ptWebgl2EngineContractTest = await readText("packages/pt-webgl2/src/__tests__/engineContract.test.ts");
for (const needle of [
  "denoiser: 'auto' resolves to no-denoise without host OIDN assets",
  "pt-webgl2.denoiser-auto-resolved",
  "reason: 'host-oidn-model-url'",
]) {
  if (!ptWebgl2EngineContractTest.includes(needle)) {
    fail(`pt-webgl2 denoiser:auto tests must pin resolver cases: ${needle}`);
  }
}

const ptWebgpuUnsupportedDenoiserTest = await readText("packages/pt-webgpu/src/__tests__/unsupportedDenoiserDegrade.test.ts");
for (const needle of [
  "denoiser:'auto' resolves to no-denoise without host OIDN assets",
  "denoiser:'auto' resolved to 'none'",
]) {
  if (!ptWebgpuUnsupportedDenoiserTest.includes(needle)) {
    fail(`pt-webgpu denoiser:auto no-asset test must stay present: ${needle}`);
  }
}

const ptWebgpuOidnFinalIntegrationTest = await readText("packages/pt-webgpu/src/__tests__/oidnFinalIntegration.test.ts");
for (const needle of [
  "denoiser:'auto' resolves to oidn-final when host OIDN config exists",
  "pt-webgpu.denoiser-auto-resolved",
  "reason: 'host-oidn-model-url'",
]) {
  if (!ptWebgpuOidnFinalIntegrationTest.includes(needle)) {
    fail(`pt-webgpu denoiser:auto OIDN test must stay present: ${needle}`);
  }
}

const unetArchitecture = await readText("packages/walkaround-hybrid/src/neural/unetArchitecture.ts");
for (const needle of [
  "pack       → enc_input",
  "output: 'enc_input'",
  "inputs: ['enc_input']",
]) {
  if (!unetArchitecture.includes(needle)) {
    fail(`walkaround neural UNet architecture must retain enc_input graph wiring: ${needle}`);
  }
}

const scripts = packageJson.scripts ?? {};
if (
  typeof scripts["road-to-100-source-check"] !== "string" ||
  !scripts["road-to-100-source-check"].includes("tools/road-to-100/check-ledger.mjs") ||
  !scripts["road-to-100-source-check"].includes("road-to-100-source-gap-scan") ||
  !scripts["road-to-100-source-check"].includes("road-to-100-validation-status")
) {
  fail("package.json must expose road-to-100-source-check with the ledger, source-gap, and validation-queue checks");
}
if (
  scripts["road-to-100-source-gap-scan"] !==
    "deno run --sloppy-imports --allow-read tools/road-to-100/check-source-gap-markers.mjs"
) {
  fail("package.json must expose road-to-100-source-gap-scan");
}
if (
  scripts["road-to-100-validation-status"] !==
    "deno run --sloppy-imports --allow-read tools/road-to-100/check-validation-queue.mjs"
) {
  fail("package.json must expose road-to-100-validation-status");
}
if (typeof scripts["proof-check"] !== "string" || !scripts["proof-check"].includes("road-to-100-source-check")) {
  fail("proof-check must include road-to-100-source-check");
}

console.log("[road-to-100-source-check] PASS (Road source files, ledger metadata, and proof umbrella agree)");
