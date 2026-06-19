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
 *   status?: string;
 *   canonicalDetail?: string;
 *   historicalBugLedger?: string;
 *   sourceCheck?: string;
 *   proofUmbrella?: string;
 *   closedContractCampaigns?: unknown[];
 *   openPromotionBuckets?: unknown[];
 *   requiredGreenGates?: unknown[];
 * }}
 */
const metadata = JSON.parse(match[1]);

if (metadata.schema !== "vitrum.road-to-100.gap-ledger.v1") fail("ledger schema mismatch");
if (metadata.ledgerDate !== "2026-06-11") fail("ledgerDate must remain 2026-06-11 for this named artifact");
if (metadata.status !== "active") fail("ledger status must be active until Road-to-100 is complete");
if (metadata.canonicalDetail !== "plan/road-to-100.md") fail("canonicalDetail must point at plan/road-to-100.md");
if (metadata.historicalBugLedger !== "items_to_fix.md") fail("historicalBugLedger must point at items_to_fix.md");
if (metadata.sourceCheck !== "npm run road-to-100-source-check") fail("sourceCheck command mismatch");
if (metadata.proofUmbrella !== "npm run proof-check") fail("proofUmbrella command mismatch");

if (!Array.isArray(metadata.closedContractCampaigns) || metadata.closedContractCampaigns.length < 5) {
  fail("closedContractCampaigns must summarize the closed implementation campaigns");
}
if (!Array.isArray(metadata.openPromotionBuckets) || metadata.openPromotionBuckets.length < 4) {
  fail("openPromotionBuckets must keep remaining proof/promotion work explicit");
}
if (!Array.isArray(metadata.requiredGreenGates) || !metadata.requiredGreenGates.includes("npm run proof-check")) {
  fail("requiredGreenGates must include npm run proof-check");
}

const road = await readText("plan/road-to-100.md");
if (!road.includes('For this ledger, "100%" = everything fully implemented')) {
  fail("road-to-100.md must retain the explicit 100% definition");
}
if (!road.includes("Still OPEN for full-path parity")) {
  fail("road-to-100.md must retain explicit open full-path parity language while active");
}

const items = await readText("items_to_fix.md");
if (!items.includes("OPEN ITEMS") || !items.includes("G-P2.6 PERF-HYGIENE RECONCILIATION")) {
  fail("items_to_fix.md must retain the current open-items/provenance markers");
}
for (const [stalePhrase, message] of [
  ["What may still be broken is the input-packing layout", "stale B3 neural input-packing uncertainty"],
  ["runtime layout \"is not the interleaved per-pixel layout", "stale B3 legacy planar-layout quote"],
  ["dispatches 0 workgroups", "stale C1 PPG no-op dispatch text"],
  ["pre-alpha prototype", "stale C1 pt-webgpu maturity text"],
]) {
  if (items.includes(stalePhrase)) {
    fail(`items_to_fix.md contains ${message}`);
  }
}
if (!items.includes("### B3. Neural denoiser layout — closed/source-verified")) {
  fail("items_to_fix.md must retain the reconciled B3 neural input-pack heading");
}
if (!items.includes("Remaining neural Road work is production checkpoint + quality A/B, not input layout.")) {
  fail("items_to_fix.md must retain the reconciled B3 residual boundary");
}
if (items.includes("point+rect direct lighting with no spot/mesh-area/env/indirect terms")) {
  fail("items_to_fix.md contains the stale H14 adjoint-lighting residual");
}
if (!items.includes("point, spot, rect/disc, mesh-area, and HDRI/environment direct-light replay")) {
  fail("items_to_fix.md must retain the reconciled H14 adjoint-lighting source summary");
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

const packageJson = JSON.parse(await readText(PACKAGE_PATH));
const scripts = packageJson.scripts ?? {};
if (scripts["road-to-100-source-check"] !== "deno run --sloppy-imports --allow-read tools/road-to-100/check-ledger.mjs") {
  fail("package.json must expose road-to-100-source-check");
}
if (typeof scripts["proof-check"] !== "string" || !scripts["proof-check"].includes("road-to-100-source-check")) {
  fail("proof-check must include road-to-100-source-check");
}

console.log("[road-to-100-source-check] PASS (Road source files, ledger metadata, and proof umbrella agree)");
