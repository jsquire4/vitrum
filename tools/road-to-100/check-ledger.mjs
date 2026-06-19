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
]) {
  if (items.includes(stalePhrase)) {
    fail(`items_to_fix.md contains ${message}`);
  }
}
if (!items.includes("### C1-C4. Documentation truthfulness sweep — closed/source-verified")) {
  fail("items_to_fix.md must retain the reconciled Section C source summary");
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

const ptWebgl2Index = await readText("packages/pt-webgl2/src/index.ts");
for (const needle of [
  "if (opts.denoiser === 'oidn-final')",
  "const modelUrl = opts.oidn?.modelUrl;",
  "this.#postDenoiser = new OIDNFinalDispatcher(",
]) {
  if (!ptWebgl2Index.includes(needle)) {
    fail(`pt-webgl2 must retain oidn-final engine consumer: ${needle}`);
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
  "for (const warning of packed.structuredWarnings)",
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
