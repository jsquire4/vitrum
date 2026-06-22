#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
// Verifies committed browser pt-webgl2 real-glTF proof artifacts.

const statusUrl = resolveInputUrl(readFlagValue("--status"), "./pt-webgl2-real-status.json");
const manifestUrl = resolveInputUrl(readFlagValue("--manifest"), "../reference-renders/gltf-real-browser-pt-webgl2/manifest.json");
const requirePass = Deno.args.includes("--require-pass");

const REQUIRED_BROWSER_ASSETS = [
  {
    assetId: "box-textured-glb",
    kind: "textured-glb",
    minTextures: 1,
    requiredExtensions: [],
    requiredHooks: [],
    goldenPath: "tools/reference-renders/gltf-real-browser-pt-webgl2/pt-webgl2-gltf-real-box-textured.png",
    thresholds: { maxRmse: 8, maxMeanAbs: 4, maxAbs: 48 },
  },
  {
    assetId: "cesium-milk-truck-draco",
    kind: "draco",
    minTextures: 0,
    requiredExtensions: ["KHR_draco_mesh_compression"],
    requiredHooks: ["draco"],
    goldenPath: "tools/reference-renders/gltf-real-browser-pt-webgl2/pt-webgl2-gltf-real-draco.png",
    thresholds: { maxRmse: 8, maxMeanAbs: 4, maxAbs: 48 },
  },
  {
    assetId: "meshopt-cube-real",
    kind: "meshopt",
    minTextures: 0,
    requiredExtensions: ["KHR_meshopt_compression"],
    requiredHooks: ["meshopt"],
    goldenPath: "tools/reference-renders/gltf-real-browser-pt-webgl2/pt-webgl2-gltf-real-meshopt.png",
    thresholds: { maxRmse: 8, maxMeanAbs: 4, maxAbs: 48 },
  },
];

/**
 * @param {string} message
 * @returns {never}
 */
function fail(message) {
  throw new Error(`[gltf-browser-proof-check] ${message}`);
}

const status = JSON.parse(await Deno.readTextFile(statusUrl));
const manifest = JSON.parse(await Deno.readTextFile(manifestUrl));

if (status.harness !== "gltf-browser-proof:pt-webgl2-real") fail("status harness mismatch");
if (status.backend !== "pt-webgl2") fail("status backend mismatch");

if (manifest.kind !== "vitrum-browser-gltf-pt-webgl2-goldens") fail("manifest kind mismatch");
if (manifest.backend !== "pt-webgl2") fail("manifest backend mismatch");
if (manifest.assets?.length !== 3) fail("manifest should contain the textured, Draco, and meshopt asset rows");
if (status.assetCount != null && status.assetCount !== manifest.assets.length) fail("status assetCount differs from manifest assets");

const assetsById = byKey(manifest.assets, "assetId");
const statusAssets = Array.isArray(status.assets) ? status.assets : [status];
const statusById = byKey(statusAssets, "assetId");

for (const required of REQUIRED_BROWSER_ASSETS) {
  const asset = assetsById.get(required.assetId);
  if (!asset) fail(`manifest is missing required browser asset ${required.assetId}`);
  if (asset.kind !== required.kind) fail(`${required.assetId}: manifest kind differs from required browser proof contract`);
  if (asset.minTextures !== required.minTextures) fail(`${required.assetId}: manifest minTextures differs from required browser proof contract`);
  if (!sameJson(asset.requiredExtensions ?? [], required.requiredExtensions)) {
    fail(`${required.assetId}: manifest requiredExtensions differ from required browser proof contract`);
  }
  if (!sameJson(asset.requiredHooks ?? [], required.requiredHooks)) {
    fail(`${required.assetId}: manifest requiredHooks differ from required browser proof contract`);
  }
  if (asset.goldenPath !== required.goldenPath) {
    fail(`${required.assetId}: manifest goldenPath differs from required browser proof contract`);
  }
  assertGoldenThresholds(asset.thresholds, required.thresholds, `${required.assetId}: manifest`);
  if (!statusById.has(required.assetId)) fail(`status is missing required browser asset ${required.assetId}`);
}

for (const asset of manifest.assets) {
  const row = statusById.get(asset.assetId);
  if (!row) fail(`status is missing ${asset.assetId}`);
  if (row.harness !== "gltf-browser-proof:pt-webgl2-real") fail(`${asset.assetId}: harness mismatch`);
  if (row.backend !== "pt-webgl2") fail(`${asset.assetId}: backend mismatch`);
  if (row.kind !== asset.kind) fail(`${asset.assetId}: kind mismatch`);
  if (row.telemetry?.backend !== "pt-webgl2") fail(`${asset.assetId}: telemetry backend mismatch`);
  if (row.telemetry?.assetId !== asset.assetId) fail(`${asset.assetId}: telemetry assetId mismatch`);
  if (row.telemetry?.realAssetReady !== true) fail(`${asset.assetId}: realAssetReady must be true`);
  if ((row.telemetry?.textureDecodeReport?.mapCount ?? 0) < (asset.minTextures ?? 0)) {
    fail(`${asset.assetId}: textureDecodeReport.mapCount below manifest expectation`);
  }
  for (const ext of asset.requiredExtensions ?? []) {
    if (!(row.telemetry?.extensionsUsed ?? []).includes(ext)) {
      fail(`${asset.assetId}: missing telemetry extension ${ext}`);
    }
  }
  for (const hook of asset.requiredHooks ?? []) {
    if (row.telemetry?.browserDecodeHooks?.[hook] !== true) {
      fail(`${asset.assetId}: missing browser decode hook proof for ${hook}`);
    }
  }
}

if (status.verdict === "HOST-BLOCKED") {
  for (const row of statusAssets) {
    if (row.verdict !== "HOST-BLOCKED" && row.verdict !== "PASS") fail(`${row.assetId}: unexpected verdict ${row.verdict}`);
    assertNoStaleBrowserBuildWarnings(row);
    if (row.verdict === "PASS") {
      await assertPassingBrowserRow(row, assetsById.get(row.assetId));
    }
    if (row.verdict === "HOST-BLOCKED") {
      assertHostBlockedPageDiagnostics(row);
      assertHostBlockedCaptureAttempts(row);
      if (
        row.step !== "canvas-screenshot" &&
        row.step !== "page-canvas-clip-screenshot" &&
        row.step !== "engine-captureFrame-output" &&
        row.step !== "canvas-data-url"
      ) {
        fail(`${row.assetId}: unexpected host-blocked step ${row.step}`);
      }
      const error = String(row.error ?? "");
      if (
        !error.includes("browser capture timed out") &&
        !error.includes("canvas PNG data URL fallback failed") &&
        !error.includes("engine captureFrame fallback timed out") &&
        !error.includes("page clipped screenshot failed") &&
        !error.includes("page.screenshot: Timeout") &&
        !error.includes("locator.screenshot: Timeout") &&
        !error.includes("page clipped screenshot timed out") &&
        !error.includes("canvas element screenshot timed out")
      ) {
        fail(`${row.assetId}: HOST-BLOCKED status must preserve the timeout/readback reason`);
      }
    }
  }
  if (requirePass) {
    fail("require-pass mode needs browser real glTF PASS; current status is HOST-BLOCKED");
  }
  console.log("[gltf-browser-proof-check] PASS (pt-webgl2 browser real glTF lanes are fail-closed HOST-BLOCKED on this WSL Playwright host)");
} else if (status.verdict === "PASS") {
  for (const row of statusAssets) {
    if (row.verdict !== "PASS") fail(`${row.assetId}: top-level PASS requires every row to PASS`);
    await assertPassingBrowserRow(row, assetsById.get(row.assetId));
  }
  console.log("[gltf-browser-proof-check] PASS (pt-webgl2 browser real glTF proof)");
} else {
  fail(`status verdict must be PASS or HOST-BLOCKED, got ${status.verdict}`);
}

/** @param {Record<string, any>} row */
function assertHostBlockedPageDiagnostics(row) {
  const diagnostics = row.pageDiagnostics;
  if (diagnostics == null || typeof diagnostics !== "object") {
    fail(`${row.assetId}: HOST-BLOCKED status must include pageDiagnostics`);
  }
  if (diagnostics.phase !== "pre-capture") {
    fail(`${row.assetId}: pageDiagnostics.phase must be pre-capture`);
  }
  if (diagnostics.ready !== true) {
    fail(`${row.assetId}: pageDiagnostics must prove VITRUM_CAPTURE_READY before readback`);
  }
  if (diagnostics.canvasPresent !== true) {
    fail(`${row.assetId}: pageDiagnostics must prove a canvas existed before readback`);
  }
  if (diagnostics.captureFrameInstalled !== true) {
    fail(`${row.assetId}: pageDiagnostics must prove VITRUM_CAPTURE_FRAME was installed before readback`);
  }
  const canvasWidth = Number(diagnostics.canvasWidth ?? 0);
  const canvasHeight = Number(diagnostics.canvasHeight ?? 0);
  const clientWidth = Number(diagnostics.canvasClientWidth ?? 0);
  const clientHeight = Number(diagnostics.canvasClientHeight ?? 0);
  const rectWidth = Number(diagnostics.canvasRect?.width ?? 0);
  const rectHeight = Number(diagnostics.canvasRect?.height ?? 0);
  if (!(canvasWidth > 0 && canvasHeight > 0 && clientWidth > 0 && clientHeight > 0 && rectWidth > 0 && rectHeight > 0)) {
    fail(`${row.assetId}: pageDiagnostics must include nonzero canvas dimensions before readback`);
  }
  if (!String(diagnostics.url ?? "").includes(row.assetId)) {
    fail(`${row.assetId}: pageDiagnostics.url must include the captured asset id`);
  }
}

/** @param {Record<string, any>} row */
function assertHostBlockedCaptureAttempts(row) {
  const attempts = row.captureAttempts;
  if (!Array.isArray(attempts) || attempts.length === 0) {
    fail(`${row.assetId}: HOST-BLOCKED status must include captureAttempts[]`);
  }
  const allowedMethods = new Set([
    "playwright-screenshot",
    "page-canvas-clip-screenshot",
    "canvas-data-url",
    "engine-captureFrame-output",
  ]);
  const allowedStatuses = new Set(["started", "failed", "succeeded"]);
  let hasBlockedAttempt = false;
  let hasStepAttempt = false;
  let hasEngineAttempt = false;
  for (const attempt of attempts) {
    if (attempt == null || typeof attempt !== "object") fail(`${row.assetId}: invalid capture attempt`);
    if (!allowedMethods.has(attempt.method)) fail(`${row.assetId}: unexpected capture attempt method ${attempt.method}`);
    if (!allowedStatuses.has(attempt.status)) fail(`${row.assetId}: unexpected capture attempt status ${attempt.status}`);
    if (attempt.status === "started" || attempt.status === "failed") hasBlockedAttempt = true;
    if (attempt.method === row.step || attempt.step === row.step) hasStepAttempt = true;
    if (attempt.method === "engine-captureFrame-output") hasEngineAttempt = true;
    if (attempt.status === "failed" && String(attempt.error ?? "").length === 0) {
      fail(`${row.assetId}: failed capture attempt ${attempt.method} must include an error`);
    }
  }
  if (!hasBlockedAttempt) fail(`${row.assetId}: HOST-BLOCKED status must preserve the blocked capture attempt`);
  if (!hasStepAttempt) fail(`${row.assetId}: captureAttempts[] must include the host-blocked step ${row.step}`);
  if (
    row.captureMode !== "canvas-only" &&
    row.captureMode !== "canvas-first" &&
    !hasEngineAttempt
  ) {
    fail(`${row.assetId}: HOST-BLOCKED status must include an engine-captureFrame-output attempt`);
  }
}

/** @param {Record<string, any>} row */
function assertNoStaleBrowserBuildWarnings(row) {
  const fragments = [
    row.error,
    row.serverLog,
    ...(Array.isArray(row.console) ? row.console : []),
    ...(Array.isArray(row.consoleMessages) ? row.consoleMessages : []),
    ...(Array.isArray(row.pageMessages) ? row.pageMessages : []),
    ...(Array.isArray(row.logs) ? row.logs : []),
  ].map((value) => String(value ?? ""));
  const joined = fragments.join("\n");
  if (
    joined.includes("vite:import-analysis") ||
    joined.includes("dynamic import cannot be analyzed") ||
    joined.includes("texturePipeline.ts")
  ) {
    fail(`${row.assetId}: stale browser build warning present in committed status`);
  }
}

/** @param {Record<string, any>} row */
function assertInformativeCapture(row) {
  const structure = row.structure;
  if (structure == null || typeof structure !== "object") {
    fail(`${row.assetId}: PASS status must include visual-structure metrics`);
  }
  const thresholds = structure.thresholds;
  const minLumaRange = thresholds?.minLumaRange ?? 12;
  const minUniqueColorCount = thresholds?.minUniqueColorCount ?? 16;
  const minNonDominantFraction = thresholds?.minNonDominantFraction ?? 0.05;
  if (!(structure.lumaRange >= minLumaRange)) {
    fail(`${row.assetId}: lumaRange ${structure.lumaRange} is below ${minLumaRange}`);
  }
  if (!(structure.uniqueColorCount >= minUniqueColorCount)) {
    fail(`${row.assetId}: uniqueColorCount ${structure.uniqueColorCount} is below ${minUniqueColorCount}`);
  }
  if (!(structure.nonDominantFraction >= minNonDominantFraction)) {
    fail(`${row.assetId}: nonDominantFraction ${structure.nonDominantFraction} is below ${minNonDominantFraction}`);
  }
}

/**
 * @param {Record<string, any>} row
 * @param {Record<string, any> | undefined} asset
 */
async function assertPassingBrowserRow(row, asset) {
  if (asset == null) fail(`${row.assetId}: PASS row missing manifest asset`);
  const manifestAsset = asset;
  if (
    row.captureMethod !== undefined &&
    row.captureMethod !== "playwright-screenshot" &&
    row.captureMethod !== "page-canvas-clip-screenshot" &&
    row.captureMethod !== "engine-captureFrame-output" &&
    row.captureMethod !== "canvas-data-url"
  ) {
    fail(`${row.assetId}: unexpected captureMethod ${row.captureMethod}`);
  }
  if (!(row.luminance > 0.005)) fail(`${row.assetId}: capture luminance must be non-black`);
  assertInformativeCapture(row);
  if (row.golden?.pass !== true) fail(`${row.assetId}: golden comparison did not pass`);
  if (row.golden?.path !== manifestAsset.goldenPath) fail(`${row.assetId}: manifest goldenPath mismatch`);
  assertGoldenThresholds(row.golden?.thresholds, manifestAsset.thresholds, `${row.assetId}: golden`);

  const goldenUrl = new URL(`../../${row.golden.path}`, import.meta.url);
  const stat = await Deno.stat(goldenUrl);
  if (!stat.isFile || stat.size <= 8) fail(`${row.assetId}: golden PNG is missing or empty`);
  const header = await Deno.readFile(goldenUrl);
  if (header[0] !== 0x89 || header[1] !== 0x50 || header[2] !== 0x4e || header[3] !== 0x47) {
    fail(`${row.assetId}: golden file is not a PNG`);
  }
}

/**
 * @param {readonly Record<string, any>[] | undefined} items
 * @param {string} key
 */
function byKey(items, key) {
  const map = new Map();
  for (const item of items ?? []) {
    const value = item[key];
    if (typeof value !== "string" || value.length === 0) fail(`invalid ${key} value`);
    if (map.has(value)) fail(`duplicate ${key}: ${value}`);
    map.set(value, item);
  }
  return map;
}

/**
 * @param {Record<string, any> | undefined} actual
 * @param {Record<string, any> | undefined} expected
 * @param {string} label
 */
function assertGoldenThresholds(actual, expected, label) {
  if (actual == null || typeof actual !== "object") fail(`${label} thresholds missing`);
  if (expected == null || typeof expected !== "object") fail(`${label} expected thresholds missing`);
  for (const key of ["maxRmse", "maxMeanAbs", "maxAbs"]) {
    if (actual[key] !== expected[key]) {
      fail(`${label} threshold ${key} must be ${expected[key]}, got ${actual[key]}`);
    }
  }
}

/**
 * @param {unknown} a
 * @param {unknown} b
 */
function sameJson(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

/** @param {string} name */
function readFlagValue(name) {
  for (let i = 0; i < Deno.args.length; i += 1) {
    const arg = Deno.args[i];
    if (arg === name && Deno.args[i + 1]) return Deno.args[i + 1];
    if (arg?.startsWith(`${name}=`)) return arg.slice(name.length + 1);
  }
  return null;
}

/**
 * @param {string | null} value
 * @param {string} fallback
 */
function resolveInputUrl(value, fallback) {
  if (value == null || value.length === 0) return new URL(fallback, import.meta.url);
  if (/^[a-z][a-z0-9+.-]*:/i.test(value)) return new URL(value);
  return new URL(value, import.meta.url);
}
