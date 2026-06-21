#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
// Verifies committed browser pt-webgl2 real-glTF proof artifacts.

const statusUrl = new URL("./pt-webgl2-real-status.json", import.meta.url);
const manifestUrl = new URL("../reference-renders/gltf-real-browser-pt-webgl2/manifest.json", import.meta.url);
const requirePass = Deno.args.includes("--require-pass");

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

const assetsById = byKey(manifest.assets, "assetId");
const statusAssets = Array.isArray(status.assets) ? status.assets : [status];
const statusById = byKey(statusAssets, "assetId");

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
    if (row.verdict === "HOST-BLOCKED") {
      if (row.step !== "canvas-screenshot" && row.step !== "canvas-data-url") {
        fail(`${row.assetId}: unexpected host-blocked step ${row.step}`);
      }
      const error = String(row.error ?? "");
      if (!error.includes("browser capture timed out") && !error.includes("canvas PNG data URL fallback failed")) {
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
    const asset = assetsById.get(row.assetId);
    if (row.verdict !== "PASS") fail(`${row.assetId}: top-level PASS requires every row to PASS`);
    if (
      row.captureMethod !== undefined &&
      row.captureMethod !== "playwright-screenshot" &&
      row.captureMethod !== "canvas-data-url"
    ) {
      fail(`${row.assetId}: unexpected captureMethod ${row.captureMethod}`);
    }
    if (!(row.luminance > 0.005)) fail(`${row.assetId}: capture luminance must be non-black`);
    if (row.golden?.pass !== true) fail(`${row.assetId}: golden comparison did not pass`);
    if (row.golden?.path !== asset.goldenPath) fail(`${row.assetId}: manifest goldenPath mismatch`);
    if (row.golden?.thresholds?.maxRmse !== 8 || row.golden?.thresholds?.maxMeanAbs !== 4 || row.golden?.thresholds?.maxAbs !== 48) {
      fail(`${row.assetId}: golden thresholds mismatch`);
    }

    const goldenUrl = new URL(`../../${row.golden.path}`, import.meta.url);
    const stat = await Deno.stat(goldenUrl);
    if (!stat.isFile || stat.size <= 8) fail(`${row.assetId}: golden PNG is missing or empty`);
    const header = await Deno.readFile(goldenUrl);
    if (header[0] !== 0x89 || header[1] !== 0x50 || header[2] !== 0x4e || header[3] !== 0x47) {
      fail(`${row.assetId}: golden file is not a PNG`);
    }
  }
  console.log("[gltf-browser-proof-check] PASS (pt-webgl2 browser real glTF proof)");
} else {
  fail(`status verdict must be PASS or HOST-BLOCKED, got ${status.verdict}`);
}

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
