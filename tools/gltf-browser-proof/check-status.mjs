#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
// Verifies committed browser pt-webgl2 real-glTF proof artifacts.

const statusUrl = new URL("./pt-webgl2-real-status.json", import.meta.url);
const manifestUrl = new URL("../reference-renders/gltf-real-browser-pt-webgl2/manifest.json", import.meta.url);

function fail(message) {
  throw new Error(`[gltf-browser-proof-check] ${message}`);
}

const status = JSON.parse(await Deno.readTextFile(statusUrl));
const manifest = JSON.parse(await Deno.readTextFile(manifestUrl));

if (status.harness !== "gltf-browser-proof:pt-webgl2-real") fail("status harness mismatch");
if (status.assetId !== "box-textured-glb") fail("status assetId mismatch");
if (status.backend !== "pt-webgl2") fail("status backend mismatch");

if (manifest.kind !== "vitrum-browser-gltf-pt-webgl2-goldens") fail("manifest kind mismatch");
if (manifest.backend !== "pt-webgl2") fail("manifest backend mismatch");
if (manifest.assets?.length !== 1) fail("manifest should contain exactly one asset row");
const [asset] = manifest.assets;
if (asset.assetId !== status.assetId) fail("manifest assetId mismatch");

if (status.verdict === "HOST-BLOCKED") {
  if (status.step !== "canvas-readback") fail(`unexpected host-blocked step ${status.step}`);
  if (!String(status.error ?? "").includes("browser capture timed out")) fail("HOST-BLOCKED status must preserve the timeout reason");
  console.log("[gltf-browser-proof-check] PASS (pt-webgl2 browser real glTF lane is fail-closed HOST-BLOCKED on this WSL Playwright host)");
} else if (status.verdict === "PASS") {
  if (status.telemetry?.backend !== "pt-webgl2") fail("telemetry backend mismatch");
  if (status.telemetry?.assetId !== "box-textured-glb") fail("telemetry assetId mismatch");
  if (status.telemetry?.realAssetReady !== true) fail("realAssetReady must be true");
  if ((status.telemetry?.textureDecodeReport?.mapCount ?? 0) < 1) fail("textureDecodeReport.mapCount must prove the textured asset");
  if (!(status.luminance > 0.005)) fail("capture luminance must be non-black");
  if (status.golden?.pass !== true) fail("golden comparison did not pass");
  if (status.golden?.path !== asset.goldenPath) fail("manifest goldenPath mismatch");
  if (status.golden?.thresholds?.maxRmse !== 8 || status.golden?.thresholds?.maxMeanAbs !== 4 || status.golden?.thresholds?.maxAbs !== 48) {
    fail("golden thresholds mismatch");
  }

  const goldenUrl = new URL(`../../${status.golden.path}`, import.meta.url);
  const stat = await Deno.stat(goldenUrl);
  if (!stat.isFile || stat.size <= 8) fail("golden PNG is missing or empty");
  const header = await Deno.readFile(goldenUrl);
  if (header[0] !== 0x89 || header[1] !== 0x50 || header[2] !== 0x4e || header[3] !== 0x47) {
    fail("golden file is not a PNG");
  }
  console.log("[gltf-browser-proof-check] PASS (pt-webgl2 browser real BoxTextured glTF proof)");
} else {
  fail(`status verdict must be PASS or HOST-BLOCKED, got ${status.verdict}`);
}
