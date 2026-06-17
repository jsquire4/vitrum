#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
// Verifies that the real glTF behavioral-proof metadata, manifest, and PNGs agree.

import { REAL_GLTF_ASSETS } from "./assetManifest.mjs";
import { REAL_GLTF_BEHAVIORAL_PROOFS } from "./proofs.mjs";

const manifestUrl = new URL("../reference-renders/gltf-real-behavioral/manifest.json", import.meta.url);
const manifest = JSON.parse(await Deno.readTextFile(manifestUrl));

/** @param {string} message */
function fail(message) {
  throw new Error(`[gltf-real-proof-check] ${message}`);
}

/**
 * @param {unknown} a
 * @param {unknown} b
 */
function sameJson(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

/**
 * @param {readonly Record<string, unknown>[]} items
 * @param {string} key
 */
function byKey(items, key) {
  const map = new Map();
  for (const item of items) {
    const value = item[key];
    if (map.has(value)) fail(`duplicate ${key}: ${value}`);
    map.set(value, item);
  }
  return map;
}

const assetsById = byKey(REAL_GLTF_ASSETS, "id");
const manifestByAssetId = byKey(manifest.assets ?? [], "assetId");
const proofsByAssetId = byKey(REAL_GLTF_BEHAVIORAL_PROOFS, "assetId");

if (manifest.kind !== "vitrum-real-gltf-behavioral-goldens") {
  fail(`unexpected manifest kind ${manifest.kind}`);
}
if (!sameJson(manifest.resolution, [64, 64])) fail("manifest resolution must match behavioral-gate W/H");
if (manifest.samplesPerPixel !== 8) fail("manifest samplesPerPixel must match behavioral-gate SPP");

for (const proof of REAL_GLTF_BEHAVIORAL_PROOFS) {
  const asset = assetsById.get(proof.assetId);
  if (!asset) fail(`proof assetId ${proof.assetId} is missing from REAL_GLTF_ASSETS`);

  const manifestAsset = manifestByAssetId.get(proof.assetId);
  if (!manifestAsset) fail(`manifest is missing proof asset ${proof.assetId}`);
  if (manifestAsset.label !== proof.label) fail(`${proof.assetId}: manifest label differs from proofs.mjs`);
  if (manifestAsset.goldenPath !== proof.goldenPath) fail(`${proof.assetId}: manifest goldenPath differs from proofs.mjs`);
  if (!sameJson(manifestAsset.thresholds, proof.thresholds)) fail(`${proof.assetId}: manifest thresholds differ from proofs.mjs`);
  if (manifestAsset.kind !== asset.kind) fail(`${proof.assetId}: manifest kind ${manifestAsset.kind} differs from asset kind ${asset.kind}`);

  const expectedRequired = asset.expect.requiredExtensions ?? [];
  if (!sameJson(manifestAsset.requiredExtensions ?? [], expectedRequired)) {
    fail(`${proof.assetId}: manifest requiredExtensions differ from asset expectations`);
  }

  const goldenUrl = new URL(`../../${proof.goldenPath}`, import.meta.url);
  const stat = await Deno.stat(goldenUrl);
  if (!stat.isFile || stat.size <= 8) fail(`${proof.assetId}: golden PNG is missing or empty`);
  const header = await Deno.readFile(goldenUrl);
  if (header[0] !== 0x89 || header[1] !== 0x50 || header[2] !== 0x4e || header[3] !== 0x47) {
    fail(`${proof.assetId}: golden file is not a PNG`);
  }
}

for (const assetId of manifestByAssetId.keys()) {
  if (!proofsByAssetId.has(assetId)) fail(`manifest asset ${assetId} is missing from proofs.mjs`);
}

console.log(`[gltf-real-proof-check] PASS (${REAL_GLTF_BEHAVIORAL_PROOFS.length} real glTF behavioral proofs)`);
