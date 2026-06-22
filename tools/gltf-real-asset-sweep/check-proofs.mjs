#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
// Verifies that the real glTF behavioral-proof metadata, manifest, and PNGs agree.

import { REAL_GLTF_ASSETS } from "./assetManifest.mjs";
import { REAL_GLTF_BEHAVIORAL_PROOFS } from "./proofs.mjs";

const REQUIRED_REAL_GLTF_PROOF_ROWS = [
  {
    assetId: "box-textured-glb",
    kind: "textured-glb",
    label: "pt/gltf-real-box-textured",
    requiredExtensions: [],
    goldenPath: "tools/reference-renders/gltf-real-behavioral/pt-gltf-real-box-textured.png",
    dznFullGoldenPath: "tools/reference-renders/gltf-real-behavioral-dzn-full/pt-gltf-real-box-textured.png",
  },
  {
    assetId: "cesium-milk-truck-draco",
    kind: "draco",
    label: "pt/gltf-real-draco",
    requiredExtensions: ["KHR_draco_mesh_compression"],
    goldenPath: "tools/reference-renders/gltf-real-behavioral/pt-gltf-real-draco.png",
    dznFullGoldenPath: "tools/reference-renders/gltf-real-behavioral-dzn-full/pt-gltf-real-draco.png",
  },
  {
    assetId: "meshopt-cube-real",
    kind: "meshopt",
    label: "pt/gltf-real-meshopt",
    requiredExtensions: ["KHR_meshopt_compression"],
    goldenPath: "tools/reference-renders/gltf-real-behavioral/pt-gltf-real-meshopt.png",
    dznFullGoldenPath: "tools/reference-renders/gltf-real-behavioral-dzn-full/pt-gltf-real-meshopt.png",
  },
];

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
const proofsByAssetId = byKey(REAL_GLTF_BEHAVIORAL_PROOFS, "assetId");

for (const required of REQUIRED_REAL_GLTF_PROOF_ROWS) {
  const asset = assetsById.get(required.assetId);
  if (!asset) fail(`missing required real glTF asset row: ${required.assetId}`);
  if (asset.kind !== required.kind) fail(`${required.assetId}: asset kind differs from required proof contract`);
  if (!sameJson(asset.expect?.requiredExtensions ?? [], required.requiredExtensions)) {
    fail(`${required.assetId}: asset requiredExtensions differ from required proof contract`);
  }

  const proof = proofsByAssetId.get(required.assetId);
  if (!proof) fail(`missing required real glTF proof row: ${required.assetId}`);
  if (proof.label !== required.label) fail(`${required.assetId}: proof label differs from required proof contract`);
  if (proof.goldenPath !== required.goldenPath) fail(`${required.assetId}: base goldenPath differs from required proof contract`);
  if (proof.variants?.["dzn-full"]?.goldenPath !== required.dznFullGoldenPath) {
    fail(`${required.assetId}: dzn-full goldenPath differs from required proof contract`);
  }
}

await checkManifest({
  label: "base",
  manifestPath: "../reference-renders/gltf-real-behavioral/manifest.json",
  variantId: null,
});
await checkManifest({
  label: "dzn-full",
  manifestPath: "../reference-renders/gltf-real-behavioral-dzn-full/manifest.json",
  variantId: "dzn-full",
});

for (const proof of REAL_GLTF_BEHAVIORAL_PROOFS) {
  if (!assetsById.has(proof.assetId)) fail(`proof assetId ${proof.assetId} is missing from REAL_GLTF_ASSETS`);
}
for (const asset of REAL_GLTF_ASSETS) {
  if (!proofsByAssetId.has(asset.id)) fail(`asset ${asset.id} is missing from REAL_GLTF_BEHAVIORAL_PROOFS`);
}

console.log(`[gltf-real-proof-check] PASS (${REAL_GLTF_BEHAVIORAL_PROOFS.length} real glTF behavioral proofs, base + dzn-full goldens)`);

async function checkManifest({ label, manifestPath, variantId }) {
  const manifestUrl = new URL(manifestPath, import.meta.url);
  const manifest = JSON.parse(await Deno.readTextFile(manifestUrl));
  const manifestByAssetId = byKey(manifest.assets ?? [], "assetId");

  if (manifest.kind !== "vitrum-real-gltf-behavioral-goldens") {
    fail(`${label}: unexpected manifest kind ${manifest.kind}`);
  }
  if ((manifest.goldenVariant ?? null) !== variantId) {
    fail(`${label}: manifest goldenVariant mismatch`);
  }
  if (!sameJson(manifest.resolution, [64, 64])) fail(`${label}: manifest resolution must match behavioral-gate W/H`);
  if (manifest.samplesPerPixel !== 8) fail(`${label}: manifest samplesPerPixel must match behavioral-gate SPP`);

  for (const proof of REAL_GLTF_BEHAVIORAL_PROOFS) {
    const asset = assetsById.get(proof.assetId);
    if (!asset) fail(`proof assetId ${proof.assetId} is missing from REAL_GLTF_ASSETS`);
    const selectedProof = variantId == null ? proof : proof.variants?.[variantId];
    if (!selectedProof) fail(`${label}: ${proof.assetId} missing ${variantId} proof variant`);

    const manifestAsset = manifestByAssetId.get(proof.assetId);
    if (!manifestAsset) fail(`${label}: manifest is missing proof asset ${proof.assetId}`);
    if (manifestAsset.label !== proof.label) fail(`${label}: ${proof.assetId}: manifest label differs from proofs.mjs`);
    if (manifestAsset.goldenPath !== selectedProof.goldenPath) fail(`${label}: ${proof.assetId}: manifest goldenPath differs from proofs.mjs`);
    if (!sameJson(manifestAsset.thresholds, selectedProof.thresholds)) fail(`${label}: ${proof.assetId}: manifest thresholds differ from proofs.mjs`);
    if (manifestAsset.kind !== asset.kind) fail(`${label}: ${proof.assetId}: manifest kind ${manifestAsset.kind} differs from asset kind ${asset.kind}`);

    const expectedRequired = asset.expect.requiredExtensions ?? [];
    if (!sameJson(manifestAsset.requiredExtensions ?? [], expectedRequired)) {
      fail(`${label}: ${proof.assetId}: manifest requiredExtensions differ from asset expectations`);
    }

    await assertPng(selectedProof.goldenPath, `${label}: ${proof.assetId}`);
  }

  for (const assetId of manifestByAssetId.keys()) {
    if (!proofsByAssetId.has(assetId)) fail(`${label}: manifest asset ${assetId} is missing from proofs.mjs`);
  }
}

async function assertPng(path, label) {
  const goldenUrl = new URL(`../../${path}`, import.meta.url);
  const stat = await Deno.stat(goldenUrl);
  if (!stat.isFile || stat.size <= 8) fail(`${label}: golden PNG is missing or empty`);
  const header = await Deno.readFile(goldenUrl);
  if (header[0] !== 0x89 || header[1] !== 0x50 || header[2] !== 0x4e || header[3] !== 0x47) {
    fail(`${label}: golden file is not a PNG`);
  }
}
