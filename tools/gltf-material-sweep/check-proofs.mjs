#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
// Verifies that the material-sweep behavioral-proof metadata, manifest, and PNG agree.

import { FIELD_TEXTURE_INDEX, SWEEP_MAPS } from "./fixture.mjs";
import { GLTF_MATERIAL_SWEEP_BEHAVIORAL_PROOF } from "./proofs.mjs";

const REQUIRED_SWEEP_MAPS = [
  "baseColorMap",
  "roughnessMap",
  "metallicMap",
  "normalMap",
  "aoMap",
  "emissiveMap",
  "transmissionMap",
  "specularIntensityMap",
  "specularColorMap",
  "sheenColorMap",
  "sheenRoughnessMap",
  "clearcoatMap",
  "clearcoatRoughnessMap",
  "clearcoatNormalMap",
  "iridescenceMap",
  "iridescenceThicknessMap",
  "anisotropyMap",
  "thicknessMap",
];

const manifestUrl = new URL("../reference-renders/gltf-material-sweep-behavioral/manifest.json", import.meta.url);
const manifest = JSON.parse(await Deno.readTextFile(manifestUrl));

function fail(message) {
  throw new Error(`[gltf-material-proof-check] ${message}`);
}

function sameJson(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function readPngU32(bytes, offset) {
  return bytes[offset] * 0x1000000 + bytes[offset + 1] * 0x10000 + bytes[offset + 2] * 0x100 + bytes[offset + 3];
}

async function sha256Hex(bytes) {
  const owned = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(owned).set(bytes);
  const digest = await crypto.subtle.digest("SHA-256", owned);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function expectedPngIdentity(proof, label) {
  const width = Number(proof.width);
  const height = Number(proof.height);
  const sha256 = proof.sha256;
  if (!Number.isInteger(width) || width <= 0) fail(`${label}: proof row must declare a positive integer width`);
  if (!Number.isInteger(height) || height <= 0) fail(`${label}: proof row must declare a positive integer height`);
  if (typeof sha256 !== "string" || !/^[0-9a-f]{64}$/.test(sha256)) {
    fail(`${label}: proof row must declare a lowercase SHA-256 digest`);
  }
  return { width, height, sha256 };
}

async function assertPng(path, label, proof) {
  const expected = expectedPngIdentity(proof, label);
  const goldenUrl = new URL(`../../${path}`, import.meta.url);
  const stat = await Deno.stat(goldenUrl);
  if (!stat.isFile || stat.size <= 24) fail(`${label}: golden PNG is missing or empty`);
  const bytes = await Deno.readFile(goldenUrl);
  if (bytes[0] !== 0x89 || bytes[1] !== 0x50 || bytes[2] !== 0x4e || bytes[3] !== 0x47) {
    fail(`${label}: golden file is not a PNG`);
  }
  const width = readPngU32(bytes, 16);
  const height = readPngU32(bytes, 20);
  if (width !== expected.width || height !== expected.height) {
    fail(`${label}: golden PNG dimensions ${width}x${height} differ from proof ${expected.width}x${expected.height}`);
  }
  const actualSha256 = await sha256Hex(bytes);
  if (actualSha256 !== expected.sha256) {
    fail(`${label}: golden PNG SHA-256 ${actualSha256} differs from proof ${expected.sha256}`);
  }
}

if (manifest.kind !== "vitrum-gltf-material-sweep-behavioral-goldens") {
  fail(`unexpected manifest kind ${manifest.kind}`);
}
if (!sameJson(manifest.resolution, [64, 64])) fail("manifest resolution must match behavioral-gate W/H");
if (manifest.samplesPerPixel !== 8) fail("manifest samplesPerPixel must match behavioral-gate SPP");
if (manifest.fixture !== GLTF_MATERIAL_SWEEP_BEHAVIORAL_PROOF.fixture) {
  fail("manifest fixture differs from proofs.mjs");
}
if (manifest.label !== GLTF_MATERIAL_SWEEP_BEHAVIORAL_PROOF.label) {
  fail("manifest label differs from proofs.mjs");
}
if (manifest.goldenPath !== GLTF_MATERIAL_SWEEP_BEHAVIORAL_PROOF.goldenPath) {
  fail("manifest goldenPath differs from proofs.mjs");
}
if (manifest.dznStatusPath !== "tools/behavioral-gate/behavioral-gate-dzn-gltf-material-sweep-status.json") {
  fail("manifest dznStatusPath must point at the committed dzn status artifact");
}
if (!sameJson(manifest.thresholds, GLTF_MATERIAL_SWEEP_BEHAVIORAL_PROOF.thresholds)) {
  fail("manifest thresholds differ from proofs.mjs");
}
if (manifest.materialMapCount !== SWEEP_MAPS.length) {
  fail(`manifest materialMapCount ${manifest.materialMapCount} differs from SWEEP_MAPS ${SWEEP_MAPS.length}`);
}
const sweepMapSet = new Set(SWEEP_MAPS);
if (sweepMapSet.size !== SWEEP_MAPS.length) fail("SWEEP_MAPS contains duplicate fields");
for (const field of REQUIRED_SWEEP_MAPS) {
  if (!sweepMapSet.has(field)) fail(`SWEEP_MAPS is missing required material field ${field}`);
  if (!FIELD_TEXTURE_INDEX.has(field)) fail(`FIELD_TEXTURE_INDEX is missing required material field ${field}`);
}

await assertPng(GLTF_MATERIAL_SWEEP_BEHAVIORAL_PROOF.goldenPath, "material sweep golden", GLTF_MATERIAL_SWEEP_BEHAVIORAL_PROOF);

const dznStatusUrl = new URL(`../../${manifest.dznStatusPath}`, import.meta.url);
const dznStatus = JSON.parse(await Deno.readTextFile(dznStatusUrl));
if (dznStatus.harness !== "behavioral-gate:dzn") fail("dzn status harness mismatch");
if (dznStatus.verdict !== "PASS") fail(`dzn status verdict must be PASS, got ${dznStatus.verdict}`);
if (dznStatus.command !== manifest.commands.compareFullTierDzn) fail("dzn status command differs from manifest");
if (dznStatus.filter !== "gltf-material-sweep") fail("dzn status filter mismatch");
if (dznStatus.summary?.totalConfigs !== 1) fail("dzn status must contain exactly one selected config");
if (dznStatus.summary?.failures !== 0) fail("dzn status must have zero failures");
if (dznStatus.summary?.knownResiduals !== 0) fail("dzn status must have zero known residuals");

const dznConfig = dznStatus.configs?.[0];
if (dznConfig == null) fail("dzn status missing config row details");
if (dznConfig.label !== GLTF_MATERIAL_SWEEP_BEHAVIORAL_PROOF.label) fail("dzn config label mismatch");
if (dznConfig.verdict !== "PASS") fail("dzn config verdict mismatch");
if (dznConfig.rawStatus !== "OK") fail("dzn config rawStatus mismatch");
if (dznConfig.tier !== "full") fail("dzn config must resolve pt-webgpu full tier");
if (dznConfig.gpuErrors !== 0) fail("dzn config must report zero GPU errors");
if (dznConfig.nan !== false) fail("dzn config must report nan=false");
if (dznConfig.goldenStatus !== "ok") fail("dzn config golden status must be ok");
if (dznConfig.rmse > manifest.thresholds.maxRmse) fail("dzn config RMSE exceeds manifest threshold");
if (dznConfig.meanAbs > manifest.thresholds.maxMeanAbs) fail("dzn config meanAbs exceeds manifest threshold");
if (dznConfig.maxAbs > manifest.thresholds.maxAbs) fail("dzn config maxAbs exceeds manifest threshold");
if (!sameJson(dznConfig.thresholds, manifest.thresholds)) fail("dzn config thresholds differ from manifest");

console.log("[gltf-material-proof-check] PASS (synthetic material-sweep behavioral proof + dzn full-tier status)");
