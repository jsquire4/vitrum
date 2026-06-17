#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
// Verifies that the material-sweep behavioral-proof metadata, manifest, and PNG agree.

import { SWEEP_MAPS } from "./fixture.mjs";
import { GLTF_MATERIAL_SWEEP_BEHAVIORAL_PROOF } from "./proofs.mjs";

const manifestUrl = new URL("../reference-renders/gltf-material-sweep-behavioral/manifest.json", import.meta.url);
const manifest = JSON.parse(await Deno.readTextFile(manifestUrl));

function fail(message) {
  throw new Error(`[gltf-material-proof-check] ${message}`);
}

function sameJson(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
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
if (!sameJson(manifest.thresholds, GLTF_MATERIAL_SWEEP_BEHAVIORAL_PROOF.thresholds)) {
  fail("manifest thresholds differ from proofs.mjs");
}
if (manifest.materialMapCount !== SWEEP_MAPS.length) {
  fail(`manifest materialMapCount ${manifest.materialMapCount} differs from SWEEP_MAPS ${SWEEP_MAPS.length}`);
}

const goldenUrl = new URL(`../../${GLTF_MATERIAL_SWEEP_BEHAVIORAL_PROOF.goldenPath}`, import.meta.url);
const stat = await Deno.stat(goldenUrl);
if (!stat.isFile || stat.size <= 8) fail("golden PNG is missing or empty");
const header = await Deno.readFile(goldenUrl);
if (header[0] !== 0x89 || header[1] !== 0x50 || header[2] !== 0x4e || header[3] !== 0x47) {
  fail("golden file is not a PNG");
}

console.log("[gltf-material-proof-check] PASS (synthetic material-sweep behavioral proof)");
