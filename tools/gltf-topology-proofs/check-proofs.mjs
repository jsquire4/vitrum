#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
// Verifies that glTF topology behavioral-proof metadata, manifests, and PNGs agree.

import { GLTF_TOPOLOGY_BEHAVIORAL_PROOFS } from "./proofs.mjs";

function fail(message) {
  throw new Error(`[gltf-topology-proof-check] ${message}`);
}

function sameJson(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function byKey(items, key) {
  const map = new Map();
  for (const item of items) {
    const value = item[key];
    if (map.has(value)) fail(`duplicate ${key}: ${value}`);
    map.set(value, item);
  }
  return map;
}

const proofsById = byKey(GLTF_TOPOLOGY_BEHAVIORAL_PROOFS, "id");
const labels = new Set();
for (const proof of GLTF_TOPOLOGY_BEHAVIORAL_PROOFS) {
  if (labels.has(proof.label)) fail(`duplicate label: ${proof.label}`);
  labels.add(proof.label);

  const manifestUrl = new URL(`../../${proof.manifestPath}`, import.meta.url);
  const manifest = JSON.parse(await Deno.readTextFile(manifestUrl));
  if (manifest.kind !== proof.kind) fail(`${proof.id}: unexpected manifest kind ${manifest.kind}`);
  if (manifest.id !== proof.id) fail(`${proof.id}: manifest id differs from proofs.mjs`);
  if (manifest.label !== proof.label) fail(`${proof.id}: manifest label differs from proofs.mjs`);
  if (manifest.fixture !== proof.fixture) fail(`${proof.id}: manifest fixture differs from proofs.mjs`);
  if (manifest.proof !== proof.proof) fail(`${proof.id}: manifest proof differs from proofs.mjs`);
  if (manifest.goldenPath !== proof.goldenPath) fail(`${proof.id}: manifest goldenPath differs from proofs.mjs`);
  if (!sameJson(manifest.sourceModes, proof.sourceModes)) fail(`${proof.id}: manifest sourceModes differ from proofs.mjs`);
  if (!sameJson(manifest.thresholds, proof.thresholds)) fail(`${proof.id}: manifest thresholds differ from proofs.mjs`);
  if (!sameJson(manifest.resolution, [64, 64])) fail(`${proof.id}: manifest resolution must match behavioral-gate W/H`);
  if (manifest.samplesPerPixel !== 8) fail(`${proof.id}: manifest samplesPerPixel must match behavioral-gate SPP`);

  const goldenUrl = new URL(`../../${proof.goldenPath}`, import.meta.url);
  const stat = await Deno.stat(goldenUrl);
  if (!stat.isFile || stat.size <= 8) fail(`${proof.id}: golden PNG is missing or empty`);
  const header = await Deno.readFile(goldenUrl);
  if (header[0] !== 0x89 || header[1] !== 0x50 || header[2] !== 0x4e || header[3] !== 0x47) {
    fail(`${proof.id}: golden file is not a PNG`);
  }
}

for (const proofId of proofsById.keys()) {
  if (!proofForId(proofId)) fail(`internal proof lookup failed for ${proofId}`);
}

function proofForId(id) {
  return GLTF_TOPOLOGY_BEHAVIORAL_PROOFS.find((proof) => proof.id === id) ?? null;
}

console.log(`[gltf-topology-proof-check] PASS (${GLTF_TOPOLOGY_BEHAVIORAL_PROOFS.length} glTF topology behavioral proofs)`);
