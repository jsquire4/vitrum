#!/usr/bin/env -S deno run --sloppy-imports --allow-read --allow-net
// @ts-check
/**
 * Real glTF asset sweep.
 *
 * This is the real-asset counterpart to tools/gltf-material-sweep:
 * - loads public Khronos sample assets by URL (no vendored binaries);
 * - resolves external buffers/images through the adapter's fetch path;
 * - decodes PNG/JPEG textures through host-supplied pixel hooks;
 * - supplies real Draco and meshopt decoder hooks from installed packages;
 * - reports texture readiness, compatibility, warnings, and diagnostics.
 *
 * It intentionally stops at import/decode/compatibility proof, then reports the
 * matching behavioral-gate golden PNG lane when one is committed for the asset.
 */

import { loadGltfForEngine } from "@vitrum/gltf-adapter";
import {
  REAL_GLTF_ASSETS,
  decodeImagePixels,
  makeRealGltfDecodeHooks,
} from "./assets.mjs";
import { proofForRealGltfAsset } from "./proofs.mjs";

const jsonMode = Deno.args.includes("--json");
const selectedIds = new Set(readMultiFlag("--asset"));

function readMultiFlag(name) {
  const values = [];
  for (let i = 0; i < Deno.args.length; i += 1) {
    const arg = Deno.args[i];
    if (arg === name && Deno.args[i + 1]) values.push(Deno.args[i + 1]);
    if (arg?.startsWith(`${name}=`)) values.push(arg.slice(name.length + 1));
  }
  return values.flatMap((value) => value.split(",").map((s) => s.trim()).filter(Boolean));
}

function fail(message, details = {}) {
  const suffix = Object.keys(details).length > 0 ? `: ${JSON.stringify(details)}` : "";
  throw Object.assign(new Error(`${message}${suffix}`), { details });
}

function summarizeCompatibility(result) {
  return result.asset.backendCompatibility.map((entry) => ({
    backend: entry.backend,
    unsupportedCount: entry.unsupportedCount,
    approximateCount: entry.approximateCount,
    requiresHookCount: entry.requiresHookCount,
    isCompatible: entry.isCompatible,
  }));
}

function assertAssetExpectations(asset, result) {
  const primitiveCount = result.controller.scene.primitives.length;
  if (primitiveCount < (asset.expect.minPrimitives ?? 0)) {
    fail(`${asset.id}: primitive count below expectation`, { primitiveCount, expect: asset.expect });
  }
  const used = new Set(result.asset.featureReport.extensions.used);
  for (const ext of asset.expect.requiredExtensions ?? []) {
    if (!used.has(ext)) fail(`${asset.id}: expected extension missing`, { ext, used: [...used] });
  }
  if ((result.textureDecodeReport.mapCount ?? 0) < (asset.expect.minTextures ?? 0)) {
    fail(`${asset.id}: texture map count below expectation`, {
      mapCount: result.textureDecodeReport.mapCount,
      expect: asset.expect,
    });
  }
  if (result.textureDecodeDiagnostics.length > 0) {
    fail(`${asset.id}: texture decode diagnostics emitted`, { diagnostics: result.textureDecodeDiagnostics });
  }
  const allowedWarnings = asset.expect.allowedWarningSubstrings ?? [];
  const unexpectedWarnings = result.warnings.filter((warning) =>
    !allowedWarnings.some((needle) => warning.includes(needle))
  );
  if (unexpectedWarnings.length > 0) {
    fail(`${asset.id}: loader warnings emitted`, { warnings: unexpectedWarnings });
  }
}

async function sweepAsset(asset, hooks) {
  const result = await loadGltfForEngine(asset.url, {
    backend: "pt-webgpu",
    decodeTextures: true,
    textureTarget: "cpu-linear",
    decodePixels: decodeImagePixels,
    maxTextureSize: 4096,
    warnOnNpotRepeatWrap: true,
    dracoDecode: hooks.dracoDecode,
    meshoptDecode: hooks.meshoptDecode,
  });
  assertAssetExpectations(asset, result);
  const renderProof = proofForRealGltfAsset(asset.id);
  return {
    id: asset.id,
    kind: asset.kind,
    url: asset.url,
    generator: result.asset.featureReport.generator ?? "",
    primitiveCount: result.controller.scene.primitives.length,
    animationCount: result.asset.animations?.length ?? 0,
    extensionsUsed: result.asset.featureReport.extensions.used,
    extensionsRequired: result.asset.featureReport.extensions.required,
    textureDecodeReport: {
      mapCount: result.textureDecodeReport.mapCount,
      cpuReadableCount: result.textureDecodeReport.cpuReadableCount,
      rawImageCount: result.textureDecodeReport.rawImageCount,
      opaqueHandleCount: result.textureDecodeReport.opaqueHandleCount,
    },
    recommendedBackend: {
      backend: result.asset.recommendedBackend.backend,
      profileId: result.asset.recommendedBackend.profileId,
      unsupportedCount: result.asset.recommendedBackend.unsupportedCount,
      approximateCount: result.asset.recommendedBackend.approximateCount,
      isCompatible: result.asset.recommendedBackend.isCompatible,
    },
    selectedProfile: result.profileId,
    compatibility: summarizeCompatibility(result),
    warnings: result.warnings,
    diagnostics: result.diagnostics,
    renderStatus: renderProof ? "covered-by-behavioral-gate" : "queued",
    renderProof,
  };
}

const assets = selectedIds.size === 0 ? REAL_GLTF_ASSETS : REAL_GLTF_ASSETS.filter((asset) => selectedIds.has(asset.id));
if (assets.length === 0) {
  fail("no assets selected", { selected: [...selectedIds], available: REAL_GLTF_ASSETS.map((a) => a.id) });
}

const hooks = await makeRealGltfDecodeHooks();

const results = [];
for (const asset of assets) {
  if (!jsonMode) console.log(`[gltf-real-asset-sweep] loading ${asset.id}`);
  results.push(await sweepAsset(asset, hooks));
}

const summary = {
  assetCount: results.length,
  assets: results,
  renderStatus: results.every((item) => item.renderStatus === "covered-by-behavioral-gate")
    ? "covered-by-behavioral-gate"
    : "partial",
  renderQueueReason:
    "This harness proves real URL load/decode/compression hooks; renderProof points to the behavioral-gate golden PNG lane for assets with committed references.",
};

if (jsonMode) {
  console.log(JSON.stringify(summary, null, 2));
} else {
  console.log("[gltf-real-asset-sweep] PASS");
  for (const item of results) {
    console.log(
      `[gltf-real-asset-sweep] ${item.id}: prims=${item.primitiveCount}; ` +
        `maps=${item.textureDecodeReport.mapCount}; cpu=${item.textureDecodeReport.cpuReadableCount}; ` +
        `ext=${item.extensionsUsed.join(",") || "(none)"}`,
    );
  }
  console.log(`[gltf-real-asset-sweep] renderStatus=${summary.renderStatus}`);
  for (const item of results) {
    if (item.renderProof) {
      console.log(`[gltf-real-asset-sweep] ${item.id}: renderProof=${item.renderProof.label} golden=${item.renderProof.goldenPath}`);
    }
  }
}
