#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
/**
 * glTF material sweep preflight.
 *
 * This is the CPU-side Phase 5A gate: it proves the one-call glTF path preserves
 * material texture fields, textureDecodeReport rows, backend compatibility
 * diagnostics, and CPU-linear decode readiness before any renderer captures run.
 *
 * The matching WSL/lavapipe pt-webgpu behavioral golden is tracked in
 * proofs.mjs; full-tier rich-material fidelity captures remain a separate queue.
 */

import {
  analyzeGltfAsset,
  evaluateGltfBackendCompatibility,
  loadGltfAndDecodeTextures,
  rankGltfBackends,
} from "@vitrum/gltf-adapter";
import {
  FIELD_TEXTURE_INDEX,
  SWEEP_MAPS,
  WALKAROUND_ATLAS_FIELDS,
  expectedSamplerPolicy,
  makeSweepGltf,
  makeSweepTextureDecodeHooks,
  samplerPolicyIsNativeForBackend,
} from "./fixture.mjs";
import { GLTF_MATERIAL_SWEEP_BEHAVIORAL_PROOF } from "./proofs.mjs";

const jsonMode = Deno.args.includes("--json");

function assertExpectedSamplerCompatibility(compatibility) {
  const mismatches = [];
  for (const backendSummary of compatibility) {
    const actual = new Set(
      backendSummary.issues
        .filter((issue) => issue.name.endsWith(".samplerPolicy"))
        .map((issue) => `${issue.name}:${issue.support}:${issue.path}`),
    );
    const expected = new Set();
    for (const field of SWEEP_MAPS) {
      const textureIndex = FIELD_TEXTURE_INDEX.get(field);
      if (textureIndex === undefined) continue;
      const policy = expectedSamplerPolicy(textureIndex);
      if (!samplerPolicyIsNativeForBackend(backendSummary.backend, policy)) {
        expected.add(`${field}.samplerPolicy:approximate:samplers[${textureIndex}].minFilter`);
      }
    }

    for (const item of expected) {
      if (!actual.has(item)) mismatches.push({ backend: backendSummary.backend, missing: item });
    }
    for (const item of actual) {
      if (!expected.has(item)) mismatches.push({ backend: backendSummary.backend, unexpected: item });
    }
  }
  if (mismatches.length > 0) {
    fail("backend sampler-policy diagnostics mismatch", { mismatches });
  }
}

function fail(message, details = {}) {
  throw Object.assign(new Error(message), { details });
}

function summarizeCompatibility(report) {
  return ["pt-webgl2", "pt-webgpu", "walkaround-hybrid"].map((backend) => {
    const compatibility = evaluateGltfBackendCompatibility(report, backend);
    return {
      backend,
      unsupportedCount: compatibility.unsupportedCount,
      approximateCount: compatibility.approximateCount,
      nativeCount: compatibility.nativeCount,
      requiresHookCount: compatibility.requiresHookCount,
      isCompatible: compatibility.isCompatible,
      issues: compatibility.issues.map((issue) => ({
        category: issue.category,
        name: issue.name,
        support: issue.support,
        path: issue.path,
      })),
    };
  });
}

async function main() {
  const { gltf, buffers } = makeSweepGltf();
  const assetReport = analyzeGltfAsset(gltf);
  const ranked = rankGltfBackends(assetReport, "fidelity");
  if (ranked.length === 0) fail("rankGltfBackends returned no candidates");

  const decoded = await loadGltfAndDecodeTextures(gltf, {
    buffers,
    backendPolicy: "fidelity",
    maxTextureSize: 8,
    warnOnNpotRepeatWrap: true,
    ...makeSweepTextureDecodeHooks(),
  });

  if (decoded.textureDecodeDiagnostics.length !== 0) {
    fail("texture decode emitted diagnostics for the material sweep", {
      diagnostics: decoded.textureDecodeDiagnostics,
    });
  }

  const entriesByField = new Map(decoded.textureDecodeReport.entries.map((entry) => [entry.materialField, entry]));
  const missing = SWEEP_MAPS.filter((field) => !entriesByField.has(field));
  if (missing.length > 0) fail("textureDecodeReport missed material fields", { missing });

  const badReadiness = [];
  const badSamplerPolicy = [];
  for (const field of SWEEP_MAPS) {
    const entry = entriesByField.get(field);
    if (!entry) continue;
    if (entry.texCoord !== 1) {
      badReadiness.push({ field, reason: `expected texCoord 1, got ${entry.texCoord}` });
    }
    if (entry.hasTransform !== true) {
      badSamplerPolicy.push({ field, reason: "expected KHR_texture_transform to survive in the decode report" });
    }
    const textureIndex = FIELD_TEXTURE_INDEX.get(field);
    if (textureIndex === undefined) {
      badSamplerPolicy.push({ field, reason: "material field is missing from FIELD_TEXTURE_INDEX" });
    } else {
      const expected = expectedSamplerPolicy(textureIndex);
      for (const key of ["wrapS", "wrapT", "magFilter", "minFilter", "mipFilter", "usesMipmaps"]) {
        if (entry[key] !== expected[key]) {
          badSamplerPolicy.push({
            field,
            textureIndex,
            key,
            expected: expected[key],
            actual: entry[key],
          });
        }
      }
    }
    if (entry.backendReadiness.ptWebgl2 !== "ready" || entry.backendReadiness.ptWebgpu !== "ready") {
      badReadiness.push({ field, reason: "decoded CPU texture was not ready for PT backends", readiness: entry.backendReadiness });
    }
    const expectedWalkaround = WALKAROUND_ATLAS_FIELDS.has(field) ? "ready" : "ignored";
    if (entry.backendReadiness.walkaroundHybrid !== expectedWalkaround) {
      badReadiness.push({
        field,
        reason: `walkaround readiness expected ${expectedWalkaround}`,
        readiness: entry.backendReadiness,
      });
    }
  }
  if (badReadiness.length > 0) fail("textureDecodeReport backend readiness mismatch", { badReadiness });
  if (badSamplerPolicy.length > 0) fail("textureDecodeReport sampler policy mismatch", { badSamplerPolicy });
  const compatibility = summarizeCompatibility(assetReport);
  assertExpectedSamplerCompatibility(compatibility);

  const proof = GLTF_MATERIAL_SWEEP_BEHAVIORAL_PROOF;
  const summary = {
    fixture: "synthetic-material-sweep",
    materialFields: decoded.featureReport.materials.materialFields,
    textureDecodeReport: {
      mapCount: decoded.textureDecodeReport.mapCount,
      uniqueHandleCount: decoded.textureDecodeReport.uniqueHandleCount,
      rawImageCount: decoded.textureDecodeReport.rawImageCount,
      cpuReadableCount: decoded.textureDecodeReport.cpuReadableCount,
      fields: decoded.textureDecodeReport.entries.map((entry) => ({
        field: entry.materialField,
        colorSpace: entry.colorSpace,
        texCoord: entry.texCoord,
        hasTransform: entry.hasTransform,
        wrapS: entry.wrapS,
        wrapT: entry.wrapT,
        magFilter: entry.magFilter,
        minFilter: entry.minFilter,
        mipFilter: entry.mipFilter,
        usesMipmaps: entry.usesMipmaps,
        readiness: entry.backendReadiness,
      })),
    },
    recommendedBackend: {
      backend: decoded.recommendedBackend.backend,
      profileId: decoded.recommendedBackend.profileId,
      unsupportedCount: decoded.recommendedBackend.unsupportedCount,
      approximateCount: decoded.recommendedBackend.approximateCount,
      isCompatible: decoded.recommendedBackend.isCompatible,
    },
    compatibility,
    renderStatus: "covered-by-behavioral-gate",
    renderProof: {
      label: proof.label,
      goldenPath: proof.goldenPath,
      thresholds: proof.thresholds,
    },
    renderQueueReason:
      "Committed WSL/lavapipe pt-webgpu behavioral golden covers boot/readback/tolerance; full-tier rich-material fidelity captures remain queued separately.",
  };

  if (jsonMode) {
    console.log(JSON.stringify(summary, null, 2));
  } else {
    console.log("[gltf-material-sweep] PASS");
    console.log(`[gltf-material-sweep] maps=${summary.textureDecodeReport.mapCount}; recommended=${summary.recommendedBackend.backend}/${summary.recommendedBackend.profileId}`);
    console.log(`[gltf-material-sweep] renderStatus=${summary.renderStatus}`);
    console.log(`[gltf-material-sweep] proof=${proof.label}; golden=${proof.goldenPath}`);
  }
}

try {
  await main();
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`[gltf-material-sweep] FAIL: ${message}`);
  if (error && typeof error === "object" && "details" in error) {
    console.error(JSON.stringify(error.details, null, 2));
  }
  Deno.exit(1);
}
