#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
/**
 * glTF material sweep preflight.
 *
 * This is the CPU-side Phase 5A gate: it proves the one-call glTF path preserves
 * material texture fields, textureDecodeReport rows, backend compatibility
 * diagnostics, and CPU-linear decode readiness before any renderer captures run.
 *
 * The Road's render/golden-PNG half remains a separate GPU capture queue item.
 */

import {
  analyzeGltfAsset,
  evaluateGltfBackendCompatibility,
  loadGltfAndDecodeTextures,
  rankGltfBackends,
} from "@vitrum/gltf-adapter";

const jsonMode = Deno.args.includes("--json");

const SWEEP_MAPS = [
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

const WALKAROUND_ATLAS_FIELDS = new Set([
  "baseColorMap",
  "normalMap",
  "roughnessMap",
  "metallicMap",
  "aoMap",
  "alphaMap",
  "emissiveMap",
  "transmissionMap",
  "thicknessMap",
  "lightMap",
  "specularColorMap",
  "specularIntensityMap",
  "clearcoatMap",
  "clearcoatRoughnessMap",
  "clearcoatNormalMap",
  "bumpMap",
  "sheenColorMap",
  "sheenRoughnessMap",
  "anisotropyMap",
  "iridescenceMap",
  "iridescenceThicknessMap",
]);

function f32Buffer(values) {
  const buf = new ArrayBuffer(values.length * 4);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setFloat32(i * 4, v, true));
  return buf;
}

function concat(buffers) {
  const total = buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const buffer of buffers) {
    out.set(new Uint8Array(buffer), offset);
    offset += buffer.byteLength;
  }
  return out.buffer;
}

function texInfo(index) {
  return {
    index,
    texCoord: 0,
    extensions: {
      KHR_texture_transform: {
        texCoord: 1,
        offset: [0.01 * (index + 1), 0.02 * (index + 1)],
        scale: [1 + 0.1 * (index + 1), 2 + 0.1 * (index + 1)],
        rotation: 0.001 * (index + 1),
      },
    },
  };
}

function makeSweepGltf() {
  const positions = f32Buffer([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const imageBytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47]);
  const buffer = concat([positions, imageBytes.buffer]);
  const imageOffset = positions.byteLength;
  const textureCount = 17;

  return {
    buffers: new Map([[0, buffer]]),
    gltf: {
      asset: { version: "2.0", generator: "vitrum-gltf-material-sweep" },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
      materials: [{
        pbrMetallicRoughness: {
          baseColorTexture: texInfo(0),
          metallicRoughnessTexture: texInfo(1),
        },
        normalTexture: { ...texInfo(2), scale: 0.5 },
        occlusionTexture: { ...texInfo(3), strength: 0.75 },
        emissiveFactor: [1, 1, 1],
        emissiveTexture: texInfo(4),
        extensions: {
          KHR_materials_transmission: {
            transmissionFactor: 0.8,
            transmissionTexture: texInfo(5),
          },
          KHR_materials_specular: {
            specularFactor: 0.5,
            specularTexture: texInfo(6),
            specularColorFactor: [0.9, 0.8, 0.7],
            specularColorTexture: texInfo(7),
          },
          KHR_materials_sheen: {
            sheenColorFactor: [0.5, 0.3, 0.1],
            sheenColorTexture: texInfo(8),
            sheenRoughnessFactor: 0.4,
            sheenRoughnessTexture: texInfo(9),
          },
          KHR_materials_clearcoat: {
            clearcoatFactor: 0.8,
            clearcoatTexture: texInfo(10),
            clearcoatRoughnessFactor: 0.1,
            clearcoatRoughnessTexture: texInfo(11),
            clearcoatNormalTexture: { ...texInfo(12), scale: 0.25 },
          },
          KHR_materials_iridescence: {
            iridescenceFactor: 0.7,
            iridescenceTexture: texInfo(13),
            iridescenceIor: 2.0,
            iridescenceThicknessMinimum: 200,
            iridescenceThicknessMaximum: 800,
            iridescenceThicknessTexture: texInfo(14),
          },
          KHR_materials_anisotropy: {
            anisotropyStrength: 0.6,
            anisotropyRotation: 1.0,
            anisotropyTexture: texInfo(15),
          },
          KHR_materials_volume: {
            thicknessFactor: 0.5,
            thicknessTexture: texInfo(16),
            attenuationDistance: 2.0,
          },
        },
      }],
      textures: Array.from({ length: textureCount }, (_, i) => ({ source: 0, sampler: i })),
      samplers: Array.from({ length: textureCount }, (_, i) => ({
        wrapS: i % 3 === 1 ? 33071 : i % 3 === 2 ? 33648 : undefined,
        wrapT: i % 3 === 0 ? undefined : i % 3 === 1 ? 33648 : 33071,
        magFilter: i % 2 === 0 ? 9728 : 9729,
        minFilter: [9728, 9729, 9984, 9985, 9986, 9987][i % 6],
      })),
      images: [{ bufferView: 1, mimeType: "image/png" }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: "VEC3" }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: positions.byteLength },
        { buffer: 0, byteOffset: imageOffset, byteLength: imageBytes.byteLength },
      ],
      buffers: [{ byteLength: buffer.byteLength }],
    },
  };
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
    decodeImage: (bytes, mimeType) => ({ kind: "raw-image", mimeType, data: bytes }),
    decodePixels: (_handle, context) => ({
      width: 4,
      height: 4,
      channels: 4,
      dataType: "uint8",
      colorSpace: context.colorSpace,
      data: new Uint8Array(4 * 4 * 4).fill(255),
    }),
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
  for (const field of SWEEP_MAPS) {
    const entry = entriesByField.get(field);
    if (!entry) continue;
    if (entry.texCoord !== 1) {
      badReadiness.push({ field, reason: `expected texCoord 1, got ${entry.texCoord}` });
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
    compatibility: summarizeCompatibility(assetReport),
    renderStatus: "queued",
    renderQueueReason:
      "Phase 5A GPU render/golden-PNG capture is intentionally outside this CPU preflight; use behavioral-gate/reference-render harness for captures.",
  };

  if (jsonMode) {
    console.log(JSON.stringify(summary, null, 2));
  } else {
    console.log("[gltf-material-sweep] PASS");
    console.log(`[gltf-material-sweep] maps=${summary.textureDecodeReport.mapCount}; recommended=${summary.recommendedBackend.backend}/${summary.recommendedBackend.profileId}`);
    console.log("[gltf-material-sweep] renderStatus=queued (GPU golden capture remains a separate proof queue item)");
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
