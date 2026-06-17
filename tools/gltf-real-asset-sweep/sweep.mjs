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
 * It intentionally stops at import/decode/compatibility proof. Render/golden PNG
 * capture remains a separate GPU harness step.
 */

import { loadGltfForEngine } from "@vitrum/gltf-adapter";
import { Buffer } from "node:buffer";
import draco3d from "npm:draco3d@1.5.7";
import jpeg from "npm:jpeg-js@0.4.4";
import { MeshoptDecoder } from "npm:meshoptimizer@1.1.1";
import { PNG } from "npm:pngjs@7.0.0";

const ASSETS = [
  {
    id: "box-textured-glb",
    kind: "textured-glb",
    url: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/BoxTextured/glTF-Binary/BoxTextured.glb",
    expect: { minPrimitives: 1, minTextures: 1 },
  },
  {
    id: "cesium-milk-truck-draco",
    kind: "draco",
    url: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/CesiumMilkTruck/glTF-Draco/CesiumMilkTruck.gltf",
    expect: { minPrimitives: 1, requiredExtensions: ["KHR_draco_mesh_compression"] },
  },
  {
    id: "meshopt-cube-real",
    kind: "meshopt",
    url: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/MeshoptCubeTest/glTF-Meshopt/MeshoptCubeTest.gltf",
    expect: {
      minPrimitives: 1,
      requiredExtensions: ["KHR_meshopt_compression"],
      allowedWarningSubstrings: ["sets doubleSided=true"],
    },
  },
];

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

async function decodeImagePixels(handle, context) {
  const isPng = handle.mimeType === "image/png" ||
    (handle.data[0] === 0x89 && handle.data[1] === 0x50 && handle.data[2] === 0x4e && handle.data[3] === 0x47);
  if (isPng) {
    const png = PNG.sync.read(Buffer.from(handle.data));
    return {
      width: png.width,
      height: png.height,
      channels: 4,
      dataType: "uint8",
      colorSpace: context.colorSpace,
      data: png.data,
    };
  }

  const isJpeg = handle.mimeType === "image/jpeg" ||
    (handle.data[0] === 0xff && handle.data[1] === 0xd8 && handle.data[handle.data.length - 2] === 0xff &&
      handle.data[handle.data.length - 1] === 0xd9);
  if (isJpeg) {
    const jpg = jpeg.decode(Buffer.from(handle.data), { useTArray: true });
    return {
      width: jpg.width,
      height: jpg.height,
      channels: 4,
      dataType: "uint8",
      colorSpace: context.colorSpace,
      data: jpg.data,
    };
  }

  throw new Error(`unsupported image MIME "${handle.mimeType}"; real-asset sweep decodes PNG/JPEG textures`);
}

async function makeDracoDecode() {
  const module = await draco3d.createDecoderModule({});
  return (compressed, attributeIds) => {
    const decoder = new module.Decoder();
    const buffer = new module.DecoderBuffer();
    let geometry = null;
    try {
      buffer.Init(new Int8Array(compressed.buffer, compressed.byteOffset, compressed.byteLength), compressed.byteLength);
      const geometryType = decoder.GetEncodedGeometryType(buffer);
      if (geometryType !== module.TRIANGULAR_MESH) {
        throw new Error(`unsupported Draco geometry type ${geometryType}`);
      }
      geometry = new module.Mesh();
      const status = decoder.DecodeBufferToMesh(buffer, geometry);
      if (!status.ok()) throw new Error(status.error_msg());

      const attributes = {};
      for (const [semantic, uniqueId] of Object.entries(attributeIds)) {
        const attribute = decoder.GetAttributeByUniqueId(geometry, Number(uniqueId));
        if (!attribute || attribute.ptr === 0) {
          throw new Error(`missing Draco attribute ${semantic} (unique id ${uniqueId})`);
        }
        const values = new module.DracoFloat32Array();
        try {
          decoder.GetAttributeFloatForAllPoints(geometry, attribute, values);
          const out = new Float32Array(values.size());
          for (let i = 0; i < out.length; i += 1) out[i] = values.GetValue(i);
          attributes[semantic] = out;
        } finally {
          module.destroy(values);
        }
      }

      const face = new module.DracoInt32Array();
      const indices = new Uint32Array(geometry.num_faces() * 3);
      try {
        for (let f = 0; f < geometry.num_faces(); f += 1) {
          decoder.GetFaceFromMesh(geometry, f, face);
          const off = f * 3;
          indices[off] = face.GetValue(0);
          indices[off + 1] = face.GetValue(1);
          indices[off + 2] = face.GetValue(2);
        }
      } finally {
        module.destroy(face);
      }

      return { attributes, indices };
    } finally {
      module.destroy(buffer);
      if (geometry) module.destroy(geometry);
      module.destroy(decoder);
    }
  };
}

async function makeMeshoptDecode() {
  await MeshoptDecoder.ready;
  return async (compressed, count, byteStride, mode, filter) => {
    if (typeof MeshoptDecoder.decodeGltfBufferAsync === "function") {
      return MeshoptDecoder.decodeGltfBufferAsync(
        count,
        byteStride,
        compressed,
        mode,
        filter === "NONE" ? undefined : filter,
      );
    }
    const target = new Uint8Array(count * byteStride);
    MeshoptDecoder.decodeGltfBuffer(
      target,
      count,
      byteStride,
      compressed,
      mode,
      filter === "NONE" ? undefined : filter,
    );
    return target;
  };
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
    renderStatus: "queued",
  };
}

const assets = selectedIds.size === 0 ? ASSETS : ASSETS.filter((asset) => selectedIds.has(asset.id));
if (assets.length === 0) fail("no assets selected", { selected: [...selectedIds], available: ASSETS.map((a) => a.id) });

const hooks = {
  dracoDecode: await makeDracoDecode(),
  meshoptDecode: await makeMeshoptDecode(),
};

const results = [];
for (const asset of assets) {
  if (!jsonMode) console.log(`[gltf-real-asset-sweep] loading ${asset.id}`);
  results.push(await sweepAsset(asset, hooks));
}

const summary = {
  assetCount: results.length,
  assets: results,
  renderStatus: "queued",
  renderQueueReason:
    "This harness proves real URL load/decode/compression hooks. Golden PNG rendering remains a separate GPU capture queue item.",
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
  console.log("[gltf-real-asset-sweep] renderStatus=queued (GPU golden capture remains separate)");
}
