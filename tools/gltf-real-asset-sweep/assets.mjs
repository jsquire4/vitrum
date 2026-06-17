// @ts-check
// Shared manifest and host decode hooks for real glTF URL sweeps/gates.

import { Buffer } from "node:buffer";
import draco3d from "npm:draco3d@1.5.7";
import jpeg from "npm:jpeg-js@0.4.4";
import { MeshoptDecoder } from "npm:meshoptimizer@1.1.1";
import { PNG } from "npm:pngjs@7.0.0";
export { REAL_GLTF_ASSETS, getRealGltfAsset } from "./assetManifest.mjs";

export async function decodeImagePixels(handle, context) {
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

export async function makeRealGltfDecodeHooks() {
  const [dracoDecode, meshoptDecode] = await Promise.all([
    makeDracoDecode(),
    makeMeshoptDecode(),
  ]);
  return { dracoDecode, meshoptDecode, decodePixels: decodeImagePixels };
}
