/**
 * Per-vertex authored/generated tangent stream for walkaround material maps.
 *
 * The scene bind group is already close to the WebGPU storage-buffer floor, so
 * tangents are carried as an `rgba32float` texture and read with `textureLoad`.
 * One texel = one vertex: xyz is the tangent direction, w is bitangent sign.
 * Zero tangents (0,0,0,0) intentionally mean "derive the frame from UVs".
 */

import {
  assertBvhTextureFitsDevice,
  normalizeBvhTextureTriangleCount as normalizeVertexCount,
  uploadElementTexture,
} from './bvhTextureLimits.js';

export const BVH_TANGENT_TEX_WIDTH = 4096;

export interface TangentTexture {
  texture: GPUTexture;
  width: number;
  height: number;
}

function tangentTextureSize(vertexCount: number): { width: number; height: number } {
  const count = normalizeVertexCount(vertexCount);
  const width = Math.min(BVH_TANGENT_TEX_WIDTH, count);
  const height = Math.ceil(count / width);
  return { width, height };
}

export function uploadTangentTexture(
  device: GPUDevice,
  tangents: Float32Array,
  vertexCount: number,
): TangentTexture {
  const { width, height } = tangentTextureSize(vertexCount);
  assertBvhTextureFitsDevice('bvhTangent', device, width, height, vertexCount, 'vertex', 'tangent textures');
  const texture = uploadElementTexture(device, {
    label: 'vitrum.bvhTangents.rgba32float',
    format: 'rgba32float',
    width,
    height,
    bytesPerTexel: 16,
    elementsPerTexel: 4,
    makePadded: (n) => new Float32Array(n),
    fill: (padded) => {
      padded.set(tangents.subarray(0, Math.min(tangents.length, normalizeVertexCount(vertexCount) * 4)));
    },
  });
  return { texture, width, height };
}
