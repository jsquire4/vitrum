/**
 * Per-vertex authored/generated tangent stream for walkaround material maps.
 *
 * The scene bind group is already close to the WebGPU storage-buffer floor, so
 * tangents are carried as an `rgba32float` texture and read with `textureLoad`.
 * One texel = one vertex: xyz is the tangent direction, w is bitangent sign.
 * Zero tangents (0,0,0,0) intentionally mean "derive the frame from UVs".
 */

import { maxTextureDimension2D } from './bvhTextureLimits.js';

export const BVH_TANGENT_TEX_WIDTH = 4096;

const TEX_BINDING = 0x04;
const COPY_DST = 0x02;

export interface TangentTexture {
  texture: GPUTexture;
  width: number;
  height: number;
}

function normalizeVertexCount(vertexCount: number): number {
  if (!Number.isFinite(vertexCount)) return 1;
  return Math.max(1, Math.floor(vertexCount));
}

function tangentTextureSize(vertexCount: number): { width: number; height: number } {
  const count = normalizeVertexCount(vertexCount);
  const width = Math.min(BVH_TANGENT_TEX_WIDTH, count);
  const height = Math.ceil(count / width);
  return { width, height };
}

function assertFitsDevice(device: GPUDevice, width: number, height: number, vertexCount: number): void {
  const maxDim = maxTextureDimension2D(device);
  if (width <= maxDim && height <= maxDim) return;
  throw new RangeError(
    `[vitrum/walkaround-hybrid] bvhTangent texture requires ${width}x${height} texels ` +
    `for ${normalizeVertexCount(vertexCount)} vertices, which exceeds ` +
    `device.limits.maxTextureDimension2D=${maxDim}. Reduce vertex count or split ` +
    'the scene before creating walkaround-hybrid tangent textures.',
  );
}

export function uploadTangentTexture(
  device: GPUDevice,
  tangents: Float32Array,
  vertexCount: number,
): TangentTexture {
  const { width, height } = tangentTextureSize(vertexCount);
  assertFitsDevice(device, width, height, vertexCount);
  const texture = device.createTexture({
    label: 'vitrum.bvhTangents.rgba32float',
    size: { width, height, depthOrArrayLayers: 1 },
    format: 'rgba32float',
    usage: TEX_BINDING | COPY_DST,
  });
  writeTangentTexture(device, texture, tangents, vertexCount, width, height);
  return { texture, width, height };
}

function writeTangentTexture(
  device: GPUDevice,
  texture: GPUTexture,
  tangents: Float32Array,
  vertexCount: number,
  width: number,
  height: number,
): void {
  const padded = new Float32Array(width * height * 4);
  padded.set(tangents.subarray(0, Math.min(tangents.length, normalizeVertexCount(vertexCount) * 4)));
  device.queue.writeTexture(
    { texture },
    padded.buffer,
    { bytesPerRow: width * 4 * 4, rowsPerImage: height },
    { width, height, depthOrArrayLayers: 1 },
  );
}
