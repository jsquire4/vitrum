/**
 * Per-vertex COLOR_0 stream for walkaround material shading.
 *
 * One `rgba32float` texel = one vertex color. Missing authored colors are
 * already expanded to white by the shared-bvh packers, so shaders can multiply
 * the sampled value directly into baseColor/alpha without a presence bit.
 */

import { maxTextureDimension2D } from './bvhTextureLimits.js';

export const BVH_VERTEX_COLOR_TEX_WIDTH = 4096;

const TEX_BINDING = 0x04;
const COPY_DST = 0x02;

export interface VertexColorTexture {
  texture: GPUTexture;
  width: number;
  height: number;
}

function normalizeVertexCount(vertexCount: number): number {
  if (!Number.isFinite(vertexCount)) return 1;
  return Math.max(1, Math.floor(vertexCount));
}

function vertexColorTextureSize(vertexCount: number): { width: number; height: number } {
  const count = normalizeVertexCount(vertexCount);
  const width = Math.min(BVH_VERTEX_COLOR_TEX_WIDTH, count);
  const height = Math.ceil(count / width);
  return { width, height };
}

function assertFitsDevice(device: GPUDevice, width: number, height: number, vertexCount: number): void {
  const maxDim = maxTextureDimension2D(device);
  if (width <= maxDim && height <= maxDim) return;
  throw new RangeError(
    `[vitrum/walkaround-hybrid] bvhVertexColor texture requires ${width}x${height} texels ` +
    `for ${normalizeVertexCount(vertexCount)} vertices, which exceeds ` +
    `device.limits.maxTextureDimension2D=${maxDim}. Reduce vertex count or split ` +
    'the scene before creating walkaround-hybrid vertex-color textures.',
  );
}

export function uploadVertexColorTexture(
  device: GPUDevice,
  colors: Float32Array,
  vertexCount: number,
): VertexColorTexture {
  const { width, height } = vertexColorTextureSize(vertexCount);
  assertFitsDevice(device, width, height, vertexCount);
  const texture = device.createTexture({
    label: 'vitrum.bvhVertexColors.rgba32float',
    size: { width, height, depthOrArrayLayers: 1 },
    format: 'rgba32float',
    usage: TEX_BINDING | COPY_DST,
  });
  const padded = new Float32Array(width * height * 4);
  padded.fill(1);
  padded.set(colors.subarray(0, Math.min(colors.length, normalizeVertexCount(vertexCount) * 4)));
  device.queue.writeTexture(
    { texture },
    padded.buffer,
    { bytesPerRow: width * 4 * 4, rowsPerImage: height },
    { width, height, depthOrArrayLayers: 1 },
  );
  return { texture, width, height };
}
