/**
 * Per-vertex COLOR_0 stream for walkaround material shading.
 *
 * One `rgba32float` texel = one vertex color. Missing authored colors are
 * already expanded to white by the shared-bvh packers, so shaders can multiply
 * the sampled value directly into baseColor/alpha without a presence bit.
 */

import {
  assertBvhTextureFitsDevice,
  normalizeBvhTextureTriangleCount as normalizeVertexCount,
  uploadElementTexture,
} from './bvhTextureLimits.js';

export const BVH_VERTEX_COLOR_TEX_WIDTH = 4096;

export interface VertexColorTexture {
  texture: GPUTexture;
  width: number;
  height: number;
}

function vertexColorTextureSize(vertexCount: number): { width: number; height: number } {
  const count = normalizeVertexCount(vertexCount);
  const width = Math.min(BVH_VERTEX_COLOR_TEX_WIDTH, count);
  const height = Math.ceil(count / width);
  return { width, height };
}

export function uploadVertexColorTexture(
  device: GPUDevice,
  colors: Float32Array,
  vertexCount: number,
): VertexColorTexture {
  const { width, height } = vertexColorTextureSize(vertexCount);
  assertBvhTextureFitsDevice('bvhVertexColor', device, width, height, vertexCount, 'vertex', 'vertex-color textures');
  const texture = uploadElementTexture(device, {
    label: 'vitrum.bvhVertexColors.rgba32float',
    format: 'rgba32float',
    width,
    height,
    bytesPerTexel: 16,
    elementsPerTexel: 4,
    makePadded: (n) => new Float32Array(n),
    fill: (padded) => {
      // Missing authored colors default to white (fill 1), then overwrite with
      // the authored prefix.
      padded.fill(1);
      padded.set(colors.subarray(0, Math.min(colors.length, normalizeVertexCount(vertexCount) * 4)));
    },
  });
  return { texture, width, height };
}
