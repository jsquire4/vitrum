/**
 * Camera-visible emitters (2026-05-30) — per-triangle HDR emissive radiance Le
 * stored as an `rgba32float` texture, the real-time analogue of the pt-webgpu
 * camera-visible-emitters fix.
 *
 * Motivation: emissive-mesh surfaces are NEE-only in walkaround (the ReSTIR-DI
 * emitter list); they render BLACK when viewed directly by the camera because
 * `shade.wgsl` only self-emits for glass (Beer-Lambert `lo_emit`). This buffer
 * carries each triangle's emissive Le so `shade.wgsl lo_emitterGlow` can add it
 * on a primary hit, making emitters camera-visible.
 *
 * Format: `rgba32float` (rgb = Le, a = mesh-NEE ownership, exactly 0 or 1).
 * Emissive is HDR (emissiveIntensity
 * may exceed 1), so unlike the LDR `bvh_beer` u32 texture this needs float
 * storage. `rgba32float` is chosen over `rgba16float` so `writeTexture` can take
 * the packed `Float32Array` directly with no host-side f16 encode. The texture
 * uses `unfilterable-float` sampleType (`tex` in the descriptor table) and is
 * read via `textureLoad` (no sampler).
 *
 * Layout mirrors `bvhBeerTexture`: fixed width {@link BVH_EMISSIVE_TEX_WIDTH},
 * `ceil(triCount / width)` rows; triangle index → texel `vec2u(tri % W, tri / W)`.
 */

import {
  assertBvhTextureFitsDevice,
  assertBvhTextureRefreshCapacity,
  normalizeBvhTextureTriangleCount,
  uploadElementTexture,
  writeElementTexture,
} from './bvhTextureLimits.js';

/** Fixed texture width (power-of-two ≤ 8192 WebGPU floor). Shader uses the SAME. */
const BVH_EMISSIVE_TEX_WIDTH = 4096;

export interface EmissiveTexture {
  texture: GPUTexture;
  width: number;
  height: number;
}

/** Compute the rgba32float emissive-texture dimensions for `triCount` triangles. */
function emissiveTextureSize(triCount: number): { width: number; height: number } {
  const count = normalizeBvhTextureTriangleCount(triCount);
  const width = Math.min(BVH_EMISSIVE_TEX_WIDTH, count);
  const height = Math.ceil(count / width);
  return { width, height };
}

/**
 * Create + upload the emissive texture from the packed per-triangle Le data
 * (`packBVHEmissiveLe` output: 4 floats per triangle, rgb + ownership). `triCount`
 * is the number of valid triangles; the last texture row is zero-padded.
 */
export function uploadEmissiveTexture(
  device: GPUDevice,
  emissiveData: Float32Array,
  triCount: number,
): EmissiveTexture {
  const { width, height } = emissiveTextureSize(triCount);
  assertBvhTextureFitsDevice('bvhEmissive', device, width, height, triCount);
  const texture = uploadElementTexture(device, {
    label: 'vitrum.bvhEmissive.rgba32float',
    format: 'rgba32float',
    width,
    height,
    bytesPerTexel: 16,
    elementsPerTexel: 4,
    makePadded: (n) => new Float32Array(n),
    fill: (padded) => fillEmissive(padded, emissiveData, triCount),
  });
  return { texture, width, height };
}

/** Re-upload the full emissive texture (material-slice fast path; see bvhBeerTexture). */
export function refreshEmissiveTexture(
  device: GPUDevice,
  tex: EmissiveTexture,
  emissiveData: Float32Array,
  triCount: number,
): void {
  assertBvhTextureRefreshCapacity('bvhEmissive', tex.width, tex.height, triCount);
  writeElementTexture(device, tex.texture, {
    width: tex.width,
    height: tex.height,
    bytesPerTexel: 16,
    elementsPerTexel: 4,
    makePadded: (n) => new Float32Array(n),
    fill: (padded) => fillEmissive(padded, emissiveData, triCount),
  });
}

/** Populate the padded rgba32float grid (source has 4 floats/triangle). */
function fillEmissive(padded: Float32Array, emissiveData: Float32Array, triCount: number): void {
  padded.set(emissiveData.subarray(0, Math.min(emissiveData.length, triCount * 4)));
}
