/**
 * WS1 (feature-completeness wave, 2026-05-29) — per-triangle Beer-Lambert
 * visible-color storage as an `r32uint` texture instead of a storage buffer.
 *
 * Motivation: the walkaround scene bind group sits exactly at the full-tier
 * `maxStorageBuffersPerShaderStage = 16` floor on the shade pass (4 frame + 11
 * scene + 1 RC cascade-0). Adding a per-vertex `bvh_normal` storage buffer for
 * smooth shading normals would push shade to 17 and fail pipeline creation.
 * Textures do NOT count against the storage-buffer limit, so moving the
 * read-only per-triangle `bvh_beer` u32 into a texture frees the slot the
 * normal buffer needs — net storage count is unchanged.
 *
 * Layout: the beer data is one packed RGBA8 `u32` per triangle (alpha = 0).
 * We store it as a 2D `r32uint` texture of fixed width {@link BVH_BEER_TEX_WIDTH}
 * (≤ the WebGPU-guaranteed `maxTextureDimension2D` floor of 8192) and
 * `ceil(triCount / width)` rows. The shade shader maps a triangle index to a
 * texel via `vec2u(tri % W, tri / W)` and `textureLoad(...).r`. The same
 * `BVH_BEER_TEX_WIDTH` constant is injected into the WGSL so the host and
 * shader address identically (single source of truth).
 */

import {
  assertBvhTextureFitsDevice,
  assertBvhTextureRefreshCapacity,
  normalizeBvhTextureTriangleCount,
} from './bvhTextureLimits.js';

/** Fixed texture width for the beer texture. Power-of-two ≤ 8192 (the WebGPU
 *  `maxTextureDimension2D` guaranteed floor). The shader uses the SAME value. */
const BVH_BEER_TEX_WIDTH = 4096;

/** `GPUTextureUsage.TEXTURE_BINDING | COPY_DST` — literals avoid a top-level
 *  `GPUTextureUsage` reference (Node vitest has no WebGPU globals). */
const TEX_BINDING = 0x04;
const COPY_DST = 0x02;

export interface BeerTexture {
  texture: GPUTexture;
  width: number;
  height: number;
}

/** Compute the r32uint beer-texture dimensions for `triCount` triangles. */
function beerTextureSize(triCount: number): { width: number; height: number } {
  const count = normalizeBvhTextureTriangleCount(triCount);
  const width = Math.min(BVH_BEER_TEX_WIDTH, count);
  const height = Math.ceil(count / width);
  return { width, height };
}

/**
 * Create + upload the beer texture from the packed per-triangle u32 data.
 * `beerData` is the `SceneBVHBuffers.bvhBeerColors.cpuData` ArrayBuffer
 * (one u32 per triangle). `triCount` is the number of valid triangles; the
 * texture's last row is zero-padded to `width`.
 */
export function uploadBeerTexture(
  device: GPUDevice,
  beerData: ArrayBuffer,
  triCount: number,
): BeerTexture {
  const { width, height } = beerTextureSize(triCount);
  assertBvhTextureFitsDevice('bvhBeer', device, width, height, triCount);
  const texture = device.createTexture({
    label: 'vitrum.bvhBeer.r32uint',
    size: { width, height, depthOrArrayLayers: 1 },
    format: 'r32uint',
    usage: TEX_BINDING | COPY_DST,
  });
  writeBeerTexture(device, texture, beerData, triCount, width, height);
  return { texture, width, height };
}

/**
 * Re-upload the FULL beer texture from packed u32 data. Used by the material-
 * slice fast path: a contiguous triangle range maps to a rectangular texture
 * region only when it spans full rows, so re-uploading the whole (small) beer
 * texture is the simplest correct choice. At 4 bytes/triangle this is a few KB
 * to a few MB even for dense scenes.
 */
export function refreshBeerTexture(
  device: GPUDevice,
  tex: BeerTexture,
  beerData: ArrayBuffer,
  triCount: number,
): void {
  assertBvhTextureRefreshCapacity('bvhBeer', tex.width, tex.height, triCount);
  writeBeerTexture(device, tex.texture, beerData, triCount, tex.width, tex.height);
}

function writeBeerTexture(
  device: GPUDevice,
  texture: GPUTexture,
  beerData: ArrayBuffer,
  triCount: number,
  width: number,
  height: number,
): void {
  // Pad to a full width×height u32 grid (the source has exactly triCount u32s).
  const src = new Uint32Array(beerData);
  const padded = new Uint32Array(width * height);
  padded.set(src.subarray(0, Math.min(src.length, triCount)));
  device.queue.writeTexture(
    { texture },
    padded.buffer,
    { bytesPerRow: width * 4, rowsPerImage: height },
    { width, height, depthOrArrayLayers: 1 },
  );
}
