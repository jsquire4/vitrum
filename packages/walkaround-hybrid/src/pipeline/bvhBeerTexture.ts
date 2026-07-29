/**
 * WS1 (feature-completeness wave, 2026-05-29) — per-triangle Beer-Lambert
 * visible-color storage as an `r32uint` texture instead of a storage buffer.
 *
 * Motivation: the shade pass sits at the WebGPU-guaranteed
 * `maxStorageBuffersPerShaderStage = 8` floor (4 frame + 3 versioned scene
 * arenas + 1 RC cascade-0). Textures do not count against that limit, so
 * keeping the read-only per-triangle `bvh_beer` lane in a texture avoids a
 * ninth storage binding.
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
  uploadElementTexture,
  writeElementTexture,
} from './bvhTextureLimits.js';

/** Fixed texture width for the beer texture. Power-of-two ≤ 8192 (the WebGPU
 *  `maxTextureDimension2D` guaranteed floor). The shader uses the SAME value. */
const BVH_BEER_TEX_WIDTH = 4096;

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
  const texture = uploadElementTexture(device, {
    label: 'vitrum.bvhBeer.r32uint',
    format: 'r32uint',
    width,
    height,
    bytesPerTexel: 4,
    elementsPerTexel: 1,
    makePadded: (n) => new Uint32Array(n),
    fill: (padded) => fillBeer(padded, beerData, triCount),
  });
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
  writeElementTexture(device, tex.texture, {
    width: tex.width,
    height: tex.height,
    bytesPerTexel: 4,
    elementsPerTexel: 1,
    makePadded: (n) => new Uint32Array(n),
    fill: (padded) => fillBeer(padded, beerData, triCount),
  });
}

/** Populate the padded r32uint grid from the packed per-triangle u32 data
 *  (the source has exactly triCount u32s). */
function fillBeer(padded: Uint32Array, beerData: ArrayBuffer, triCount: number): void {
  const src = new Uint32Array(beerData);
  padded.set(src.subarray(0, Math.min(src.length, triCount)));
}
