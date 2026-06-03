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
 * Format: `rgba32float` (rgb = Le, a = 0). Emissive is HDR (emissiveIntensity
 * may exceed 1), so unlike the LDR `bvh_beer` u32 texture this needs float
 * storage. `rgba32float` is chosen over `rgba16float` so `writeTexture` can take
 * the packed `Float32Array` directly with no host-side f16 encode. The texture
 * uses `unfilterable-float` sampleType (`tex` in the descriptor table) and is
 * read via `textureLoad` (no sampler).
 *
 * Layout mirrors `bvhBeerTexture`: fixed width {@link BVH_EMISSIVE_TEX_WIDTH},
 * `ceil(triCount / width)` rows; triangle index → texel `vec2u(tri % W, tri / W)`.
 */

/** Fixed texture width (power-of-two ≤ 8192 WebGPU floor). Shader uses the SAME. */
const BVH_EMISSIVE_TEX_WIDTH = 4096;

/** `GPUTextureUsage.TEXTURE_BINDING | COPY_DST` literals (Node vitest lacks WebGPU globals). */
const TEX_BINDING = 0x04;
const COPY_DST = 0x02;

export interface EmissiveTexture {
  texture: GPUTexture;
  width: number;
  height: number;
}

/** Compute the rgba32float emissive-texture dimensions for `triCount` triangles. */
function emissiveTextureSize(triCount: number): { width: number; height: number } {
  const count = Math.max(1, triCount | 0);
  const width = Math.min(BVH_EMISSIVE_TEX_WIDTH, count);
  const height = Math.ceil(count / width);
  return { width, height };
}

/**
 * Create + upload the emissive texture from the packed per-triangle Le data
 * (`packBVHEmissiveLe` output: 4 floats per triangle, rgb + 0 pad). `triCount`
 * is the number of valid triangles; the last texture row is zero-padded.
 */
export function uploadEmissiveTexture(
  device: GPUDevice,
  emissiveData: Float32Array,
  triCount: number,
): EmissiveTexture {
  const { width, height } = emissiveTextureSize(triCount);
  const texture = device.createTexture({
    label: 'vitrum.bvhEmissive.rgba32float',
    size: { width, height, depthOrArrayLayers: 1 },
    format: 'rgba32float',
    usage: TEX_BINDING | COPY_DST,
  });
  writeEmissiveTexture(device, texture, emissiveData, triCount, width, height);
  return { texture, width, height };
}

/** Re-upload the full emissive texture (material-slice fast path; see bvhBeerTexture). */
export function refreshEmissiveTexture(
  device: GPUDevice,
  tex: EmissiveTexture,
  emissiveData: Float32Array,
  triCount: number,
): void {
  writeEmissiveTexture(device, tex.texture, emissiveData, triCount, tex.width, tex.height);
}

function writeEmissiveTexture(
  device: GPUDevice,
  texture: GPUTexture,
  emissiveData: Float32Array,
  triCount: number,
  width: number,
  height: number,
): void {
  // Pad to a full width×height rgba32float grid (source has 4 floats/triangle).
  const padded = new Float32Array(width * height * 4);
  padded.set(emissiveData.subarray(0, Math.min(emissiveData.length, triCount * 4)));
  device.queue.writeTexture(
    { texture },
    padded.buffer,
    { bytesPerRow: width * 4 * 4, rowsPerImage: height },
    { width, height, depthOrArrayLayers: 1 },
  );
}
