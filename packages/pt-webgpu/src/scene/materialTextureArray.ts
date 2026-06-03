// materialTextureArray.ts — P2 GPU upload of collected material textures into a
// single `texture_2d_array` consumed by the full-tier path-trace kernel.
//
// `collectMaterialTextures` (materialTextures.ts) dedups the host texture
// handles into an upload-ordered `sources` list; this module turns that list
// into one sampled `rgba8unorm-srgb` 2D-array (one source per array layer) plus
// a filtering sampler. The WGSL sampler indexes a layer by the per-material
// `baseColorIdx` descriptor and samples with the interpolated hit UV.
//
// THREE-free by duck-typing: a source is either a THREE.Texture-like object
// carrying `.image` (ImageBitmap / HTMLCanvasElement / HTMLImageElement, or a
// DataTexture's `{ data, width, height }`), or one of those payloads directly.
// No `import 'three'` — pt-webgpu stays host-agnostic.
//
// v1 scope: baseColor only; all layers share the array's dimensions (the max
// across sources). Same-size texture sets (the common case — one texture, or a
// uniform set) are exact; genuinely heterogeneous sizes are copied at native
// size into the max-sized layer and a warning is logged (a per-layer UV-fit /
// true atlas is the documented follow-on). A textureless scene gets a 1×1 white
// dummy layer so the binding is always satisfied (the descriptors are all -1, so
// the kernel never samples it and the render stays byte-identical to pre-P2).

export interface MaterialTextureArray {
  readonly texture: GPUTexture;
  readonly view: GPUTextureView;
  readonly sampler: GPUSampler;
  /** Array layer count (≥ 1; 1 = the white dummy when there are no sources). */
  readonly layerCount: number;
  readonly warnings: readonly string[];
}

interface ImagePayload {
  readonly width: number;
  readonly height: number;
  /** Present for raw-data (DataTexture-style) sources → writeTexture path. */
  readonly data?: ArrayBufferView;
  /** Present for GPU-copyable external images → copyExternalImageToTexture path. */
  readonly external?: GPUCopyExternalImageSource;
}

/** Duck-type a host texture handle into an upload payload, or null if unusable. */
function payloadOf(source: unknown): ImagePayload | null {
  if (source == null || typeof source !== 'object') return null;
  // THREE.Texture-like: unwrap `.image`; otherwise treat the source as the image.
  const img = ('image' in source && (source as { image?: unknown }).image != null
    ? (source as { image: unknown }).image
    : source) as Record<string, unknown>;
  if (img == null || typeof img !== 'object') return null;
  const width = typeof img.width === 'number' ? img.width : 0;
  const height = typeof img.height === 'number' ? img.height : 0;
  if (width <= 0 || height <= 0) return null;
  // DataTexture-style { data, width, height } → writeTexture.
  if (ArrayBuffer.isView(img.data as ArrayBufferView)) {
    return { width, height, data: img.data as ArrayBufferView };
  }
  // ImageBitmap / HTMLCanvasElement / HTMLImageElement / OffscreenCanvas /
  // VideoFrame — all valid copyExternalImageToTexture sources. We can't
  // `instanceof`-check headlessly, so accept any object with positive
  // dimensions that isn't raw data and let the device validate it.
  return { width, height, external: img as unknown as GPUCopyExternalImageSource };
}

const DUMMY_LABEL = 'vitrum.pt-webgpu.scene.materialTextures.dummy';
const ARRAY_LABEL = 'vitrum.pt-webgpu.scene.materialTextures';

function makeSampler(device: GPUDevice): GPUSampler {
  return device.createSampler({
    label: 'vitrum.pt-webgpu.scene.materialTextures.sampler',
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
  });
}

/** 1×1 white single-layer array — the always-bound placeholder for scenes with
 *  no sampled textures (kernel never reads it; descriptors are all -1). */
function createDummyArray(device: GPUDevice): MaterialTextureArray {
  const texture = device.createTexture({
    label: DUMMY_LABEL,
    size: { width: 1, height: 1, depthOrArrayLayers: 1 },
    format: 'rgba8unorm-srgb',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture, origin: { x: 0, y: 0, z: 0 } },
    new Uint8Array([255, 255, 255, 255]),
    { bytesPerRow: 4, rowsPerImage: 1 },
    { width: 1, height: 1, depthOrArrayLayers: 1 },
  );
  return {
    texture,
    view: texture.createView({ dimension: '2d-array' }),
    sampler: makeSampler(device),
    layerCount: 1,
    warnings: [],
  };
}

/**
 * Build the material texture 2D-array. `sources` is the dedup'd, upload-ordered
 * handle list from {@link collectMaterialTextures}; layer `i` holds `sources[i]`,
 * matching the `baseColorIdx` the descriptor buffer stores.
 */
export function createMaterialTextureArray(
  device: GPUDevice,
  sources: ReadonlyArray<unknown>,
): MaterialTextureArray {
  if (sources.length === 0) return createDummyArray(device);

  const warnings: string[] = [];
  const payloads = sources.map(payloadOf);
  const maxDim = device.limits.maxTextureDimension2D;
  let width = 1;
  let height = 1;
  for (const p of payloads) {
    if (p == null) continue;
    width = Math.max(width, Math.min(p.width, maxDim));
    height = Math.max(height, Math.min(p.height, maxDim));
  }

  const texture = device.createTexture({
    label: ARRAY_LABEL,
    size: { width, height, depthOrArrayLayers: sources.length },
    format: 'rgba8unorm-srgb',
    // RENDER_ATTACHMENT is required by copyExternalImageToTexture.
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.RENDER_ATTACHMENT,
  });

  for (let layer = 0; layer < payloads.length; layer += 1) {
    const p = payloads[layer];
    if (p == null) {
      warnings.push(`[materialTextureArray] source ${layer} has no usable image; layer left black.`);
      continue;
    }
    if (p.width !== width || p.height !== height) {
      warnings.push(
        `[materialTextureArray] source ${layer} is ${p.width}×${p.height} but the array is ` +
          `${width}×${height}; copied at native size (UVs for this layer may be off until ` +
          `per-layer UV-fit lands). Use same-size baseColor textures for exact v1 results.`,
      );
    }
    const copyW = Math.min(p.width, width);
    const copyH = Math.min(p.height, height);
    if (p.external != null) {
      device.queue.copyExternalImageToTexture(
        { source: p.external, flipY: false },
        { texture, origin: { x: 0, y: 0, z: layer } },
        { width: copyW, height: copyH },
      );
    } else if (p.data != null) {
      device.queue.writeTexture(
        { texture, origin: { x: 0, y: 0, z: layer } },
        p.data as GPUAllowSharedBufferSource,
        { bytesPerRow: p.width * 4, rowsPerImage: p.height },
        { width: copyW, height: copyH },
      );
    }
  }

  return {
    texture,
    view: texture.createView({ dimension: '2d-array' }),
    sampler: makeSampler(device),
    layerCount: sources.length,
    warnings,
  };
}
