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
// Layers share the array's dimensions (the max across sources), but each layer
// also exposes a UV-fit scale so the descriptor packer can remap samples into the
// copied source rectangle. A textureless scene gets a 1×1 white dummy layer so
// the binding is always satisfied (the descriptors are all -1, so the kernel
// never samples it and the render stays byte-identical to pre-P2).

import type { MaterialTextureLayerInfo, MaterialTextureLayerUse } from './materialTextures.js';

export type MaterialTextureLayerUvScale = readonly [number, number];

export type MaterialTextureArrayWarningCode =
  | 'texture-unreadable'
  | 'texture-size-mismatch'
  | 'texture-unsupported-layout'
  | 'texture-sampler-policy-approximation';

export interface MaterialTextureArrayWarning {
  readonly code: MaterialTextureArrayWarningCode;
  readonly warning: string;
  readonly layer: number;
  readonly uses: readonly MaterialTextureLayerUse[];
  readonly fallback?: string;
  readonly width?: number;
  readonly height?: number;
  readonly arrayWidth?: number;
  readonly arrayHeight?: number;
  readonly byteLength?: number;
  readonly requestedSamplerPolicies?: readonly MaterialTextureSamplerPolicyUse[];
}

export interface MaterialTextureSamplerPolicyUse {
  readonly materialIndex: number;
  readonly field: string;
  readonly magFilter?: string;
  readonly minFilter?: string;
  readonly mipFilter?: string;
}

export interface MaterialTextureArray {
  readonly texture: GPUTexture;
  readonly view: GPUTextureView;
  readonly sampler: GPUSampler;
  /** Array layer count (≥ 1; 1 = the white dummy when there are no sources). */
  readonly layerCount: number;
  /** Number of mip levels allocated/generated for every array layer. */
  readonly mipLevelCount: number;
  /** Per-layer source-rect UV scale: [copyWidth / arrayWidth, copyHeight / arrayHeight]. */
  readonly layerUvScales: readonly MaterialTextureLayerUvScale[];
  readonly warnings: readonly string[];
  readonly structuredWarnings: readonly MaterialTextureArrayWarning[];
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
    ? (source).image
    : source) as Record<string, unknown>;
  if (img == null || typeof img !== 'object') return null;
  const width = typeof img.width === 'number' ? img.width : 0;
  const height = typeof img.height === 'number' ? img.height : 0;
  if (width <= 0 || height <= 0) return null;
  // DataTexture-style { data, width, height } → writeTexture.
  if (ArrayBuffer.isView(img.data)) {
    return { width, height, data: img.data };
  }
  // ImageBitmap / HTMLCanvasElement / HTMLImageElement / OffscreenCanvas /
  // VideoFrame — all valid copyExternalImageToTexture sources. We can't
  // `instanceof`-check headlessly, so accept any object with positive
  // dimensions that isn't raw data and let the device validate it.
  return { width, height, external: img as unknown as GPUCopyExternalImageSource };
}

interface NormalizedRgba8Upload {
  readonly data: Uint8Array<ArrayBuffer>;
  readonly bytesPerRow: number;
  readonly rowsPerImage: number;
}

function byteViewOf(data: ArrayBufferView): Uint8Array<ArrayBuffer> | null {
  const bytesPerElement = (data as { readonly BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT;
  if (!(data instanceof DataView) && bytesPerElement !== 1) return null;
  return new Uint8Array(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
}

function numericChannelMax(data: ArrayBufferView): number | null {
  if (data instanceof Float32Array || data instanceof Float64Array) return 1;
  if (data instanceof Uint16Array) return 65535;
  if (data instanceof Uint32Array) return 4294967295;
  if (data instanceof Int16Array) return 32767;
  if (data instanceof Int32Array) return 2147483647;
  return null;
}

function normalizedNumberToByte(value: number, maxValue: number): number {
  if (!Number.isFinite(value)) return 0;
  const unit = maxValue === 1 ? value : value / maxValue;
  return Math.round(Math.min(1, Math.max(0, unit)) * 255);
}

function numericChannelAt(data: ArrayBufferView, index: number): number {
  const view = data as unknown as ArrayLike<number>;
  return Number(view[index] ?? 0);
}

function normalizeRawRgba8(
  data: ArrayBufferView,
  width: number,
  height: number,
): NormalizedRgba8Upload | null {
  const bytes = byteViewOf(data);
  if (bytes == null) return null;
  const pixelCount = width * height;
  if (pixelCount <= 0) return null;
  const channels = bytes.byteLength / pixelCount;
  if (![1, 2, 3, 4].includes(channels) || !Number.isInteger(channels)) return null;
  if (channels === 4) {
    return { data: bytes, bytesPerRow: width * 4, rowsPerImage: height };
  }
  const rgba = new Uint8Array(pixelCount * 4);
  for (let i = 0; i < pixelCount; i += 1) {
    const src = i * channels;
    const dst = i * 4;
    const r = bytes[src] ?? 0;
    const g = channels >= 2 ? bytes[src + 1] ?? 0 : r;
    const b = channels >= 3 ? bytes[src + 2] ?? 0 : r;
    const a = channels >= 4 ? bytes[src + 3] ?? 255 : 255;
    rgba[dst] = r;
    rgba[dst + 1] = g;
    rgba[dst + 2] = b;
    rgba[dst + 3] = a;
  }
  return { data: rgba, bytesPerRow: width * 4, rowsPerImage: height };
}

function normalizeRawNumericRgba8(
  data: ArrayBufferView,
  width: number,
  height: number,
): NormalizedRgba8Upload | null {
  const maxValue = numericChannelMax(data);
  if (maxValue == null) return null;
  const length = (data as unknown as ArrayLike<number>).length;
  const pixelCount = width * height;
  if (pixelCount <= 0 || length == null) return null;
  const channels = length / pixelCount;
  if (![1, 2, 3, 4].includes(channels) || !Number.isInteger(channels)) return null;
  const rgba = new Uint8Array(pixelCount * 4);
  for (let i = 0; i < pixelCount; i += 1) {
    const src = i * channels;
    const dst = i * 4;
    const r = numericChannelAt(data, src);
    const g = channels >= 2 ? numericChannelAt(data, src + 1) : r;
    const b = channels >= 3 ? numericChannelAt(data, src + 2) : r;
    const a = channels >= 4 ? numericChannelAt(data, src + 3) : maxValue;
    rgba[dst] = normalizedNumberToByte(r, maxValue);
    rgba[dst + 1] = normalizedNumberToByte(g, maxValue);
    rgba[dst + 2] = normalizedNumberToByte(b, maxValue);
    rgba[dst + 3] = normalizedNumberToByte(a, maxValue);
  }
  return { data: rgba, bytesPerRow: width * 4, rowsPerImage: height };
}

function normalizeRawTextureUpload(
  data: ArrayBufferView,
  width: number,
  height: number,
): NormalizedRgba8Upload | null {
  return normalizeRawRgba8(data, width, height) ?? normalizeRawNumericRgba8(data, width, height);
}

const DUMMY_LABEL = 'vitrum.pt-webgpu.scene.materialTextures.dummy';
const ARRAY_LABEL = 'vitrum.pt-webgpu.scene.materialTextures';

export function materialTextureMipLevelCount(width: number, height: number): number {
  const maxDim = Math.max(1, Math.floor(Math.max(width, height)));
  return Math.floor(Math.log2(maxDim)) + 1;
}

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

const MATERIAL_TEXTURE_MIPMAP_WGSL = /* wgsl */ `
struct VsOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vsMain(@builtin(vertex_index) vertexIndex: u32) -> VsOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
  );
  let p = positions[vertexIndex];
  var out: VsOut;
  out.position = vec4f(p, 0.0, 1.0);
  out.uv = p * 0.5 + vec2f(0.5);
  return out;
}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;

@fragment
fn fsMain(in: VsOut) -> @location(0) vec4f {
  return textureSample(srcTex, srcSampler, in.uv);
}
`;

function generateTextureArrayMips(
  device: GPUDevice,
  texture: GPUTexture,
  layerCount: number,
  format: GPUTextureFormat,
  mipLevelCount: number,
): void {
  if (mipLevelCount <= 1) return;

  const module = device.createShaderModule({
    label: 'vitrum.pt-webgpu.scene.materialTextures.mipmap.module',
    code: MATERIAL_TEXTURE_MIPMAP_WGSL,
  });
  const sampler = device.createSampler({
    label: 'vitrum.pt-webgpu.scene.materialTextures.mipmap.sampler',
    magFilter: 'linear',
    minFilter: 'linear',
  });
  const pipeline = device.createRenderPipeline({
    label: 'vitrum.pt-webgpu.scene.materialTextures.mipmap.pipeline',
    layout: 'auto',
    vertex: { module, entryPoint: 'vsMain' },
    fragment: { module, entryPoint: 'fsMain', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  const encoder = device.createCommandEncoder({
    label: 'vitrum.pt-webgpu.scene.materialTextures.mipmap.encoder',
  });

  for (let layer = 0; layer < layerCount; layer += 1) {
    for (let mip = 1; mip < mipLevelCount; mip += 1) {
      const sourceView = texture.createView({
        dimension: '2d',
        baseMipLevel: mip - 1,
        mipLevelCount: 1,
        baseArrayLayer: layer,
        arrayLayerCount: 1,
      });
      const targetView = texture.createView({
        dimension: '2d',
        baseMipLevel: mip,
        mipLevelCount: 1,
        baseArrayLayer: layer,
        arrayLayerCount: 1,
      });
      const bindGroup = device.createBindGroup({
        label: 'vitrum.pt-webgpu.scene.materialTextures.mipmap.bindGroup',
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: sourceView },
          { binding: 1, resource: sampler },
        ],
      });
      const pass = encoder.beginRenderPass({
        label: 'vitrum.pt-webgpu.scene.materialTextures.mipmap.pass',
        colorAttachments: [{
          view: targetView,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
      pass.end();
    }
  }

  device.queue.submit([encoder.finish()]);
}

/** 1×1 white single-layer array — the always-bound placeholder for scenes with
 *  no sampled textures (kernel never reads it; descriptors are all -1). */
function createDummyArray(device: GPUDevice, format: GPUTextureFormat): MaterialTextureArray {
  const texture = device.createTexture({
    label: DUMMY_LABEL,
    size: { width: 1, height: 1, depthOrArrayLayers: 1 },
    format,
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
    mipLevelCount: 1,
    layerUvScales: [[1, 1]],
    warnings: [],
    structuredWarnings: [],
  };
}

function layerUses(
  layerInfos: ReadonlyArray<MaterialTextureLayerInfo> | undefined,
  layer: number,
): readonly MaterialTextureLayerUse[] {
  return layerInfos?.[layer]?.uses ?? [];
}

function samplerPolicyIsNativeForPtWebgpu(use: MaterialTextureLayerUse): boolean {
  if (use.magFilter != null && use.magFilter !== 'linear') return false;
  if (use.minFilter != null && use.minFilter !== 'linear') return false;
  if (use.mipFilter != null && use.mipFilter !== 'linear') return false;
  return true;
}

function requestedSamplerPolicy(use: MaterialTextureLayerUse): MaterialTextureSamplerPolicyUse {
  return {
    materialIndex: use.materialIndex,
    field: use.field,
    ...(use.magFilter != null ? { magFilter: use.magFilter } : {}),
    ...(use.minFilter != null ? { minFilter: use.minFilter } : {}),
    ...(use.mipFilter != null ? { mipFilter: use.mipFilter } : {}),
  };
}

function appendSamplerPolicyWarnings(
  warnings: string[],
  structuredWarnings: MaterialTextureArrayWarning[],
  layerInfos: ReadonlyArray<MaterialTextureLayerInfo> | undefined,
): void {
  if (layerInfos == null) return;
  for (const info of layerInfos) {
    const nonNative = info.uses.filter((use) => !samplerPolicyIsNativeForPtWebgpu(use));
    if (nonNative.length === 0) continue;
    const fields = Array.from(new Set(nonNative.map((use) => use.field))).join(', ');
    const warning =
      `[materialTextureArray] source ${info.layer} requests sampler policy outside pt-webgpu's ` +
      `shared linear/linear/mipmap-linear material texture sampler for fields ${fields}; ` +
      'the texture remains uploadable, but filtering/mipmap behavior is approximate.';
    warnings.push(warning);
    structuredWarnings.push({
      code: 'texture-sampler-policy-approximation',
      warning,
      layer: info.layer,
      uses: nonNative,
      fallback: 'shared-linear-mipmapped-sampler',
      requestedSamplerPolicies: nonNative.map(requestedSamplerPolicy),
    });
  }
}

/**
 * Build the material texture 2D-array. `sources` is the dedup'd, upload-ordered
 * handle list from {@link collectMaterialTextures}; layer `i` holds `sources[i]`,
 * matching the `baseColorIdx` the descriptor buffer stores.
 */
export function createMaterialTextureArray(
  device: GPUDevice,
  sources: ReadonlyArray<unknown>,
  format: GPUTextureFormat = 'rgba8unorm-srgb',
  layerInfos?: ReadonlyArray<MaterialTextureLayerInfo>,
): MaterialTextureArray {
  if (sources.length === 0) return createDummyArray(device, format);

  const warnings: string[] = [];
  const structuredWarnings: MaterialTextureArrayWarning[] = [];
  appendSamplerPolicyWarnings(warnings, structuredWarnings, layerInfos);
  const payloads = sources.map(payloadOf);
  const maxDim = device.limits.maxTextureDimension2D;
  let width = 1;
  let height = 1;
  for (const p of payloads) {
    if (p == null) continue;
    width = Math.max(width, Math.min(p.width, maxDim));
    height = Math.max(height, Math.min(p.height, maxDim));
  }
  const mipLevelCount = materialTextureMipLevelCount(width, height);

  const texture = device.createTexture({
    label: ARRAY_LABEL,
    size: { width, height, depthOrArrayLayers: sources.length },
    mipLevelCount,
    format,
    // RENDER_ATTACHMENT lets us downsample base uploads into a real mip chain.
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.RENDER_ATTACHMENT,
  });

  for (let layer = 0; layer < payloads.length; layer += 1) {
    const p = payloads[layer];
    if (p == null) {
      const warning = `[materialTextureArray] source ${layer} has no usable image; layer left black.`;
      warnings.push(warning);
      structuredWarnings.push({
        code: 'texture-unreadable',
        warning,
        layer,
        uses: layerUses(layerInfos, layer),
        fallback: 'black-layer',
      });
      continue;
    }
    if (p.width !== width || p.height !== height) {
      const warning =
        `[materialTextureArray] source ${layer} is ${p.width}×${p.height} but the array is ` +
          `${width}×${height}; copied at native size and sampled through a per-layer UV-fit scale. ` +
          `Use same-size textures when exact mip/border filtering parity is required.`;
      warnings.push(warning);
      structuredWarnings.push({
        code: 'texture-size-mismatch',
        warning,
        layer,
        uses: layerUses(layerInfos, layer),
        width: p.width,
        height: p.height,
        arrayWidth: width,
        arrayHeight: height,
      });
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
      const upload = normalizeRawTextureUpload(p.data, p.width, p.height);
      if (upload == null) {
        const warning =
          `[materialTextureArray] source ${layer} has raw data with unsupported byte layout ` +
            `(${p.data.byteLength} bytes for ${p.width}×${p.height}); expected 1, 2, 3, or 4 ` +
            `8-bit or normalized numeric channel(s) per pixel. Layer left black.`;
        warnings.push(warning);
        structuredWarnings.push({
          code: 'texture-unsupported-layout',
          warning,
          layer,
          uses: layerUses(layerInfos, layer),
          width: p.width,
          height: p.height,
          byteLength: p.data.byteLength,
          fallback: 'black-layer',
        });
        continue;
      }
      device.queue.writeTexture(
        { texture, origin: { x: 0, y: 0, z: layer } },
        upload.data,
        { bytesPerRow: upload.bytesPerRow, rowsPerImage: upload.rowsPerImage },
        { width: copyW, height: copyH },
      );
    }
  }
  generateTextureArrayMips(device, texture, sources.length, format, mipLevelCount);

  return {
    texture,
    view: texture.createView({ dimension: '2d-array' }),
    sampler: makeSampler(device),
    layerCount: sources.length,
    mipLevelCount,
    layerUvScales: payloads.map((p): MaterialTextureLayerUvScale => {
      if (p == null) return [1, 1];
      const copyW = Math.min(p.width, width);
      const copyH = Math.min(p.height, height);
      return [copyW / width, copyH / height];
    }),
    warnings,
    structuredWarnings,
  };
}
