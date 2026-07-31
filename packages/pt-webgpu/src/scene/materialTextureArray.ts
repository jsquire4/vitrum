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
// Layers share the array's dimensions (the max across sources), but every
// source is first uploaded and mipmapped in its own native-size scratch texture.
// Each native mip rectangle is then copied into the corresponding array mip.
// The shader derives the exact integer source extent from the per-layer UV-fit
// scale and performs source-rectangle-aware wrap/filtering with textureLoad.
// Consequently heterogeneous layers never filter against padded texels and
// retain independent mip chains. A textureless scene gets a 1×1 white dummy
// layer so the binding is always satisfied (the descriptors are all -1).

import type { MaterialTextureLayerInfo, MaterialTextureLayerUse } from './materialTextures.js';
import {
  halfToFloat,
  packRadianceRgbProductF32,
  type RadianceRgb,
} from '@vitrum/shared-bvh';
import {
  isPtWebgpuTextureSource,
  type PtWebgpuTextureSource,
} from '../materialTextureSource.js';

export type MaterialTextureLayerUvScale = readonly [number, number];

export type MaterialTextureArrayWarningCode =
  | 'texture-unreadable'
  | 'texture-unsupported-layout';

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
}

export interface MaterialTextureArray {
  readonly texture: GPUTexture;
  readonly view: GPUTextureView;
  readonly sampler: GPUSampler;
  /** Array layer count (≥ 1; 1 = the white dummy when there are no sources). */
  readonly layerCount: number;
  /** Maximum allocated mip count; each layer owns its exact native-size prefix. */
  readonly mipLevelCount: number;
  /** Per-layer source-rect UV scale: [copyWidth / arrayWidth, copyHeight / arrayHeight]. */
  readonly layerUvScales: readonly MaterialTextureLayerUvScale[];
  readonly warnings: readonly string[];
  readonly structuredWarnings: readonly MaterialTextureArrayWarning[];
}

export type MaterialTextureRadianceFactorsByMaterial = ReadonlyArray<
  readonly RadianceRgb[] | undefined
>;

/**
 * Exact shader operands for material maps that publish outgoing radiance.
 *
 * A material can have more than one emissive factor: camera-hit shading reads
 * the packed material record, while an explicit mesh-area emitter can publish a
 * distinct base Le for NEE. Every factor that can multiply a layer is therefore
 * retained and checked independently against every exact CPU-readable texel.
 */
export interface MaterialTextureRadianceEnvelope {
  readonly emissiveMap?: MaterialTextureRadianceFactorsByMaterial;
  readonly lightMap?: MaterialTextureRadianceFactorsByMaterial;
}

interface ImagePayload {
  readonly width: number;
  readonly height: number;
  /** Present for raw-data (DataTexture-style) sources → writeTexture path. */
  readonly data?: ArrayBufferView;
  /** Present for GPU-copyable external images → copyExternalImageToTexture path. */
  readonly external?: GPUCopyExternalImageSource;
  /** Explicit same-device GPU source → shader conversion path, no readback. */
  readonly gpuSource?: PtWebgpuTextureSource;
}

/** Conservative per-array peak for destination mips + retained/decoded inputs. */
export const MATERIAL_TEXTURE_ARRAY_PEAK_BUDGET_BYTES = 512 * 1024 * 1024;

function positiveSafeDimension(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 1) {
    throw new RangeError(
      `[materialTextureArray] ${label} must be a positive safe integer; received ${String(value)}.`,
    );
  }
  return value as number;
}

function looksLikeRawGpuTexture(value: unknown): value is GPUTexture {
  if (value == null || typeof value !== 'object') return false;
  const candidate = value as Partial<GPUTexture>;
  return (
    typeof candidate.createView === 'function' &&
    typeof candidate.width === 'number' &&
    typeof candidate.height === 'number' &&
    typeof candidate.format === 'string'
  );
}

/** Duck-type a host texture handle into an upload payload, or null if unusable. */
function payloadOf(source: unknown): ImagePayload | null {
  if (isPtWebgpuTextureSource(source)) {
    return {
      width: positiveSafeDimension(source.width, 'GPU source width'),
      height: positiveSafeDimension(source.height, 'GPU source height'),
      gpuSource: source,
    };
  }
  if (looksLikeRawGpuTexture(source)) {
    throw new TypeError(
      '[materialTextureArray] raw GPUTexture handles are ambiguous. Wrap the texture ' +
      'with createPtWebgpuTextureSource so device, format, color space, and selected ' +
      'subresource are explicit.',
    );
  }
  if (source == null || typeof source !== 'object') return null;
  // THREE.Texture-like: unwrap `.image`; otherwise treat the source as the image.
  const img = ('image' in source && (source as { image?: unknown }).image != null
    ? (source).image
    : source) as Record<string, unknown>;
  if (img == null || typeof img !== 'object') return null;
  const width = img.width;
  const height = img.height;
  if (
    !Number.isSafeInteger(width) ||
    !Number.isSafeInteger(height) ||
    (width as number) < 1 ||
    (height as number) < 1
  ) {
    return null;
  }
  // DataTexture-style { data, width, height } → writeTexture.
  if (ArrayBuffer.isView(img.data)) {
    return { width: width as number, height: height as number, data: img.data };
  }
  // ImageBitmap / HTMLCanvasElement / HTMLImageElement / OffscreenCanvas /
  // VideoFrame — all valid copyExternalImageToTexture sources. We can't
  // `instanceof`-check headlessly, so accept any object with positive
  // dimensions that isn't raw data and let the device validate it.
  return {
    width: width as number,
    height: height as number,
    external: img as unknown as GPUCopyExternalImageSource,
  };
}

interface NormalizedRgba8Upload {
  readonly data: Uint8Array<ArrayBuffer>;
  readonly bytesPerRow: number;
  readonly rowsPerImage: number;
}

/** rgba16float upload payload — 4 half-floats (Uint16 bit patterns) per texel. */
interface NormalizedFloatUpload {
  readonly data: Uint16Array<ArrayBuffer>;
  readonly bytesPerRow: number;
  readonly rowsPerImage: number;
}

/**
 * sRGB electro-optical transfer function (sRGB-encoded value → linear). Matches
 * the hardware decode a `rgba8unorm-srgb`-sampled texture applies, so an LDR
 * emissive texture that previously rode the sRGB array produces the IDENTICAL
 * linear value when decoded here and stored in the linear `rgba16float` array.
 * Alpha is NOT transfer-encoded in sRGB, so callers must pass alpha straight
 * through (this fn is applied per RGB channel only).
 */
function srgbToLinear(c: number): number {
  if (!Number.isFinite(c)) return 0;
  const u = Math.min(1, Math.max(0, c));
  return u <= 0.04045 ? u / 12.92 : ((u + 0.055) / 1.055) ** 2.4;
}

/**
 * IEEE-754 float32 → float16 (half) bit pattern, returned as a u16. Round-to-
 * nearest-even is not required for texture data; round-toward-zero on the
 * mantissa (truncation) is used, matching common GPU upload encoders. Handles
 * sub-normals, overflow→±inf, and NaN. rgba16float half-floats carry values
 * well beyond 1.0 (max finite ≈ 65504), so HDR emissive > 1.0 survives.
 */
function float32ToFloat16(value: number): number {
  if (Number.isNaN(value)) return 0x7e00; // canonical NaN
  const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0;
  const a = Math.abs(value);
  if (a === 0) return sign;
  if (!Number.isFinite(a) || a >= 65520) return sign | 0x7c00; // ±inf / overflow
  if (a < 6.103515625e-5) {
    // Sub-normal half: scale the mantissa into the 10-bit field.
    const mant = Math.round(a / 5.960464477539063e-8);
    return sign | (mant & 0x03ff);
  }
  let e = Math.floor(Math.log2(a));
  if (a < 2 ** e) e -= 1;
  const exp = e + 15;
  if (exp <= 0) return sign;
  if (exp >= 0x1f) return sign | 0x7c00;
  const mant = Math.round((a / 2 ** e - 1) * 1024);
  // Rounding can carry the mantissa to 1024 → bump the exponent.
  if (mant >= 1024) {
    if (exp + 1 >= 0x1f) return sign | 0x7c00;
    return sign | ((exp + 1) << 10);
  }
  return sign | (exp << 10) | (mant & 0x03ff);
}

function radianceFactorsForUse(
  envelope: MaterialTextureRadianceEnvelope,
  use: MaterialTextureLayerUse,
): readonly RadianceRgb[] {
  if (use.field === 'emissiveMap') {
    return envelope.emissiveMap?.[use.materialIndex] ?? [];
  }
  if (use.field === 'lightMap') {
    return envelope.lightMap?.[use.materialIndex] ?? [];
  }
  return [];
}

function exactNormalizedUploadRgb(
  upload: NormalizedRgba8Upload | NormalizedFloatUpload,
  format: GPUTextureFormat,
  texel: number,
): [number, number, number] {
  const base = texel * 4;
  if (format === 'rgba16float') {
    const data = upload.data as Uint16Array;
    return [
      halfToFloat(data[base] ?? 0),
      halfToFloat(data[base + 1] ?? 0),
      halfToFloat(data[base + 2] ?? 0),
    ];
  }
  const data = upload.data as Uint8Array;
  return [
    (data[base] ?? 0) / 255,
    (data[base + 1] ?? 0) / 255,
    (data[base + 2] ?? 0) / 255,
  ];
}

function decodedMirrorChannel(
  source: PtWebgpuTextureSource,
  channel: number,
  texel: number,
): number {
  const mirror = source.cpuMirror!;
  if (channel === 3 && mirror.channels < 4) return 1;
  if (channel === 2 && mirror.channels === 2) return 0;
  const sourceChannel = channel < mirror.channels
    ? channel
    : channel === 1
      ? 0
      : Math.min(channel, mirror.channels - 1);
  const raw = Number(mirror.data[texel * mirror.channels + sourceChannel] ?? 0);
  let value = mirror.dataType === 'uint8'
    ? raw / 255
    : mirror.dataType === 'uint16'
      ? raw / 65535
      : mirror.dataType === 'float16' || mirror.dataType === 'half-float'
        ? halfToFloat(raw)
        : raw;
  if (channel < 3 && mirror.colorSpace === 'srgb') {
    value = srgbToLinear(value);
  }
  return value;
}

function normalizeGpuMirrorFloatUpload(
  source: PtWebgpuTextureSource,
): NormalizedFloatUpload {
  const mirror = source.cpuMirror!;
  const pixelCount = mirror.width * mirror.height;
  const out = new Uint16Array(pixelCount * 4);
  for (let texel = 0; texel < pixelCount; texel += 1) {
    const base = texel * 4;
    for (let channel = 0; channel < 4; channel += 1) {
      const value = decodedMirrorChannel(source, channel, texel);
      if (!Number.isFinite(value) || Math.abs(value) > 65504) {
        throw new RangeError(
          `[materialTextureArray] GPU source cpuMirror texel ${texel} channel ` +
          `${channel} exceeds finite rgba16float range: ${String(value)}.`,
        );
      }
      out[base + channel] = float32ToFloat16(value);
    }
  }
  return {
    data: out,
    bytesPerRow: mirror.width * 8,
    rowsPerImage: mirror.height,
  };
}

function quantizeRadianceTargetChannel(
  value: number,
  format: GPUTextureFormat,
): number {
  if (format === 'rgba16float') {
    return halfToFloat(float32ToFloat16(value));
  }
  return Math.round(Math.min(1, Math.max(0, value)) * 255) / 255;
}

function exactGpuMirrorRgb(
  source: PtWebgpuTextureSource,
  format: GPUTextureFormat,
  texel: number,
): [number, number, number] {
  return [
    quantizeRadianceTargetChannel(decodedMirrorChannel(source, 0, texel), format),
    quantizeRadianceTargetChannel(decodedMirrorChannel(source, 1, texel), format),
    quantizeRadianceTargetChannel(decodedMirrorChannel(source, 2, texel), format),
  ];
}

function validateMaterialTextureRadianceEnvelope(
  payloads: readonly (ImagePayload | null)[],
  normalizedUploads: readonly (
    NormalizedRgba8Upload | NormalizedFloatUpload | null
  )[],
  format: GPUTextureFormat,
  layerInfos: ReadonlyArray<MaterialTextureLayerInfo> | undefined,
  envelope: MaterialTextureRadianceEnvelope | undefined,
): void {
  if (envelope == null) return;
  for (let layer = 0; layer < payloads.length; layer += 1) {
    const factorUses = layerUses(layerInfos, layer).flatMap((use) =>
      radianceFactorsForUse(envelope, use).map((factor) => ({ use, factor }))
    );
    if (factorUses.length === 0) continue;
    const payload = payloads[layer];
    if (payload == null) {
      // An unreadable layer is deterministically black and cannot overflow.
      continue;
    }
    const normalized = normalizedUploads[layer];
    const mirror = payload.gpuSource?.cpuMirror;
    if (normalized == null && mirror == null) {
      if (factorUses.some(({ use }) => use.field === 'emissiveMap')) {
        throw new TypeError(
          `[materialTextureArray] emissive source ${layer} has no exact CPU-readable ` +
          'texels for radiance-envelope validation.',
        );
      }
      // Opaque light maps remain supported. Their shader multiplication uses
      // the deterministic finite product guard because WebGPU cannot read them
      // synchronously during setScene.
      continue;
    }
    const texelCount = payload.width * payload.height;
    for (let texel = 0; texel < texelCount; texel += 1) {
      const rgb = normalized != null
        ? exactNormalizedUploadRgb(normalized, format, texel)
        : exactGpuMirrorRgb(payload.gpuSource!, format, texel);
      for (const { use, factor } of factorUses) {
        packRadianceRgbProductF32(
          factor,
          rgb,
          `@vitrum/pt-webgpu material ${use.materialIndex} ${use.field} ` +
            `layer ${layer} texel ${texel} radiance`,
        );
      }
    }
  }
}

function byteViewOf(data: ArrayBufferView): Uint8Array<ArrayBuffer> | null {
  if (!(data instanceof Uint8Array) && !(data instanceof Uint8ClampedArray)) {
    return null;
  }
  return new Uint8Array(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
}

function numericChannelMax(data: ArrayBufferView): number | null {
  if (data instanceof Float32Array) return 1;
  if (data instanceof Uint16Array) return 65535;
  return null;
}

function normalizedNumberToByte(value: number, maxValue: number): number {
  const unit = maxValue === 1 ? value : value / maxValue;
  return Math.round(unit * 255);
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
    const b = channels >= 3 ? bytes[src + 2] ?? 0 : channels === 1 ? r : 0;
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
  for (let i = 0; i < length; i += 1) {
    const value = numericChannelAt(data, i);
    if (!Number.isFinite(value)) {
      throw new RangeError(
        `[materialTextureArray] raw sample ${i} must be finite.`,
      );
    }
    if (data instanceof Float32Array && (value < 0 || value > 1)) {
      throw new RangeError(
        `[materialTextureArray] rgba8 float sample ${i} must be in [0, 1]; received ${value}.`,
      );
    }
  }
  const rgba = new Uint8Array(pixelCount * 4);
  for (let i = 0; i < pixelCount; i += 1) {
    const src = i * channels;
    const dst = i * 4;
    const r = numericChannelAt(data, src);
    const g = channels >= 2 ? numericChannelAt(data, src + 1) : r;
    const b = channels >= 3 ? numericChannelAt(data, src + 2) : channels === 1 ? r : 0;
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

/**
 * Raw-data upload path for the `rgba16float` emissive array. Two colour-space
 * conventions, keyed by the SOURCE typed-array element type so the LDR path
 * round-trips byte-for-byte against the previous sRGB-8-bit array while the HDR
 * path passes linear radiance through unclamped:
 *
 *  - **Integer channels** (`Uint8Array` / `DataView` / normalized `Uint16`/…):
 *    treated as sRGB-ENCODED LDR (the same data that previously wrote into the
 *    `rgba8unorm-srgb` array and was hardware-decoded on sample). RGB is decoded
 *    `srgbToLinear` after normalizing to [0,1]; alpha is passed through linearly.
 *    The stored half-float therefore equals the exact linear value the old sRGB
 *    sampler produced → LDR emissive is visually identical.
 *  - **Float channels** (`Float32Array` / `Float64Array`): treated as ALREADY
 *    LINEAR HDR radiance. Values pass straight through with NO sRGB decode and NO
 *    [0,1] clamp, so authored emissive > 1.0 survives packing (the HDR win).
 *
 * Returns half-float (Uint16 bit-pattern) rows for `queue.writeTexture` into an
 * `rgba16float` target.
 */
function normalizeRawTextureUploadFloat(
  data: ArrayBufferView,
  width: number,
  height: number,
  floatInputIsLinear = true,
): NormalizedFloatUpload | null {
  const pixelCount = width * height;
  if (pixelCount <= 0) return null;

  // Integer byte data (Uint8 / DataView) — sRGB-encoded LDR.
  const bytes = byteViewOf(data);
  if (bytes != null) {
    const channels = bytes.byteLength / pixelCount;
    if (![1, 2, 3, 4].includes(channels) || !Number.isInteger(channels)) return null;
    const out = new Uint16Array(pixelCount * 4);
    for (let i = 0; i < pixelCount; i += 1) {
      const src = i * channels;
      const dst = i * 4;
      const r = (bytes[src] ?? 0) / 255;
      const g = channels >= 2 ? (bytes[src + 1] ?? 0) / 255 : r;
      const b = channels >= 3
        ? (bytes[src + 2] ?? 0) / 255
        : channels === 1
          ? r
          : 0;
      const a = channels >= 4 ? (bytes[src + 3] ?? 255) / 255 : 1;
      out[dst] = float32ToFloat16(srgbToLinear(r));
      out[dst + 1] = float32ToFloat16(srgbToLinear(g));
      out[dst + 2] = float32ToFloat16(srgbToLinear(b));
      out[dst + 3] = float32ToFloat16(a);
    }
    return { data: out, bytesPerRow: width * 8, rowsPerImage: height };
  }

  // Numeric typed arrays — floats are linear HDR (pass-through, unclamped);
  // normalized integers are sRGB-encoded LDR (decode like the byte path).
  const maxValue = numericChannelMax(data);
  if (maxValue == null) return null;
  const length = (data as unknown as ArrayLike<number>).length;
  if (length == null) return null;
  const channels = length / pixelCount;
  if (![1, 2, 3, 4].includes(channels) || !Number.isInteger(channels)) return null;
  const isLinearFloat = data instanceof Float32Array && floatInputIsLinear;
  const encodeRgb = (v: number): number =>
    isLinearFloat ? v : srgbToLinear(v / maxValue);
  const encodeAlpha = (v: number): number => (isLinearFloat ? v : v / maxValue);
  const out = new Uint16Array(pixelCount * 4);
  for (let i = 0; i < length; i += 1) {
    const value = numericChannelAt(data, i);
    if (!Number.isFinite(value)) {
      throw new RangeError(
        `[materialTextureArray] raw HDR sample ${i} must be finite.`,
      );
    }
    if (isLinearFloat && Math.abs(value) > 65504) {
      throw new RangeError(
        `[materialTextureArray] raw HDR sample ${i} exceeds finite rgba16float range: ${value}.`,
      );
    }
  }
  for (let i = 0; i < pixelCount; i += 1) {
    const src = i * channels;
    const dst = i * 4;
    const r = numericChannelAt(data, src);
    const g = channels >= 2 ? numericChannelAt(data, src + 1) : r;
    const b = channels >= 3 ? numericChannelAt(data, src + 2) : channels === 1 ? r : 0;
    const a = channels >= 4 ? numericChannelAt(data, src + 3) : maxValue;
    out[dst] = float32ToFloat16(encodeRgb(r));
    out[dst + 1] = float32ToFloat16(encodeRgb(g));
    out[dst + 2] = float32ToFloat16(encodeRgb(b));
    out[dst + 3] = float32ToFloat16(encodeAlpha(a));
  }
  return { data: out, bytesPerRow: width * 8, rowsPerImage: height };
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

export const MATERIAL_TEXTURE_MIPMAP_WGSL = /* wgsl */ `
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

const GPU_MATERIAL_SOURCE_FORMATS: ReadonlySet<string> = new Set([
  'r8unorm',
  'r8snorm',
  'rg8unorm',
  'rg8snorm',
  'rgba8unorm',
  'rgba8unorm-srgb',
  'rgba8snorm',
  'bgra8unorm',
  'bgra8unorm-srgb',
  'r16float',
  'rg16float',
  'rgba16float',
  'r32float',
  'rg32float',
  'rgba32float',
  'rgb10a2unorm',
  'rg11b10ufloat',
  'rgb9e5ufloat',
  'rgba16unorm',
  'rgba16snorm',
]);

export const MATERIAL_TEXTURE_GPU_SOURCE_BLIT_WGSL = /* wgsl */ `
override decodeSrgb: f32 = 0.0;
override sourceChannels: f32 = 4.0;

struct VsOut {
  @builtin(position) position: vec4f,
};

@vertex
fn vsMain(@builtin(vertex_index) vertexIndex: u32) -> VsOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
  );
  var out: VsOut;
  out.position = vec4f(positions[vertexIndex], 0.0, 1.0);
  return out;
}

@group(0) @binding(0) var sourceTexture: texture_2d<f32>;

fn srgbChannelToLinear(value: f32) -> f32 {
  let c = clamp(value, 0.0, 1.0);
  if (c <= 0.04045) { return c / 12.92; }
  return pow((c + 0.055) / 1.055, 2.4);
}

@fragment
fn fsMain(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let sourceSize = textureDimensions(sourceTexture);
  let coord = min(vec2u(position.xy), sourceSize - vec2u(1u));
  var value = textureLoad(sourceTexture, vec2i(coord), 0);
  // Match the public raw-data and shared CPU-readable texture convention: one
  // channel expands to RGB; two channels expand to R,G,0; three channels get
  // opaque alpha. WebGPU textureLoad otherwise supplies zero for absent G/B,
  // which made a cpuMirror-backed emissive source disagree with forward hits.
  if (sourceChannels < 1.5) {
    value = vec4f(value.rrr, 1.0);
  } else if (sourceChannels < 2.5) {
    value = vec4f(value.r, value.g, 0.0, 1.0);
  } else if (sourceChannels < 3.5) {
    value = vec4f(value.rgb, 1.0);
  }
  if (decodeSrgb > 0.5) {
    value = vec4f(
      srgbChannelToLinear(value.r),
      srgbChannelToLinear(value.g),
      srgbChannelToLinear(value.b),
      value.a,
    );
  }
  return value;
}
`;

function gpuMaterialSourceChannelCount(format: GPUTextureFormat): 1 | 2 | 3 | 4 {
  if (
    format.startsWith('rgba') ||
    format.startsWith('bgra') ||
    format === 'rgb10a2unorm'
  ) {
    return 4;
  }
  if (format.startsWith('rgb') || format === 'rg11b10ufloat') return 3;
  if (format.startsWith('rg')) return 2;
  return 1;
}

function validateGpuMaterialSources(
  device: GPUDevice,
  payloads: readonly (ImagePayload | null)[],
  forbiddenResources: ReadonlySet<object>,
): void {
  for (let layer = 0; layer < payloads.length; layer += 1) {
    const source = payloads[layer]?.gpuSource;
    if (source == null) continue;
    if (source.device !== device) {
      throw new Error(
        `[materialTextureArray] GPU source ${layer} belongs to a different GPUDevice.`,
      );
    }
    if (!GPU_MATERIAL_SOURCE_FORMATS.has(source.format)) {
      throw new Error(
        `[materialTextureArray] GPU source ${layer} format ${source.format} is not a ` +
        'float-sampleable color format accepted by the material conversion pass.',
      );
    }
    if ((source.texture.usage & GPUTextureUsage.TEXTURE_BINDING) === 0) {
      throw new Error(
        `[materialTextureArray] GPU source ${layer} lacks GPUTextureUsage.TEXTURE_BINDING.`,
      );
    }
    if (forbiddenResources.has(source.texture)) {
      throw new Error(
        `[materialTextureArray] GPU source ${layer} aliases an engine-owned scene texture.`,
      );
    }
  }
}

function blitGpuMaterialSourceLayer(
  device: GPUDevice,
  source: PtWebgpuTextureSource,
  target: GPUTexture,
  targetFormat: GPUTextureFormat,
  layer: number,
  copyW: number,
  copyH: number,
): void {
  const module = device.createShaderModule({
    label: 'vitrum.pt-webgpu.scene.materialTextures.gpuSource.module',
    code: MATERIAL_TEXTURE_GPU_SOURCE_BLIT_WGSL,
  });
  const bindGroupLayout = device.createBindGroupLayout({
    label: 'vitrum.pt-webgpu.scene.materialTextures.gpuSource.bindings',
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.FRAGMENT,
      texture: {
        sampleType: 'unfilterable-float',
        viewDimension: '2d',
        multisampled: false,
      },
    }],
  });
  const pipeline = device.createRenderPipeline({
    label: 'vitrum.pt-webgpu.scene.materialTextures.gpuSource.pipeline',
    layout: device.createPipelineLayout({
      label: 'vitrum.pt-webgpu.scene.materialTextures.gpuSource.pipelineLayout',
      bindGroupLayouts: [bindGroupLayout],
    }),
    vertex: { module, entryPoint: 'vsMain' },
    fragment: {
      module,
      entryPoint: 'fsMain',
      constants: {
        decodeSrgb: source.colorSpace === 'srgb' && !source.format.endsWith('-srgb') ? 1 : 0,
        sourceChannels: gpuMaterialSourceChannelCount(source.format),
      },
      targets: [{ format: targetFormat }],
    },
    primitive: { topology: 'triangle-list' },
  });
  const sourceView = source.texture.createView({
    label: `vitrum.pt-webgpu.scene.materialTextures.gpuSource.${layer}`,
    dimension: '2d',
    baseMipLevel: source.baseMipLevel,
    mipLevelCount: 1,
    baseArrayLayer: source.arrayLayer,
    arrayLayerCount: 1,
  });
  const bindGroup = device.createBindGroup({
    label: `vitrum.pt-webgpu.scene.materialTextures.gpuSource.${layer}.bindGroup`,
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: sourceView }],
  });
  const encoder = device.createCommandEncoder({
    label: `vitrum.pt-webgpu.scene.materialTextures.gpuSource.${layer}.encoder`,
  });
  const pass = encoder.beginRenderPass({
    label: `vitrum.pt-webgpu.scene.materialTextures.gpuSource.${layer}.pass`,
    colorAttachments: [{
      view: target.createView({
        dimension: '2d',
        baseMipLevel: 0,
        mipLevelCount: 1,
        baseArrayLayer: layer,
        arrayLayerCount: 1,
      }),
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      loadOp: 'clear',
      storeOp: 'store',
    }],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.setViewport(0, 0, copyW, copyH, 0, 1);
  pass.draw(3);
  pass.end();
  device.queue.submit([encoder.finish()]);
}

function generateTextureMips(
  device: GPUDevice,
  texture: GPUTexture,
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

  for (let mip = 1; mip < mipLevelCount; mip += 1) {
      const sourceView = texture.createView({
        dimension: '2d',
        baseMipLevel: mip - 1,
        mipLevelCount: 1,
        baseArrayLayer: 0,
        arrayLayerCount: 1,
      });
      const targetView = texture.createView({
        dimension: '2d',
        baseMipLevel: mip,
        mipLevelCount: 1,
        baseArrayLayer: 0,
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

  device.queue.submit([encoder.finish()]);
}

/**
 * Stage an sRGB-encoded external image into a linear `rgba16float` array layer.
 * `copyExternalImageToTexture` does NOT apply sRGB decode when the destination
 * format is non-sRGB, so we copy the image into a scratch `rgba8unorm-srgb`
 * texture (which the sampler DOES decode) and render it into the float layer.
 * The fragment output is linear, matching what the previous sRGB-8-bit emissive
 * array produced on sample. Values are clamped to [0,1] by the 8-bit source, so
 * only true-HDR (raw-float) emissive exceeds 1.0 — external LDR images stay LDR.
 */
function blitExternalSrgbToFloatLayer(
  device: GPUDevice,
  external: GPUCopyExternalImageSource,
  target: GPUTexture,
  layer: number,
  copyW: number,
  copyH: number,
  forbiddenResources: ReadonlySet<object>,
): void {
  const scratch = device.createTexture({
    label: 'vitrum.pt-webgpu.scene.materialTextures.emissive.srgbStage',
    size: { width: copyW, height: copyH, depthOrArrayLayers: 1 },
    format: 'rgba8unorm-srgb',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  const scratchIsOwned = scratch !== target && !forbiddenResources.has(scratch);
  try {
  if (!scratchIsOwned) {
    throw new Error(
      '[pt-webgpu] emissive staging texture aliased another live texture',
    );
  }
  device.queue.copyExternalImageToTexture(
    { source: external, flipY: false },
    { texture: scratch, origin: { x: 0, y: 0, z: 0 } },
    { width: copyW, height: copyH },
  );
  const module = device.createShaderModule({
    label: 'vitrum.pt-webgpu.scene.materialTextures.emissive.srgbStage.module',
    code: MATERIAL_TEXTURE_MIPMAP_WGSL,
  });
  const sampler = device.createSampler({
    label: 'vitrum.pt-webgpu.scene.materialTextures.emissive.srgbStage.sampler',
    magFilter: 'nearest',
    minFilter: 'nearest',
  });
  const pipeline = device.createRenderPipeline({
    label: 'vitrum.pt-webgpu.scene.materialTextures.emissive.srgbStage.pipeline',
    layout: 'auto',
    vertex: { module, entryPoint: 'vsMain' },
    fragment: { module, entryPoint: 'fsMain', targets: [{ format: 'rgba16float' }] },
    primitive: { topology: 'triangle-list' },
  });
  const bindGroup = device.createBindGroup({
    label: 'vitrum.pt-webgpu.scene.materialTextures.emissive.srgbStage.bindGroup',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: scratch.createView() },
      { binding: 1, resource: sampler },
    ],
  });
  const encoder = device.createCommandEncoder({
    label: 'vitrum.pt-webgpu.scene.materialTextures.emissive.srgbStage.encoder',
  });
  const pass = encoder.beginRenderPass({
    label: 'vitrum.pt-webgpu.scene.materialTextures.emissive.srgbStage.pass',
    colorAttachments: [{
      view: target.createView({
        dimension: '2d',
        baseMipLevel: 0,
        mipLevelCount: 1,
        baseArrayLayer: layer,
        arrayLayerCount: 1,
      }),
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      loadOp: 'clear',
      storeOp: 'store',
    }],
  });
  pass.setPipeline(pipeline);
  // Constrain the draw to the copyW×copyH top-left rect (mirrors the
  // copyExternalImageToTexture origin/extent), so the per-layer UV-fit scale
  // remaps repeat-wrapped UVs into the same source rectangle as the 8-bit path.
  pass.setViewport(0, 0, copyW, copyH, 0, 1);
  pass.setBindGroup(0, bindGroup);
  pass.draw(3);
  pass.end();
  device.queue.submit([encoder.finish()]);
  } finally {
    // The staging texture is a transaction-local allocation.  It must not leak
    // when copy, pipeline construction, encoding, or submission throws.
    try {
      if (scratchIsOwned) scratch.destroy();
    } catch { /* preserve the original outcome */ }
  }
}

/** 1×1 white single-layer array — the always-bound placeholder for scenes with
 *  no sampled textures (kernel never reads it; descriptors are all -1). */
function createDummyArray(
  device: GPUDevice,
  format: GPUTextureFormat,
  forbiddenResources: ReadonlySet<object>,
): MaterialTextureArray {
  const texture = device.createTexture({
    label: DUMMY_LABEL,
    size: { width: 1, height: 1, depthOrArrayLayers: 1 },
    format,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  if (forbiddenResources.has(texture)) {
    throw new Error(
      '[pt-webgpu] material texture candidate aliased an existing scene resource',
    );
  }
  // rgba16float dummy carries a linear-white half-float texel (1.0 → 0x3c00);
  // 8-bit formats keep the byte-white texel. The kernel never samples the dummy
  // (all descriptor indices are -1), so this only satisfies the binding.
  const isFloat = format === 'rgba16float';
  const dummyTexel = isFloat
    ? new Uint16Array([0x3c00, 0x3c00, 0x3c00, 0x3c00])
    : new Uint8Array([255, 255, 255, 255]);
  try {
    device.queue.writeTexture(
      { texture, origin: { x: 0, y: 0, z: 0 } },
      dummyTexel,
      { bytesPerRow: isFloat ? 8 : 4, rowsPerImage: 1 },
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
  } catch (error) {
    try { texture.destroy(); } catch { /* preserve the upload failure */ }
    throw error;
  }
}

type MaterialTextureArrayPreflight = {
  readonly payloads: readonly (ImagePayload | null)[];
  readonly width: number;
  readonly height: number;
  readonly mipLevelCount: number;
  readonly isFloatArray: boolean;
  readonly estimatedPeakBytes: number;
};

function layerUses(
  layerInfos: ReadonlyArray<MaterialTextureLayerInfo> | undefined,
  layer: number,
): readonly MaterialTextureLayerUse[] {
  return layerInfos?.[layer]?.uses ?? [];
}

function preflightMaterialTextureArray(
  device: GPUDevice,
  sources: ReadonlyArray<unknown>,
  format: GPUTextureFormat,
  forbiddenResources: ReadonlySet<object>,
): MaterialTextureArrayPreflight {
  if (
    format !== 'rgba8unorm-srgb' &&
    format !== 'rgba8unorm' &&
    format !== 'rgba16float'
  ) {
    throw new RangeError(
      `[materialTextureArray] unsupported destination format ${format}.`,
    );
  }
  const isFloatArray = format === 'rgba16float';
  if (sources.length === 0) {
    return {
      payloads: [],
      width: 1,
      height: 1,
      mipLevelCount: 1,
      isFloatArray,
      estimatedPeakBytes: isFloatArray ? 8 : 4,
    };
  }
  const maxLayers = positiveSafeDimension(
    device.limits.maxTextureArrayLayers,
    'device maxTextureArrayLayers',
  );
  if (sources.length > maxLayers) {
    throw new RangeError(
      `[materialTextureArray] ${sources.length} layers exceed device maxTextureArrayLayers ${maxLayers}.`,
    );
  }
  const payloads = sources.map(payloadOf);
  validateGpuMaterialSources(device, payloads, forbiddenResources);
  const maxDim = positiveSafeDimension(
    device.limits.maxTextureDimension2D,
    'device maxTextureDimension2D',
  );
  let width = 1;
  let height = 1;
  for (let layer = 0; layer < payloads.length; layer += 1) {
    const payload = payloads[layer];
    if (payload == null) continue;
    if (payload.width > maxDim || payload.height > maxDim) {
      throw new RangeError(
        `[materialTextureArray] source ${layer} dimensions ${payload.width}x${payload.height} exceed ` +
        `device maxTextureDimension2D ${maxDim}; truncation is not permitted.`,
      );
    }
    width = Math.max(width, payload.width);
    height = Math.max(height, payload.height);
  }
  const mipLevelCount = materialTextureMipLevelCount(width, height);
  let mipStorageBytes = 0;
  for (let mip = 0; mip < mipLevelCount; mip += 1) {
    const mipWidth = Math.max(1, Math.floor(width / 2 ** mip));
    const mipHeight = Math.max(1, Math.floor(height / 2 ** mip));
    mipStorageBytes +=
      mipWidth * mipHeight * sources.length * (isFloatArray ? 8 : 4);
  }
  let retainedMirrorBytes = 0;
  let largestDecodedBytes = 0;
  let largestStagingBytes = 0;
  let largestNativeMipStorageBytes = 0;
  for (const payload of payloads) {
    if (payload == null) continue;
    const pixelCount = payload.width * payload.height;
    if (!Number.isSafeInteger(pixelCount)) {
      throw new RangeError('[materialTextureArray] source pixel count exceeds safe integer range.');
    }
    if (payload.data != null) {
      largestDecodedBytes = Math.max(
        largestDecodedBytes,
        pixelCount * (isFloatArray ? 8 : 4),
      );
    }
    if (isFloatArray && payload.external != null) {
      largestStagingBytes = Math.max(largestStagingBytes, pixelCount * 4);
    }
    const nativeMipCount = materialTextureMipLevelCount(payload.width, payload.height);
    let nativeMipBytes = 0;
    for (let mip = 0; mip < nativeMipCount; mip += 1) {
      nativeMipBytes +=
        Math.max(1, Math.floor(payload.width / 2 ** mip)) *
        Math.max(1, Math.floor(payload.height / 2 ** mip)) *
        (isFloatArray ? 8 : 4);
    }
    largestNativeMipStorageBytes = Math.max(
      largestNativeMipStorageBytes,
      nativeMipBytes,
    );
    const mirror = payload.gpuSource?.cpuMirror;
    if (mirror != null) {
      const bytesPerElement = mirror.dataType === 'uint8'
        ? 1
        : mirror.dataType === 'float32'
          ? 4
          : 2;
      retainedMirrorBytes += mirror.data.length * bytesPerElement;
      if (isFloatArray) {
        largestDecodedBytes = Math.max(
          largestDecodedBytes,
          pixelCount * 8,
        );
      }
    }
  }
  const estimatedPeakBytes =
    mipStorageBytes +
    retainedMirrorBytes +
    largestDecodedBytes +
    largestStagingBytes +
    largestNativeMipStorageBytes;
  if (
    !Number.isSafeInteger(estimatedPeakBytes) ||
    estimatedPeakBytes > MATERIAL_TEXTURE_ARRAY_PEAK_BUDGET_BYTES
  ) {
    throw new RangeError(
      `[materialTextureArray] estimated array upload peak ${estimatedPeakBytes} bytes exceeds ` +
      `${MATERIAL_TEXTURE_ARRAY_PEAK_BUDGET_BYTES}-byte budget ` +
      '(destination mips + CPU mirrors + decoded/staging peak + one native mip chain).',
    );
  }
  return { payloads, width, height, mipLevelCount, isFloatArray, estimatedPeakBytes };
}

/** Read-only validation/estimation used to reject a three-atlas upload before its first allocation. */
export function estimateMaterialTextureArrayPeakBytes(
  device: GPUDevice,
  sources: ReadonlyArray<unknown>,
  format: GPUTextureFormat,
  forbiddenResources: ReadonlySet<object> = new Set(),
): number {
  return preflightMaterialTextureArray(
    device, sources, format, forbiddenResources,
  ).estimatedPeakBytes;
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
  forbiddenResources: ReadonlySet<object> = new Set(),
  radianceEnvelope?: MaterialTextureRadianceEnvelope,
): MaterialTextureArray {
  const preflight = preflightMaterialTextureArray(
    device, sources, format, forbiddenResources,
  );
  if (sources.length === 0) return createDummyArray(device, format, forbiddenResources);

  const warnings: string[] = [];
  const structuredWarnings: MaterialTextureArrayWarning[] = [];
  const { payloads, width, height, mipLevelCount, isFloatArray } = preflight;
  // rgba16float is a LINEAR (non-sRGB) format. Raw-data uploads apply the sRGB
  // decode on the CPU (normalizeRawTextureUploadFloat); external images (which
  // carry sRGB-encoded 8-bit samples and get NO hardware sRGB decode when copied
  // into a float target) are staged through an rgba8unorm-srgb texture and blit
  // via a render pass so the sampler applies the sRGB→linear decode — keeping LDR
  // emissive visually identical to the previous sRGB-8-bit array path.
  // Decode/validate every raw payload before allocating the destination. A
  // malformed typed layout, non-finite float or lossy rgba8 float is rejected
  // synchronously and cannot leave a partially black material layer behind.
  const normalizedUploads = payloads.map((p, layer) => {
    if (p == null) return null;
    if (p.data == null) {
      // Emissive GPU sources already require an immutable exact cpuMirror for
      // NEE. Upload that same snapshot into the rgba16float array so CPU
      // preflight and shader sampling share the identical CPU sRGB conversion
      // and half quantization. This removes native-sRGB/WGSL-pow rounding drift
      // at extreme scalar×texel envelopes.
      if (isFloatArray && p.gpuSource?.cpuMirror != null) {
        return normalizeGpuMirrorFloatUpload(p.gpuSource);
      }
      return null;
    }
    const upload = isFloatArray
      ? normalizeRawTextureUploadFloat(
          p.data,
          p.width,
          p.height,
          (layerInfos?.[layer]?.uses.length ?? 0) === 0 ||
            (layerInfos?.[layer]?.uses.every((use) => use.field === 'emissiveMap') ?? false),
        )
      : normalizeRawTextureUpload(p.data, p.width, p.height);
    if (upload == null) {
      throw new TypeError(
        `[materialTextureArray] source ${layer} has unsupported raw layout ` +
        `(${p.data.constructor.name}, ${p.data.byteLength} bytes for ${p.width}x${p.height}); ` +
        'accepted payloads are tightly packed Uint8/Uint8Clamped, normalized Uint16, ' +
        'or Float32 with exactly 1, 2, 3, or 4 channels per pixel.',
      );
    }
    return upload;
  });
  validateMaterialTextureRadianceEnvelope(
    payloads,
    normalizedUploads,
    format,
    layerInfos,
    radianceEnvelope,
  );

  const texture = device.createTexture({
    label: ARRAY_LABEL,
    size: { width, height, depthOrArrayLayers: sources.length },
    mipLevelCount,
    format,
    // Native source mip chains are copied into this max-size array.
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST,
  });
  if (forbiddenResources.has(texture)) {
    try { texture.destroy(); } catch { /* preserve alias failure */ }
    throw new Error(
      '[pt-webgpu] material texture candidate aliased an existing scene resource',
    );
  }

  try {
  for (let layer = 0; layer < payloads.length; layer += 1) {
    const p = payloads[layer];
    if (p == null) {
      const warning =
        `[materialTextureArray] source ${layer} has no usable image; layer left black.`;
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
    const copyW = p.width;
    const copyH = p.height;
    const nativeMipLevelCount = materialTextureMipLevelCount(copyW, copyH);
    const nativeTexture = device.createTexture({
      label: `${ARRAY_LABEL}.native.${layer}`,
      size: { width: copyW, height: copyH, depthOrArrayLayers: 1 },
      mipLevelCount: nativeMipLevelCount,
      format,
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    if (forbiddenResources.has(nativeTexture) || nativeTexture === texture) {
      try { nativeTexture.destroy(); } catch { /* preserve alias failure */ }
      throw new Error(
        '[pt-webgpu] native material texture staging aliased a live scene resource',
      );
    }
    try {
      const normalizedUpload = normalizedUploads[layer];
      if (
        normalizedUpload != null &&
        (p.data != null || (isFloatArray && p.gpuSource?.cpuMirror != null))
      ) {
        device.queue.writeTexture(
          { texture: nativeTexture, origin: { x: 0, y: 0, z: 0 } },
          normalizedUpload.data,
          {
            bytesPerRow: normalizedUpload.bytesPerRow,
            rowsPerImage: normalizedUpload.rowsPerImage,
          },
          { width: copyW, height: copyH },
        );
      } else if (p.gpuSource != null) {
        blitGpuMaterialSourceLayer(
          device, p.gpuSource, nativeTexture, format, 0, copyW, copyH,
        );
      } else if (p.external != null) {
        if (isFloatArray) {
          blitExternalSrgbToFloatLayer(
            device,
            p.external,
            nativeTexture,
            0,
            copyW,
            copyH,
            new Set([...forbiddenResources, texture]),
          );
        } else {
          device.queue.copyExternalImageToTexture(
            { source: p.external, flipY: false },
            { texture: nativeTexture, origin: { x: 0, y: 0, z: 0 } },
            { width: copyW, height: copyH },
          );
        }
      }
      generateTextureMips(device, nativeTexture, format, nativeMipLevelCount);
      const copyEncoder = device.createCommandEncoder({
        label: `${ARRAY_LABEL}.native.${layer}.copy`,
      });
      for (let mip = 0; mip < nativeMipLevelCount; mip += 1) {
        copyEncoder.copyTextureToTexture(
          { texture: nativeTexture, mipLevel: mip },
          { texture, mipLevel: mip, origin: { x: 0, y: 0, z: layer } },
          {
            width: Math.max(1, Math.floor(copyW / 2 ** mip)),
            height: Math.max(1, Math.floor(copyH / 2 ** mip)),
            depthOrArrayLayers: 1,
          },
        );
      }
      device.queue.submit([copyEncoder.finish()]);
    } finally {
      try { nativeTexture.destroy(); } catch { /* preserve the upload outcome */ }
    }
  }

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
  } catch (error) {
    // createMaterialTextureArray owns the texture until it returns.  Outer scene
    // upload tracking cannot see it during a failed copy/mip/view stage.
    try { texture.destroy(); } catch { /* preserve the upload failure */ }
    throw error;
  }
}
