// materialTextureArray.ts — P2 GPU upload of collected material textures into a
// single `texture_2d_array` consumed by the full-tier path-trace kernel.
//
// `collectMaterialTextures` (materialTextures.ts) dedups the host texture
// handles into upload-ordered `sources` lists; this module turns each list into
// one sampled format-specific 2D-array (one source per array layer) plus a
// filtering sampler. Bounded colour maps use rgba8 formats, while outgoing
// radiance uses linear rgba16float. WGSL indexes a layer through the matching
// per-material descriptor and samples with the interpolated hit UV.
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

import {
  materialTextureRoleProfile,
  type MaterialTextureLayerInfo,
  type MaterialTextureLayerUse,
  type MaterialTextureRoleProfile,
} from './materialTextures.js';
import {
  halfToFloat,
  packRadianceRgbProductF32,
  type RadianceRgb,
} from '@vitrum/shared-bvh';
import {
  isPtWebgpuTextureSource,
  type PtWebgpuTextureSource,
  type PtWebgpuTextureCpuMirrorDataType,
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
  readonly gpuSource?: StagedPtWebgpuTextureSource;
  /** Engine-owned immutable CPU bytes retained by this unique staged handle. */
  readonly stagedByteLength: number;
}

interface StagedPtWebgpuTextureCpuMirror {
  readonly width: number;
  readonly height: number;
  readonly channels: 1 | 2 | 3 | 4;
  readonly dataType: PtWebgpuTextureCpuMirrorDataType;
  readonly colorSpace: 'srgb' | 'linear';
  readonly data: Uint8Array<ArrayBuffer> | Uint16Array<ArrayBuffer> | Float32Array<ArrayBuffer>;
}

/**
 * One-observation snapshot of the descriptor around a host-owned GPU texture.
 * The texture reference and its pixels remain host-owned and must not be
 * mutated until the synchronous staging/upload call returns; every observable
 * descriptor scalar and cpuMirror element is captured into this token once.
 */
interface StagedPtWebgpuTextureSource {
  readonly device: GPUDevice;
  readonly texture: GPUTexture;
  readonly textureUsage: GPUTextureUsageFlags;
  readonly format: GPUTextureFormat;
  readonly colorSpace: 'srgb' | 'linear';
  readonly baseMipLevel: number;
  readonly arrayLayer: number;
  readonly width: number;
  readonly height: number;
  readonly cpuMirror?: StagedPtWebgpuTextureCpuMirror;
}

export interface MaterialTextureArrayUploadRequest {
  readonly sources: ReadonlyArray<unknown>;
  readonly format: GPUTextureFormat;
  readonly layerInfos?: ReadonlyArray<MaterialTextureLayerInfo>;
  readonly radianceEnvelope?: MaterialTextureRadianceEnvelope;
}

/** Opaque, immutable input to one material-atlas creation. */
export interface StagedMaterialTextureArrayUpload {
  readonly estimatedPeakBytes: number;
  readonly layerCount: number;
  readonly format: GPUTextureFormat;
}

/**
 * Scene-level transaction token. All source lists are observed once, identity
 * aliases share one owned CPU snapshot, and aggregate admission happens before
 * the first GPU allocation.
 */
export interface MaterialTextureUploadPlan {
  readonly arrays: readonly StagedMaterialTextureArrayUpload[];
  readonly estimatedPeakBytes: number;
}

/** Conservative per-array peak for destination mips + retained/decoded inputs. */
export const MATERIAL_TEXTURE_ARRAY_PEAK_BUDGET_BYTES = 512 * 1024 * 1024;

function snapshotMaterialTextureSources(
  sources: ReadonlyArray<unknown>,
  maxLayers: number,
): readonly unknown[] {
  if (sources == null || typeof sources !== 'object') {
    throw new TypeError('[materialTextureArray] sources must be an array-like object.');
  }
  const length = sources.length;
  if (!Number.isSafeInteger(length) || length < 0) {
    throw new RangeError(
      `[materialTextureArray] sources.length must be a non-negative safe integer; ` +
      `received ${String(length)}.`,
    );
  }
  const layerLimit = length === 0 ? 0 : maxLayers;
  if (length > layerLimit) {
    throw new RangeError(
      `[materialTextureArray] ${length} layers exceed device maxTextureArrayLayers ` +
      `${layerLimit}.`,
    );
  }
  const snapshot = new Array<unknown>(length);
  for (let layer = 0; layer < length; layer += 1) {
    snapshot[layer] = sources[layer];
  }
  return snapshot;
}

interface MaterialTextureSourceStageContext {
  readonly sourcePayloads: Map<unknown, ImagePayload | null>;
  readonly imagePayloads: Map<object, ImagePayload | null>;
  ownedSnapshotBytes: number;
}

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

function reserveOwnedSnapshotBytes(
  context: MaterialTextureSourceStageContext,
  byteLength: number,
  label: string,
): void {
  if (!Number.isSafeInteger(byteLength) || byteLength < 0) {
    throw new RangeError(`[materialTextureArray] ${label} byte length is not a safe integer.`);
  }
  const next = context.ownedSnapshotBytes + byteLength;
  if (
    !Number.isSafeInteger(next) ||
    next > MATERIAL_TEXTURE_ARRAY_PEAK_BUDGET_BYTES
  ) {
    throw new RangeError(
      `[materialTextureArray] immutable source staging requires ${next} bytes; ` +
      `the scene-level budget is ${MATERIAL_TEXTURE_ARRAY_PEAK_BUDGET_BYTES} bytes.`,
    );
  }
  context.ownedSnapshotBytes = next;
}

function assertUnsharedArrayBufferView(input: unknown, label: string): void {
  if (!ArrayBuffer.isView(input)) return;
  const inputBuffer = input.buffer;
  if (
    typeof SharedArrayBuffer !== 'undefined' &&
    inputBuffer instanceof SharedArrayBuffer
  ) {
    throw new TypeError(
      `[materialTextureArray] SharedArrayBuffer-backed ${label} is not accepted; ` +
      'a concurrently mutable buffer cannot provide an exact scene-upload snapshot.',
    );
  }
}

function snapshotRawTextureData(
  input: ArrayBufferView,
  context: MaterialTextureSourceStageContext,
): ArrayBufferView {
  assertUnsharedArrayBufferView(input, 'material texels');
  const inputLength = (input as unknown as ArrayLike<number>).length;
  const inputByteLength = input.byteLength;
  if (!Number.isSafeInteger(inputLength) || (inputLength as number) < 0) {
    throw new TypeError(
      '[materialTextureArray] raw material data must be Uint8, Uint8Clamped, Uint16, or Float32.',
    );
  }
  reserveOwnedSnapshotBytes(context, inputByteLength, 'raw material data');
  let snapshot:
    | Uint8Array<ArrayBuffer>
    | Uint8ClampedArray<ArrayBuffer>
    | Uint16Array<ArrayBuffer>
    | Float32Array<ArrayBuffer>;
  if (input instanceof Uint8ClampedArray) {
    snapshot = new Uint8ClampedArray(inputLength);
  } else if (input instanceof Uint8Array) {
    snapshot = new Uint8Array(inputLength);
  } else if (input instanceof Uint16Array) {
    snapshot = new Uint16Array(inputLength);
  } else if (input instanceof Float32Array) {
    snapshot = new Float32Array(inputLength);
  } else {
    throw new TypeError(
      '[materialTextureArray] raw material data must be Uint8, Uint8Clamped, Uint16, or Float32.',
    );
  }
  if (snapshot.byteLength !== inputByteLength) {
    throw new TypeError(
      '[materialTextureArray] raw material typed-array byte length changed during staging.',
    );
  }
  for (let index = 0; index < inputLength; index += 1) {
    snapshot[index] = Number((input as unknown as ArrayLike<number>)[index]);
  }
  return snapshot;
}

function snapshotGpuCpuMirror(
  input: NonNullable<PtWebgpuTextureSource['cpuMirror']>,
  sourceWidth: number,
  sourceHeight: number,
  sourceFormat: GPUTextureFormat,
  sourceColorSpace: 'srgb' | 'linear',
  context: MaterialTextureSourceStageContext,
): StagedPtWebgpuTextureCpuMirror {
  const width = input.width;
  const height = input.height;
  const channels = input.channels;
  const dataType = input.dataType;
  const colorSpace = input.colorSpace;
  const data = input.data;
  // Genuine wrapped sources already contain an ordinary-ArrayBuffer snapshot.
  // Keep this second check at the staging copy boundary so no future descriptor
  // path can reintroduce concurrently mutable mirror bytes.
  assertUnsharedArrayBufferView(data, 'GPU source cpuMirror data');
  if (width !== sourceWidth || height !== sourceHeight) {
    throw new RangeError(
      '[materialTextureArray] GPU source cpuMirror dimensions changed after source creation.',
    );
  }
  if (channels !== 1 && channels !== 2 && channels !== 3 && channels !== 4) {
    throw new RangeError('[materialTextureArray] GPU source cpuMirror channel count is invalid.');
  }
  if (channels !== gpuMaterialSourceChannelCount(sourceFormat)) {
    throw new RangeError(
      '[materialTextureArray] GPU source cpuMirror channels no longer match its format.',
    );
  }
  if (
    dataType !== 'uint8' &&
    dataType !== 'uint16' &&
    dataType !== 'float16' &&
    dataType !== 'half-float' &&
    dataType !== 'float32'
  ) {
    throw new RangeError('[materialTextureArray] GPU source cpuMirror data type is invalid.');
  }
  if (colorSpace !== sourceColorSpace) {
    throw new RangeError(
      '[materialTextureArray] GPU source cpuMirror color space changed after source creation.',
    );
  }
  const dataLength = data.length;
  const pixelCount = sourceWidth * sourceHeight;
  const expectedLength = pixelCount * channels;
  if (
    !Number.isSafeInteger(pixelCount) ||
    !Number.isSafeInteger(expectedLength) ||
    dataLength !== expectedLength
  ) {
    throw new RangeError('[materialTextureArray] GPU source cpuMirror length is invalid.');
  }
  const bytesPerElement = dataType === 'uint8' ? 1 : dataType === 'float32' ? 4 : 2;
  const byteLength = expectedLength * bytesPerElement;
  reserveOwnedSnapshotBytes(context, byteLength, 'GPU cpuMirror');
  const snapshot = dataType === 'uint8'
    ? new Uint8Array(expectedLength)
    : dataType === 'float32'
      ? new Float32Array(expectedLength)
      : new Uint16Array(expectedLength);
  const integerMax = dataType === 'uint8' ? 255 : dataType === 'float32' ? null : 65535;
  for (let index = 0; index < expectedLength; index += 1) {
    const value = Number(data[index]);
    if (!Number.isFinite(value)) {
      throw new RangeError(
        `[materialTextureArray] GPU source cpuMirror sample ${index} must be finite.`,
      );
    }
    if (dataType === 'float32') {
      const packed = Math.fround(value);
      if (!Number.isFinite(packed) || (value !== 0 && packed === 0)) {
        throw new RangeError(
          `[materialTextureArray] GPU source cpuMirror sample ${index} is not exactly stageable as float32.`,
        );
      }
      snapshot[index] = packed;
    } else {
      if (!Number.isInteger(value) || value < 0 || value > integerMax!) {
        throw new RangeError(
          `[materialTextureArray] GPU source cpuMirror sample ${index} is outside ${dataType}.`,
        );
      }
      snapshot[index] = value;
    }
  }
  return Object.freeze({ width, height, channels, dataType, colorSpace, data: snapshot });
}

function snapshotGpuTextureSource(
  input: PtWebgpuTextureSource,
  context: MaterialTextureSourceStageContext,
): StagedPtWebgpuTextureSource {
  const device = input.device;
  const texture = input.texture;
  const format = input.format;
  const colorSpace = input.colorSpace;
  const baseMipLevel = input.baseMipLevel;
  const arrayLayer = input.arrayLayer;
  const width = positiveSafeDimension(input.width, 'GPU source width');
  const height = positiveSafeDimension(input.height, 'GPU source height');
  const cpuMirrorInput = input.cpuMirror;
  if (texture == null || typeof texture !== 'object') {
    throw new TypeError('[materialTextureArray] GPU source texture reference is invalid.');
  }
  const textureUsage = texture.usage;
  if (colorSpace !== 'srgb' && colorSpace !== 'linear') {
    throw new RangeError('[materialTextureArray] GPU source colorSpace must be srgb or linear.');
  }
  if (!Number.isSafeInteger(baseMipLevel) || baseMipLevel < 0) {
    throw new RangeError('[materialTextureArray] GPU source baseMipLevel is invalid.');
  }
  if (!Number.isSafeInteger(arrayLayer) || arrayLayer < 0) {
    throw new RangeError('[materialTextureArray] GPU source arrayLayer is invalid.');
  }
  const cpuMirror = cpuMirrorInput == null
    ? undefined
    : snapshotGpuCpuMirror(
      cpuMirrorInput,
      width,
      height,
      format,
      colorSpace,
      context,
    );
  return Object.freeze({
    device,
    texture,
    textureUsage,
    format,
    colorSpace,
    baseMipLevel,
    arrayLayer,
    width,
    height,
    ...(cpuMirror == null ? {} : { cpuMirror }),
  });
}

/** Duck-type a host texture handle into an upload payload, or null if unusable. */
function payloadOf(
  source: unknown,
  context: MaterialTextureSourceStageContext,
): ImagePayload | null {
  if (context.sourcePayloads.has(source)) {
    return context.sourcePayloads.get(source) ?? null;
  }
  if (isPtWebgpuTextureSource(source)) {
    const gpuSource = snapshotGpuTextureSource(source, context);
    const payload = Object.freeze({
      width: gpuSource.width,
      height: gpuSource.height,
      gpuSource,
      stagedByteLength: gpuSource.cpuMirror?.data.byteLength ?? 0,
    });
    context.sourcePayloads.set(source, payload);
    return payload;
  }
  if (looksLikeRawGpuTexture(source)) {
    throw new TypeError(
      '[materialTextureArray] raw GPUTexture handles are ambiguous. Wrap the texture ' +
      'with createPtWebgpuTextureSource so device, format, color space, and selected ' +
      'subresource are explicit.',
    );
  }
  if (source == null || typeof source !== 'object') {
    context.sourcePayloads.set(source, null);
    return null;
  }
  // THREE.Texture-like: unwrap `.image`; otherwise treat the source as the image.
  const imageValue = 'image' in source
    ? (source as { image?: unknown }).image
    : undefined;
  const img = (imageValue != null ? imageValue : source) as Record<string, unknown>;
  if (img == null || typeof img !== 'object') {
    context.sourcePayloads.set(source, null);
    return null;
  }
  if (context.imagePayloads.has(img)) {
    const cached = context.imagePayloads.get(img) ?? null;
    context.sourcePayloads.set(source, cached);
    return cached;
  }
  const width = img.width;
  const height = img.height;
  if (
    !Number.isSafeInteger(width) ||
    !Number.isSafeInteger(height) ||
    (width as number) < 1 ||
    (height as number) < 1
  ) {
    context.imagePayloads.set(img, null);
    context.sourcePayloads.set(source, null);
    return null;
  }
  // DataTexture-style { data, width, height } → writeTexture.
  const imageData = img.data;
  if (ArrayBuffer.isView(imageData)) {
    const data = snapshotRawTextureData(imageData, context);
    const payload = Object.freeze({
      width: width as number,
      height: height as number,
      data,
      stagedByteLength: data.byteLength,
    });
    context.imagePayloads.set(img, payload);
    context.sourcePayloads.set(source, payload);
    return payload;
  }
  // ImageBitmap / HTMLCanvasElement / HTMLImageElement / OffscreenCanvas /
  // VideoFrame — all valid copyExternalImageToTexture sources. We can't
  // `instanceof`-check headlessly, so accept any object with positive
  // dimensions that isn't raw data and let the device validate it.
  const payload = Object.freeze({
    width: width as number,
    height: height as number,
    external: img as unknown as GPUCopyExternalImageSource,
    stagedByteLength: 0,
  });
  context.imagePayloads.set(img, payload);
  context.sourcePayloads.set(source, payload);
  return payload;
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

interface NormalizedFloatMipUpload extends NormalizedFloatUpload {
  readonly width: number;
  readonly height: number;
}

function layerRoleProfile(
  layerInfos: ReadonlyArray<MaterialTextureLayerInfo> | undefined,
  layer: number,
): MaterialTextureRoleProfile | null {
  const profiles = new Set(
    (layerInfos?.[layer]?.uses ?? []).map((use) => materialTextureRoleProfile(use.field)),
  );
  if (profiles.size > 1) {
    throw new TypeError(
      `[materialTextureArray] layer ${layer} mixes incompatible material-map source profiles; ` +
      'collect each profile into a deterministic duplicate layer.',
    );
  }
  return profiles.values().next().value ?? null;
}

function assertLayerChannelPolicy(
  layerInfos: ReadonlyArray<MaterialTextureLayerInfo> | undefined,
  layer: number,
  channels: number,
): void {
  for (const use of layerInfos?.[layer]?.uses ?? []) {
    const allowed = use.field === 'metallicMap'
      ? channels === 1 || channels === 3 || channels === 4
      : use.field === 'sheenRoughnessMap' || use.field === 'specularIntensityMap'
        ? channels === 4
        : materialTextureRoleProfile(use.field) === 'normal' ||
            materialTextureRoleProfile(use.field) === 'anisotropy'
          ? channels === 3 || channels === 4
          : channels >= 1 && channels <= 4;
    if (!allowed) {
      throw new TypeError(
        `[materialTextureArray] layer ${layer} ${use.field} cannot consume an authored ` +
        `${channels}-channel source without inventing or discarding a required channel.`,
      );
    }
  }
}

function validateLayerPayloadPolicies(
  payloads: readonly (ImagePayload | null)[],
  layerInfos: ReadonlyArray<MaterialTextureLayerInfo> | undefined,
): void {
  for (let layer = 0; layer < payloads.length; layer += 1) {
    const profile = layerRoleProfile(layerInfos, layer);
    const payload = payloads[layer];
    if (payload == null) {
      continue;
    }
    if (payload.external != null && profile === 'alpha-scalar') {
      throw new TypeError(
        `[materialTextureArray] layer ${layer} requires an explicitly authored alpha channel; ` +
        'an external image does not expose whether opaque alpha was synthesized by its decoder.',
      );
    }
    const gpuSource = payload.gpuSource;
    if (gpuSource != null) {
      const channels = gpuMaterialSourceChannelCount(gpuSource.format);
      assertLayerChannelPolicy(layerInfos, layer, channels);
      const isSnorm = gpuSource.format.endsWith('snorm');
      const uses = layerUses(layerInfos, layer);
      for (const use of uses) {
        if (use.colorSpace !== gpuSource.colorSpace) {
          throw new TypeError(
            `[materialTextureArray] GPU source ${layer} declares ${gpuSource.colorSpace} ` +
            `but ${use.field} requires ${use.colorSpace}.`,
          );
        }
      }
      if (
        profile != null &&
        profile !== 'color' &&
        profile !== 'radiance' &&
        gpuSource.colorSpace !== 'linear'
      ) {
        throw new TypeError(
          `[materialTextureArray] ${profile} GPU source ${layer} must use linear color space.`,
        );
      }
      if (isSnorm && gpuSource.colorSpace !== 'linear') {
        throw new TypeError(
          `[materialTextureArray] signed-normalized GPU source ${layer} must use linear color space.`,
        );
      }
      if (isSnorm && profile !== 'normal') {
        throw new TypeError(
          `[materialTextureArray] signed-normalized GPU source ${layer} is only valid for normal maps.`,
        );
      }
    }
  }
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
 * IEEE-754 float32 → float16 (half) bit pattern, returned as a u16. The
 * mantissa is rounded to the nearest representable magnitude (halfway cases
 * round upward in magnitude). Handles subnormals, the subnormal→normal carry,
 * overflow→±inf, and NaN. rgba16float half-floats carry values well beyond
 * 1.0 (max finite ≈ 65504), so HDR emissive > 1.0 survives.
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
    // The upper half-ulp below 2^-14 rounds to the minimum normal half. Do not
    // mask 1024 back to a zero subnormal.
    if (mant >= 0x0400) return sign | 0x0400;
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
  source: StagedPtWebgpuTextureSource,
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
  source: StagedPtWebgpuTextureSource,
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
  source: StagedPtWebgpuTextureSource,
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
  normalizedFloatMipUploads: readonly (
    readonly NormalizedFloatMipUpload[] | null
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
      throw new TypeError(
        `[materialTextureArray] outgoing-radiance source ${layer} has no exact ` +
        'CPU-readable texels for radiance-envelope validation.',
      );
    }
    const validateLevel = (
      mip: number,
      width: number,
      height: number,
      rgbAt: (texel: number) => [number, number, number],
    ): void => {
      const texelCount = width * height;
      for (let texel = 0; texel < texelCount; texel += 1) {
        const rgb = rgbAt(texel);
        for (const { use, factor } of factorUses) {
          packRadianceRgbProductF32(
            factor,
            rgb,
            `@vitrum/pt-webgpu material ${use.materialIndex} ${use.field} ` +
              `layer ${layer} mip ${mip} texel ${texel} radiance`,
          );
        }
      }
    };
    const mipChain = normalizedFloatMipUploads[layer];
    if (mipChain != null) {
      mipChain.forEach((level, mip) => {
        validateLevel(mip, level.width, level.height, (texel) =>
          exactNormalizedUploadRgb(level, format, texel));
      });
    } else if (normalized != null) {
      validateLevel(0, payload.width, payload.height, (texel) =>
        exactNormalizedUploadRgb(normalized, format, texel));
    } else {
      validateLevel(0, payload.width, payload.height, (texel) =>
        exactGpuMirrorRgb(payload.gpuSource!, format, texel));
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
  layerInfos: ReadonlyArray<MaterialTextureLayerInfo> | undefined,
  layer: number,
): NormalizedRgba8Upload | null {
  const bytes = byteViewOf(data);
  if (bytes == null) return null;
  const pixelCount = width * height;
  if (pixelCount <= 0) return null;
  const channels = bytes.byteLength / pixelCount;
  if (![1, 2, 3, 4].includes(channels) || !Number.isInteger(channels)) return null;
  assertLayerChannelPolicy(layerInfos, layer, channels);
  if (channels === 4) return { data: bytes, bytesPerRow: width * 4, rowsPerImage: height };
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
  layerInfos: ReadonlyArray<MaterialTextureLayerInfo> | undefined,
  layer: number,
): NormalizedRgba8Upload | null {
  const maxValue = numericChannelMax(data);
  if (maxValue == null) return null;
  const length = (data as unknown as ArrayLike<number>).length;
  const pixelCount = width * height;
  if (pixelCount <= 0 || length == null) return null;
  const channels = length / pixelCount;
  if (![1, 2, 3, 4].includes(channels) || !Number.isInteger(channels)) return null;
  assertLayerChannelPolicy(layerInfos, layer, channels);
  const rgba = new Uint8Array(pixelCount * 4);
  for (let i = 0; i < pixelCount; i += 1) {
    const src = i * channels;
    const dst = i * 4;
    const authored: number[] = [];
    for (let channel = 0; channel < channels; channel += 1) {
      const sampleIndex = src + channel;
      const value = numericChannelAt(data, sampleIndex);
      if (!Number.isFinite(value)) {
        throw new RangeError(`[materialTextureArray] raw sample ${sampleIndex} must be finite.`);
      }
      if (data instanceof Float32Array && (value < 0 || value > 1)) {
        throw new RangeError(
          `[materialTextureArray] rgba8 float sample ${sampleIndex} must be in [0, 1]; ` +
          `received ${value}.`,
        );
      }
      authored.push(value);
    }
    const r = authored[0] ?? 0;
    const g = channels >= 2 ? authored[1]! : r;
    const b = channels >= 3 ? authored[2]! : channels === 1 ? r : 0;
    const a = channels >= 4 ? authored[3]! : maxValue;
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
  layerInfos: ReadonlyArray<MaterialTextureLayerInfo> | undefined,
  layer: number,
): NormalizedRgba8Upload | null {
  return normalizeRawRgba8(data, width, height, layerInfos, layer) ??
    normalizeRawNumericRgba8(data, width, height, layerInfos, layer);
}

/**
 * Raw-data upload path for the `rgba16float` outgoing-radiance array. Two
 * colour-space conventions keep the LDR emissive path byte-compatible while
 * preserving linear light maps and HDR payloads:
 *
 *  - **Integer channels** are normalized to [0,1], then either sRGB-decoded for
 *    emissiveMap or retained as linear for lightMap according to layer provenance.
 *    Alpha remains linear in both cases.
 *  - **Float32 channels** are always ALREADY-LINEAR HDR radiance. Values pass
 *    through with no transfer decode and no [0,1] clamp.
 *
 * Returns half-float (Uint16 bit-pattern) rows for `queue.writeTexture` into an
 * `rgba16float` target.
 */
function normalizeRawTextureUploadFloat(
  data: ArrayBufferView,
  width: number,
  height: number,
  integerInputColorSpace: 'srgb' | 'linear' = 'srgb',
  layerInfos?: ReadonlyArray<MaterialTextureLayerInfo>,
  layer = 0,
): NormalizedFloatUpload | null {
  const pixelCount = width * height;
  if (pixelCount <= 0) return null;

  // Integer byte data — transfer policy comes from the consuming map role.
  const bytes = byteViewOf(data);
  if (bytes != null) {
    const channels = bytes.byteLength / pixelCount;
    if (![1, 2, 3, 4].includes(channels) || !Number.isInteger(channels)) return null;
    assertLayerChannelPolicy(layerInfos, layer, channels);
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
      const decode = integerInputColorSpace === 'srgb' ? srgbToLinear : (value: number) => value;
      out[dst] = float32ToFloat16(decode(r));
      out[dst + 1] = float32ToFloat16(decode(g));
      out[dst + 2] = float32ToFloat16(decode(b));
      out[dst + 3] = float32ToFloat16(a);
    }
    return { data: out, bytesPerRow: width * 8, rowsPerImage: height };
  }

  // Numeric typed arrays — floats are linear HDR (pass-through, unclamped);
  // normalized integers follow the consuming map's transfer policy.
  const maxValue = numericChannelMax(data);
  if (maxValue == null) return null;
  const length = (data as unknown as ArrayLike<number>).length;
  if (length == null) return null;
  const channels = length / pixelCount;
  if (![1, 2, 3, 4].includes(channels) || !Number.isInteger(channels)) return null;
  assertLayerChannelPolicy(layerInfos, layer, channels);
  // Unhinted Float32 material-radiance payloads are always linear HDR. Transfer
  // policy applies only to normalized integer samples.
  const isLinearFloat = data instanceof Float32Array;
  const encodeRgb = (v: number): number =>
    isLinearFloat
      ? v
      : integerInputColorSpace === 'srgb'
        ? srgbToLinear(v / maxValue)
        : v / maxValue;
  const encodeAlpha = (v: number): number => (isLinearFloat ? v : v / maxValue);
  const out = new Uint16Array(pixelCount * 4);
  for (let i = 0; i < pixelCount; i += 1) {
    const src = i * channels;
    const dst = i * 4;
    const authored: number[] = [];
    for (let channel = 0; channel < channels; channel += 1) {
      const sampleIndex = src + channel;
      const value = numericChannelAt(data, sampleIndex);
      if (!Number.isFinite(value)) {
        throw new RangeError(`[materialTextureArray] raw HDR sample ${sampleIndex} must be finite.`);
      }
      if (isLinearFloat && Math.abs(value) > 65504) {
        throw new RangeError(
          `[materialTextureArray] raw HDR sample ${sampleIndex} exceeds finite rgba16float range: ` +
          `${value}.`,
        );
      }
      authored.push(value);
    }
    const r = authored[0] ?? 0;
    const g = channels >= 2 ? authored[1]! : r;
    const b = channels >= 3 ? authored[2]! : channels === 1 ? r : 0;
    const a = channels >= 4 ? authored[3]! : maxValue;
    out[dst] = float32ToFloat16(encodeRgb(r));
    out[dst + 1] = float32ToFloat16(encodeRgb(g));
    out[dst + 2] = float32ToFloat16(encodeRgb(b));
    out[dst + 3] = float32ToFloat16(encodeAlpha(a));
  }
  return { data: out, bytesPerRow: width * 8, rowsPerImage: height };
}

/**
 * Build the exact rgba16float mip payloads that will be uploaded. Generating the
 * radiance chain on the CPU makes validation and publication consume identical
 * half-float texels at every reachable LOD; no driver-dependent filtered-render
 * rounding sits between the checked values and the stored texture.
 *
 * The coordinate mapping matches a fullscreen linear sample at each destination
 * texel centre. Each result is quantized to half before it becomes the source of
 * the next level, exactly like a render-to-rgba16float chain.
 */
function buildNormalizedFloatMipChain(
  base: NormalizedFloatUpload,
  width: number,
  height: number,
  mipLevelCount: number,
): readonly NormalizedFloatMipUpload[] {
  const levels: NormalizedFloatMipUpload[] = [{
    ...base,
    width,
    height,
  }];
  for (let mip = 1; mip < mipLevelCount; mip += 1) {
    const previous = levels[mip - 1]!;
    const nextWidth = Math.max(1, Math.floor(previous.width / 2));
    const nextHeight = Math.max(1, Math.floor(previous.height / 2));
    const next = new Uint16Array(nextWidth * nextHeight * 4);
    const channelAt = (x: number, y: number, channel: number): number => {
      const clampedX = Math.min(previous.width - 1, Math.max(0, x));
      const clampedY = Math.min(previous.height - 1, Math.max(0, y));
      return halfToFloat(
        previous.data[(clampedY * previous.width + clampedX) * 4 + channel] ?? 0,
      );
    };
    for (let y = 0; y < nextHeight; y += 1) {
      const sourceY = (y + 0.5) * previous.height / nextHeight - 0.5;
      const y0 = Math.floor(sourceY);
      const ty = sourceY - y0;
      for (let x = 0; x < nextWidth; x += 1) {
        const sourceX = (x + 0.5) * previous.width / nextWidth - 0.5;
        const x0 = Math.floor(sourceX);
        const tx = sourceX - x0;
        const dst = (y * nextWidth + x) * 4;
        for (let channel = 0; channel < 4; channel += 1) {
          const top = channelAt(x0, y0, channel) * (1 - tx) +
            channelAt(x0 + 1, y0, channel) * tx;
          const bottom = channelAt(x0, y0 + 1, channel) * (1 - tx) +
            channelAt(x0 + 1, y0 + 1, channel) * tx;
          next[dst + channel] = float32ToFloat16(top * (1 - ty) + bottom * ty);
        }
      }
    }
    levels.push({
      data: next,
      bytesPerRow: nextWidth * 8,
      rowsPerImage: nextHeight,
      width: nextWidth,
      height: nextHeight,
    });
  }
  return levels;
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
override decodeSignedNormal: f32 = 0.0;
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

fn gpuSourceFiniteVec4(value: vec4f) -> bool {
  return all(value == value) && all(abs(value) <= vec4f(3.402823466e+38));
}

@fragment
fn fsMain(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let sourceSize = textureDimensions(sourceTexture);
  let coord = min(vec2u(position.xy), sourceSize - vec2u(1u));
  var value = textureLoad(sourceTexture, vec2i(coord), 0);
  // Float GPU sources are admitted only into rgba16float destinations, where
  // this NaN sentinel survives the blit and the material sampler's explicit
  // validity bit rejects it. Unorm/SNORM source formats cannot encode NaN/Inf.
  if (!gpuSourceFiniteVec4(value)) {
    return vec4f(bitcast<f32>(0x7fc00000u));
  }
  if (decodeSignedNormal > 0.5) {
    value = vec4f(value.xyz * 0.5 + vec3f(0.5), value.a);
  }
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
  targetFormat: GPUTextureFormat,
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
    if (source.format.includes('float') && targetFormat !== 'rgba16float') {
      throw new Error(
        `[materialTextureArray] float GPU source ${layer} requires an rgba16float ` +
        'destination so non-finite texels remain explicitly invalid.',
      );
    }
    if ((source.textureUsage & GPUTextureUsage.TEXTURE_BINDING) === 0) {
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
  source: StagedPtWebgpuTextureSource,
  target: GPUTexture,
  targetFormat: GPUTextureFormat,
  layer: number,
  copyW: number,
  copyH: number,
  roleProfile: MaterialTextureRoleProfile | null,
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
        decodeSignedNormal: source.format.endsWith('snorm') && roleProfile === 'normal' ? 1 : 0,
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
 * Stage an external image into a linear `rgba16float` array layer. External
 * images carry no core-level transfer hint, so the consuming map role supplies
 * it: emissiveMap stages through rgba8unorm-srgb (hardware decode), whereas
 * lightMap stages through rgba8unorm (linear bytes). The final render always
 * writes linear float values.
 */
function blitExternalToFloatLayer(
  device: GPUDevice,
  external: GPUCopyExternalImageSource,
  target: GPUTexture,
  layer: number,
  copyW: number,
  copyH: number,
  liveResources: Set<object>,
  integerColorSpace: 'srgb' | 'linear',
): void {
  const stageKind = integerColorSpace === 'srgb' ? 'srgbStage' : 'linearStage';
  const scratch = device.createTexture({
    label: `vitrum.pt-webgpu.scene.materialTextures.radiance.${stageKind}`,
    size: { width: copyW, height: copyH, depthOrArrayLayers: 1 },
    format: integerColorSpace === 'srgb' ? 'rgba8unorm-srgb' : 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  const scratchIsOwned = !liveResources.has(scratch);
  if (scratchIsOwned) liveResources.add(scratch);
  try {
  if (!scratchIsOwned) {
    throw new Error(
      '[pt-webgpu] radiance staging texture aliased another live texture',
    );
  }
  device.queue.copyExternalImageToTexture(
    { source: external, flipY: false },
    { texture: scratch, origin: { x: 0, y: 0, z: 0 } },
    { width: copyW, height: copyH },
  );
  const module = device.createShaderModule({
    label: `vitrum.pt-webgpu.scene.materialTextures.radiance.${stageKind}.module`,
    code: MATERIAL_TEXTURE_MIPMAP_WGSL,
  });
  const sampler = device.createSampler({
    label: `vitrum.pt-webgpu.scene.materialTextures.radiance.${stageKind}.sampler`,
    magFilter: 'nearest',
    minFilter: 'nearest',
  });
  const pipeline = device.createRenderPipeline({
    label: `vitrum.pt-webgpu.scene.materialTextures.radiance.${stageKind}.pipeline`,
    layout: 'auto',
    vertex: { module, entryPoint: 'vsMain' },
    fragment: { module, entryPoint: 'fsMain', targets: [{ format: 'rgba16float' }] },
    primitive: { topology: 'triangle-list' },
  });
  const bindGroup = device.createBindGroup({
    label: `vitrum.pt-webgpu.scene.materialTextures.radiance.${stageKind}.bindGroup`,
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: scratch.createView() },
      { binding: 1, resource: sampler },
    ],
  });
  const encoder = device.createCommandEncoder({
    label: `vitrum.pt-webgpu.scene.materialTextures.radiance.${stageKind}.encoder`,
  });
  const pass = encoder.beginRenderPass({
    label: `vitrum.pt-webgpu.scene.materialTextures.radiance.${stageKind}.pass`,
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
  liveResources: Set<object>,
): MaterialTextureArray {
  const texture = device.createTexture({
    label: DUMMY_LABEL,
    size: { width: 1, height: 1, depthOrArrayLayers: 1 },
    format,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  if (liveResources.has(texture)) {
    throw new Error(
      '[pt-webgpu] material texture candidate aliased an existing scene resource',
    );
  }
  liveResources.add(texture);
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
  readonly estimatedPeakBytesWithoutSourceSnapshots: number;
};

function snapshotLayerInfos(
  layerInfos: ReadonlyArray<MaterialTextureLayerInfo> | undefined,
  layerCount: number,
): readonly MaterialTextureLayerInfo[] | undefined {
  if (layerInfos == null) return undefined;
  if (typeof layerInfos !== 'object') {
    throw new TypeError('[materialTextureArray] layerInfos must be array-like.');
  }
  const snapshot = new Array<MaterialTextureLayerInfo>(layerCount);
  for (let layer = 0; layer < layerCount; layer += 1) {
    const info = layerInfos[layer];
    if (info == null) {
      snapshot[layer] = Object.freeze({ layer, uses: Object.freeze([]) });
      continue;
    }
    if (typeof info !== 'object') {
      throw new TypeError(`[materialTextureArray] layerInfos[${layer}] must be an object.`);
    }
    const declaredLayer = info.layer;
    const inputUses = info.uses;
    if (!Number.isSafeInteger(declaredLayer) || declaredLayer < 0) {
      throw new RangeError(`[materialTextureArray] layerInfos[${layer}].layer is invalid.`);
    }
    if (inputUses == null || typeof inputUses !== 'object') {
      throw new TypeError(`[materialTextureArray] layerInfos[${layer}].uses must be array-like.`);
    }
    const useCount = inputUses.length;
    if (!Number.isSafeInteger(useCount) || useCount < 0) {
      throw new RangeError(`[materialTextureArray] layerInfos[${layer}].uses.length is invalid.`);
    }
    const uses = new Array<MaterialTextureLayerUse>(useCount);
    for (let useIndex = 0; useIndex < useCount; useIndex += 1) {
      const use = inputUses[useIndex];
      if (use == null || typeof use !== 'object') {
        throw new TypeError(
          `[materialTextureArray] layerInfos[${layer}].uses[${useIndex}] must be an object.`,
        );
      }
      const materialIndex = use.materialIndex;
      const field = use.field;
      const colorSpace = use.colorSpace;
      const texCoord = use.texCoord;
      const magFilter = use.magFilter;
      const minFilter = use.minFilter;
      const mipFilter = use.mipFilter;
      if (!Number.isSafeInteger(materialIndex) || materialIndex < 0) {
        throw new RangeError(
          `[materialTextureArray] layer ${layer} use ${useIndex} materialIndex is invalid.`,
        );
      }
      if (typeof field !== 'string') {
        throw new TypeError(
          `[materialTextureArray] layer ${layer} use ${useIndex} field must be a string.`,
        );
      }
      if (colorSpace !== 'srgb' && colorSpace !== 'linear') {
        throw new RangeError(
          `[materialTextureArray] layer ${layer} use ${useIndex} colorSpace is invalid.`,
        );
      }
      if (!Number.isSafeInteger(texCoord) || texCoord < 0) {
        throw new RangeError(
          `[materialTextureArray] layer ${layer} use ${useIndex} texCoord is invalid.`,
        );
      }
      uses[useIndex] = Object.freeze({
        materialIndex,
        field,
        colorSpace,
        texCoord,
        ...(magFilter == null ? {} : { magFilter }),
        ...(minFilter == null ? {} : { minFilter }),
        ...(mipFilter == null ? {} : { mipFilter }),
      });
    }
    snapshot[layer] = Object.freeze({
      layer: declaredLayer,
      uses: Object.freeze(uses),
    });
  }
  return Object.freeze(snapshot);
}

function snapshotRadianceFactorsByMaterial(
  input: MaterialTextureRadianceFactorsByMaterial | undefined,
  label: string,
): MaterialTextureRadianceFactorsByMaterial | undefined {
  if (input == null) return undefined;
  if (typeof input !== 'object') {
    throw new TypeError(`[materialTextureArray] ${label} radiance factors must be array-like.`);
  }
  const materialCount = input.length;
  if (!Number.isSafeInteger(materialCount) || materialCount < 0) {
    throw new RangeError(`[materialTextureArray] ${label} radiance factor length is invalid.`);
  }
  const snapshot = new Array<readonly RadianceRgb[] | undefined>(materialCount);
  for (let material = 0; material < materialCount; material += 1) {
    const factorsInput = input[material];
    if (factorsInput == null) continue;
    if (typeof factorsInput !== 'object') {
      throw new TypeError(
        `[materialTextureArray] ${label}[${material}] factors must be array-like.`,
      );
    }
    const factorCount = factorsInput.length;
    if (!Number.isSafeInteger(factorCount) || factorCount < 0) {
      throw new RangeError(
        `[materialTextureArray] ${label}[${material}] factor length is invalid.`,
      );
    }
    const factors = new Array<RadianceRgb>(factorCount);
    for (let factorIndex = 0; factorIndex < factorCount; factorIndex += 1) {
      const factor = factorsInput[factorIndex];
      if (factor == null || typeof factor !== 'object') {
        throw new TypeError(
          `[materialTextureArray] ${label}[${material}][${factorIndex}] must be RGB.`,
        );
      }
      const r = factor[0];
      const g = factor[1];
      const b = factor[2];
      factors[factorIndex] = Object.freeze([r, g, b]) as RadianceRgb;
    }
    snapshot[material] = Object.freeze(factors);
  }
  return Object.freeze(snapshot);
}

function snapshotRadianceEnvelope(
  input: MaterialTextureRadianceEnvelope | undefined,
): MaterialTextureRadianceEnvelope | undefined {
  if (input == null) return undefined;
  if (typeof input !== 'object') {
    throw new TypeError('[materialTextureArray] radianceEnvelope must be an object.');
  }
  const emissiveMapInput = input.emissiveMap;
  const lightMapInput = input.lightMap;
  return Object.freeze({
    ...(emissiveMapInput == null
      ? {}
      : { emissiveMap: snapshotRadianceFactorsByMaterial(emissiveMapInput, 'emissiveMap')! }),
    ...(lightMapInput == null
      ? {}
      : { lightMap: snapshotRadianceFactorsByMaterial(lightMapInput, 'lightMap')! }),
  });
}

function layerUses(
  layerInfos: ReadonlyArray<MaterialTextureLayerInfo> | undefined,
  layer: number,
): readonly MaterialTextureLayerUse[] {
  return layerInfos?.[layer]?.uses ?? [];
}

function floatLayerIntegerColorSpace(
  layerInfos: ReadonlyArray<MaterialTextureLayerInfo> | undefined,
  layer: number,
): 'srgb' | 'linear' {
  const spaces = new Set(
    layerUses(layerInfos, layer)
      .filter((use) => use.field === 'emissiveMap' || use.field === 'lightMap')
      .map((use) => use.colorSpace),
  );
  if (spaces.size > 1) {
    throw new TypeError(
      `[materialTextureArray] outgoing-radiance layer ${layer} mixes sRGB and ` +
      'linear integer interpretations; collect them as separate layers.',
    );
  }
  return spaces.values().next().value ?? 'srgb';
}

function preflightMaterialTextureArray(
  device: GPUDevice,
  payloads: readonly (ImagePayload | null)[],
  format: GPUTextureFormat,
  forbiddenResources: ReadonlySet<object>,
  maxTextureArrayLayers: number,
  maxTextureDimension2D: number,
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
  if (payloads.length === 0) {
    return {
      payloads: [],
      width: 1,
      height: 1,
      mipLevelCount: 1,
      isFloatArray,
      estimatedPeakBytes: isFloatArray ? 8 : 4,
      estimatedPeakBytesWithoutSourceSnapshots: isFloatArray ? 8 : 4,
    };
  }
  if (payloads.length > maxTextureArrayLayers) {
    throw new RangeError(
      `[materialTextureArray] ${payloads.length} layers exceed device maxTextureArrayLayers ` +
      `${maxTextureArrayLayers}.`,
    );
  }
  validateGpuMaterialSources(device, payloads, forbiddenResources, format);
  let width = 1;
  let height = 1;
  for (let layer = 0; layer < payloads.length; layer += 1) {
    const payload = payloads[layer];
    if (payload == null) {
      throw new TypeError(
        `[materialTextureArray] authored source ${layer} has no usable image; ` +
        'upload is rejected so material-role fallbacks cannot be bypassed by a finite black layer.',
      );
    }
    if (
      payload.width > maxTextureDimension2D ||
      payload.height > maxTextureDimension2D
    ) {
      throw new RangeError(
        `[materialTextureArray] source ${layer} dimensions ${payload.width}x${payload.height} exceed ` +
        `device maxTextureDimension2D ${maxTextureDimension2D}; truncation is not permitted.`,
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
      mipWidth * mipHeight * payloads.length * (isFloatArray ? 8 : 4);
  }
  let retainedSourceSnapshotBytes = 0;
  let retainedNormalizedCpuBytes = 0;
  let largestStagingBytes = 0;
  let largestNativeMipStorageBytes = 0;
  for (const payload of payloads) {
    if (payload == null) continue;
    const pixelCount = payload.width * payload.height;
    if (!Number.isSafeInteger(pixelCount)) {
      throw new RangeError('[materialTextureArray] source pixel count exceeds safe integer range.');
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
    if (isFloatArray && (payload.data != null || mirror != null)) {
      // Every deterministic CPU radiance chain is built before destination
      // allocation and retained until all layers are uploaded. Its level zero
      // shares the normalized base array, so count the complete chain once.
      retainedNormalizedCpuBytes += nativeMipBytes;
    } else if (payload.data != null) {
      // rgba8 normalization also happens eagerly for every raw layer.
      retainedNormalizedCpuBytes += pixelCount * 4;
    }
  }
  const uniquePayloads = new Set<ImagePayload>();
  for (const payload of payloads) {
    if (payload != null) uniquePayloads.add(payload);
  }
  for (const payload of uniquePayloads) {
    retainedSourceSnapshotBytes += payload.stagedByteLength;
  }
  const estimatedPeakBytesWithoutSourceSnapshots =
    mipStorageBytes +
    retainedNormalizedCpuBytes +
    largestStagingBytes +
    largestNativeMipStorageBytes;
  const estimatedPeakBytes =
    estimatedPeakBytesWithoutSourceSnapshots + retainedSourceSnapshotBytes;
  if (
    !Number.isSafeInteger(estimatedPeakBytes) ||
    estimatedPeakBytes > MATERIAL_TEXTURE_ARRAY_PEAK_BUDGET_BYTES
  ) {
    throw new RangeError(
      `[materialTextureArray] estimated array upload peak ${estimatedPeakBytes} bytes exceeds ` +
      `${MATERIAL_TEXTURE_ARRAY_PEAK_BUDGET_BYTES}-byte budget ` +
      '(destination mips + CPU mirrors + all retained decoded CPU payloads/mip chains + ' +
      'staging peak + one native GPU mip chain).',
    );
  }
  return {
    payloads,
    width,
    height,
    mipLevelCount,
    isFloatArray,
    estimatedPeakBytes,
    estimatedPeakBytesWithoutSourceSnapshots,
  };
}

type PreparedMaterialTextureUpload =
  | NormalizedRgba8Upload
  | NormalizedFloatUpload
  | null;

interface StagedMaterialTextureArrayUploadData {
  readonly device: GPUDevice;
  /** Plan-owned live identity registry shared by every array consume. Seeded
   * with pre-existing engine resources and all host-owned GPU source textures;
   * every newly acquired texture identity is retained through the transaction. */
  readonly liveResources: Set<object>;
  readonly format: GPUTextureFormat;
  readonly layerInfos: readonly MaterialTextureLayerInfo[] | undefined;
  readonly radianceEnvelope: MaterialTextureRadianceEnvelope | undefined;
  readonly preflight: MaterialTextureArrayPreflight;
  readonly normalizedUploads: readonly PreparedMaterialTextureUpload[];
  readonly normalizedFloatMipUploads: readonly (
    readonly NormalizedFloatMipUpload[] | null
  )[];
}

const STAGED_MATERIAL_TEXTURE_ARRAY_UPLOADS =
  new WeakMap<StagedMaterialTextureArrayUpload, StagedMaterialTextureArrayUploadData>();

function prepareNormalizedUploads(
  preflight: MaterialTextureArrayPreflight,
  layerInfos: readonly MaterialTextureLayerInfo[] | undefined,
): {
  readonly normalizedUploads: readonly PreparedMaterialTextureUpload[];
  readonly normalizedFloatMipUploads: readonly (
    readonly NormalizedFloatMipUpload[] | null
  )[];
} {
  const { payloads, isFloatArray } = preflight;
  const normalizedUploads = payloads.map((payload, layer): PreparedMaterialTextureUpload => {
    if (payload == null) {
      throw new TypeError('[materialTextureArray] staged upload contains an unreadable source.');
    }
    if (payload.data == null) {
      if (isFloatArray && payload.gpuSource?.cpuMirror != null) {
        return normalizeGpuMirrorFloatUpload(payload.gpuSource);
      }
      return null;
    }
    const upload = isFloatArray
      ? normalizeRawTextureUploadFloat(
          payload.data,
          payload.width,
          payload.height,
          floatLayerIntegerColorSpace(layerInfos, layer),
          layerInfos,
          layer,
        )
      : normalizeRawTextureUpload(
          payload.data,
          payload.width,
          payload.height,
          layerInfos,
          layer,
        );
    if (upload == null) {
      throw new TypeError(
        `[materialTextureArray] source ${layer} has unsupported raw layout ` +
        `(${payload.data.constructor.name}, ${payload.data.byteLength} bytes for ` +
        `${payload.width}x${payload.height}); accepted payloads are tightly packed ` +
        'Uint8/Uint8Clamped, normalized Uint16, or Float32 with exactly 1, 2, 3, ' +
        'or 4 channels per pixel.',
      );
    }
    return upload;
  });
  const normalizedFloatMipUploads = normalizedUploads.map((upload, layer) => {
    const payload = payloads[layer];
    if (!isFloatArray || upload == null || payload == null) return null;
    return buildNormalizedFloatMipChain(
      upload as NormalizedFloatUpload,
      payload.width,
      payload.height,
      materialTextureMipLevelCount(payload.width, payload.height),
    );
  });
  return {
    normalizedUploads: Object.freeze(normalizedUploads),
    normalizedFloatMipUploads: Object.freeze(normalizedFloatMipUploads),
  };
}

/**
 * Observe every material source and role descriptor once and seal the complete
 * multi-atlas transaction before any GPU allocation. Raw texels and cpuMirror
 * data are engine-owned snapshots; external/GPU pixel resources remain
 * host-owned and must not mutate until the synchronous upload call returns.
 */
export function stageMaterialTextureUploadPlan(
  device: GPUDevice,
  requests: ReadonlyArray<MaterialTextureArrayUploadRequest>,
  forbiddenResources: ReadonlySet<object> = new Set(),
): MaterialTextureUploadPlan {
  if (device == null || typeof device !== 'object') {
    throw new TypeError('[materialTextureArray] device must be a GPUDevice.');
  }
  if (requests == null || typeof requests !== 'object') {
    throw new TypeError('[materialTextureArray] upload requests must be array-like.');
  }
  const requestCount = requests.length;
  if (!Number.isSafeInteger(requestCount) || requestCount < 1) {
    throw new RangeError('[materialTextureArray] upload request count must be a positive safe integer.');
  }
  const limits = device.limits;
  const maxTextureArrayLayers = positiveSafeDimension(
    limits.maxTextureArrayLayers,
    'device maxTextureArrayLayers',
  );
  const maxTextureDimension2D = positiveSafeDimension(
    limits.maxTextureDimension2D,
    'device maxTextureDimension2D',
  );
  const forbiddenSnapshot = new Set<object>();
  for (const resource of forbiddenResources) forbiddenSnapshot.add(resource);
  const liveResources = new Set<object>(forbiddenSnapshot);
  const sourceContext: MaterialTextureSourceStageContext = {
    sourcePayloads: new Map(),
    imagePayloads: new Map(),
    ownedSnapshotBytes: 0,
  };
  const pending: Array<{
    readonly format: GPUTextureFormat;
    readonly layerInfos: readonly MaterialTextureLayerInfo[] | undefined;
    readonly radianceEnvelope: MaterialTextureRadianceEnvelope | undefined;
    readonly preflight: MaterialTextureArrayPreflight;
  }> = [];
  let aggregateWithoutSourceSnapshots = 0;
  for (let requestIndex = 0; requestIndex < requestCount; requestIndex += 1) {
    const request = requests[requestIndex];
    if (request == null || typeof request !== 'object') {
      throw new TypeError(`[materialTextureArray] upload request ${requestIndex} must be an object.`);
    }
    const sources = request.sources;
    const format = request.format;
    const layerInfosInput = request.layerInfos;
    const radianceEnvelopeInput = request.radianceEnvelope;
    const sourceSnapshot = snapshotMaterialTextureSources(sources, maxTextureArrayLayers);
    const layerInfos = snapshotLayerInfos(layerInfosInput, sourceSnapshot.length);
    const radianceEnvelope = snapshotRadianceEnvelope(radianceEnvelopeInput);
    const payloads = new Array<ImagePayload | null>(sourceSnapshot.length);
    for (let layer = 0; layer < sourceSnapshot.length; layer += 1) {
      payloads[layer] = payloadOf(sourceSnapshot[layer], sourceContext);
      const gpuTexture = payloads[layer]?.gpuSource?.texture;
      if (gpuTexture != null) liveResources.add(gpuTexture);
    }
    const sealedPayloads = Object.freeze(payloads);
    validateLayerPayloadPolicies(sealedPayloads, layerInfos);
    const preflight = preflightMaterialTextureArray(
      device,
      sealedPayloads,
      format,
      forbiddenSnapshot,
      maxTextureArrayLayers,
      maxTextureDimension2D,
    );
    aggregateWithoutSourceSnapshots += preflight.estimatedPeakBytesWithoutSourceSnapshots;
    if (!Number.isSafeInteger(aggregateWithoutSourceSnapshots)) {
      throw new RangeError('[materialTextureArray] aggregate upload estimate exceeds safe integer range.');
    }
    pending.push({ format, layerInfos, radianceEnvelope, preflight });
  }
  const estimatedPeakBytes =
    aggregateWithoutSourceSnapshots + sourceContext.ownedSnapshotBytes;
  if (
    !Number.isSafeInteger(estimatedPeakBytes) ||
    estimatedPeakBytes > MATERIAL_TEXTURE_ARRAY_PEAK_BUDGET_BYTES
  ) {
    throw new RangeError(
      `[materialTextureArray] aggregate material-atlas peak ${estimatedPeakBytes} bytes exceeds ` +
      `${MATERIAL_TEXTURE_ARRAY_PEAK_BUDGET_BYTES}-byte budget before GPU allocation.`,
    );
  }
  const arrays = pending.map((entry): StagedMaterialTextureArrayUpload => {
    const prepared = prepareNormalizedUploads(entry.preflight, entry.layerInfos);
    validateMaterialTextureRadianceEnvelope(
      entry.preflight.payloads,
      prepared.normalizedUploads,
      prepared.normalizedFloatMipUploads,
      entry.format,
      entry.layerInfos,
      entry.radianceEnvelope,
    );
    const token = Object.freeze({
      estimatedPeakBytes: entry.preflight.estimatedPeakBytes,
      layerCount: entry.preflight.payloads.length,
      format: entry.format,
    });
    STAGED_MATERIAL_TEXTURE_ARRAY_UPLOADS.set(token, {
      device,
      liveResources,
      format: entry.format,
      layerInfos: entry.layerInfos,
      radianceEnvelope: entry.radianceEnvelope,
      preflight: entry.preflight,
      normalizedUploads: prepared.normalizedUploads,
      normalizedFloatMipUploads: prepared.normalizedFloatMipUploads,
    });
    return token;
  });
  return Object.freeze({ arrays: Object.freeze(arrays), estimatedPeakBytes });
}

/** Read-only validation/estimation used to reject a three-atlas upload before its first allocation. */
export function estimateMaterialTextureArrayPeakBytes(
  device: GPUDevice,
  sources: ReadonlyArray<unknown>,
  format: GPUTextureFormat,
  forbiddenResources: ReadonlySet<object> = new Set(),
): number {
  return stageMaterialTextureUploadPlan(
    device,
    [{ sources, format }],
    forbiddenResources,
  ).arrays[0]!.estimatedPeakBytes;
}

/**
 * Build the material texture 2D-array. `sources` is the dedup'd, upload-ordered
 * handle list from {@link collectMaterialTextures}; layer `i` holds `sources[i]`,
 * matching the `baseColorIdx` the descriptor buffer stores.
 */
export function createMaterialTextureArrayFromStaged(
  device: GPUDevice,
  staged: StagedMaterialTextureArrayUpload,
): MaterialTextureArray {
  const stagedData = STAGED_MATERIAL_TEXTURE_ARRAY_UPLOADS.get(staged);
  if (stagedData == null) {
    throw new TypeError('[materialTextureArray] staged upload token is invalid or foreign.');
  }
  if (stagedData.device !== device) {
    throw new Error('[materialTextureArray] staged upload belongs to a different GPUDevice.');
  }
  // A staged array is an ownership transaction, not a reusable recipe. Consume
  // it before allocation so a failed or successful upload cannot be replayed.
  STAGED_MATERIAL_TEXTURE_ARRAY_UPLOADS.delete(staged);
  const {
    liveResources,
    format,
    layerInfos,
    preflight,
    normalizedUploads,
    normalizedFloatMipUploads,
  } = stagedData;
  if (preflight.payloads.length === 0) {
    return createDummyArray(device, format, liveResources);
  }

  const warnings: string[] = [];
  const structuredWarnings: MaterialTextureArrayWarning[] = [];
  const { payloads, width, height, mipLevelCount, isFloatArray } = preflight;
  const texture = device.createTexture({
    label: ARRAY_LABEL,
    size: { width, height, depthOrArrayLayers: payloads.length },
    mipLevelCount,
    format,
    // Native source mip chains are copied into this max-size array.
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST,
  });
  if (liveResources.has(texture)) {
    throw new Error(
      '[pt-webgpu] material texture candidate aliased an existing scene resource',
    );
  }
  liveResources.add(texture);

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
    if (liveResources.has(nativeTexture)) {
      throw new Error(
        '[pt-webgpu] native material texture staging aliased a live scene resource',
      );
    }
    liveResources.add(nativeTexture);
    try {
      const normalizedUpload = normalizedUploads[layer];
      const normalizedMipChain = normalizedFloatMipUploads[layer];
      if (normalizedMipChain != null) {
        normalizedMipChain.forEach((level, mip) => {
          device.queue.writeTexture(
            { texture: nativeTexture, mipLevel: mip, origin: { x: 0, y: 0, z: 0 } },
            level.data,
            {
              bytesPerRow: level.bytesPerRow,
              rowsPerImage: level.rowsPerImage,
            },
            { width: level.width, height: level.height },
          );
        });
      } else if (normalizedUpload != null && p.data != null) {
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
          device,
          p.gpuSource,
          nativeTexture,
          format,
          0,
          copyW,
          copyH,
          layerRoleProfile(layerInfos, layer),
        );
      } else if (p.external != null) {
        if (isFloatArray) {
          blitExternalToFloatLayer(
            device,
            p.external,
            nativeTexture,
            0,
            copyW,
            copyH,
            liveResources,
            floatLayerIntegerColorSpace(layerInfos, layer),
          );
        } else {
          device.queue.copyExternalImageToTexture(
            { source: p.external, flipY: false },
            { texture: nativeTexture, origin: { x: 0, y: 0, z: 0 } },
            { width: copyW, height: copyH },
          );
        }
      }
      if (normalizedMipChain == null) {
        generateTextureMips(device, nativeTexture, format, nativeMipLevelCount);
      }
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
    layerCount: payloads.length,
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

/**
 * Standalone convenience wrapper. Scene upload should stage all three role
 * arrays together with {@link stageMaterialTextureUploadPlan} and call
 * {@link createMaterialTextureArrayFromStaged} so aliases are observed once.
 */
export function createMaterialTextureArray(
  device: GPUDevice,
  sources: ReadonlyArray<unknown>,
  format: GPUTextureFormat = 'rgba8unorm-srgb',
  layerInfos?: ReadonlyArray<MaterialTextureLayerInfo>,
  forbiddenResources: ReadonlySet<object> = new Set(),
  radianceEnvelope?: MaterialTextureRadianceEnvelope,
): MaterialTextureArray {
  const plan = stageMaterialTextureUploadPlan(
    device,
    [{
      sources,
      format,
      ...(layerInfos == null ? {} : { layerInfos }),
      ...(radianceEnvelope == null ? {} : { radianceEnvelope }),
    }],
    forbiddenResources,
  );
  return createMaterialTextureArrayFromStaged(device, plan.arrays[0]!);
}
