// textureCodecs.ts — raw-image codec detection + decoder shims (D15-6).
// Extracted verbatim from texturePipeline.ts. Owns: the platform/browser bitmap
// decoder, deterministic pngjs/jpeg-js decoders, the Node-only webp-wasm
// decoder, raw-image format sniffers + host-capability probes, and the
// low-level buffer/canvas readback helpers. texturePipeline.ts imports these
// back; PlatformTextureDecodeError
// lives here so both modules can throw/catch the same class without a runtime cycle
// (the DecodeSceneTextureDiagnosticCode + DecodeGltfTexturePixelsFn imports below are
// type-only, so there is no runtime import cycle).

import type { RawImageHandle } from './textures.js';
import type { DecodeGltfTexturePixelsFn, DecodeSceneTextureDiagnosticCode } from './texturePipeline.js';
import {
  RawImageDimensionsError,
  readRawImageDimensions,
  type RawImageDimensions,
  type RawImageFormat,
} from './rawImageDimensions.js';
import {
  inspectIntrinsicTypedArray,
  isIntrinsicArrayBuffer,
  localUint8ArrayView,
} from './intrinsicTypedArrays.js';
export {
  canDecodeRawBasisKtx2Pixels,
  decodeRawBasisKtx2Pixels,
} from './basisKtx2Codec.js';
export {
  canDecodeRawDdsPixels,
  decodeRawDdsPixels,
} from './ddsCodec.js';

const arrayBufferSlice: unknown = Reflect.get(ArrayBuffer.prototype, 'slice');

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

export class PlatformTextureDecodeError extends Error {
  readonly code: Extract<
    DecodeSceneTextureDiagnosticCode,
    | 'platform-image-decode-failed'
    | 'platform-image-readback-unavailable'
    | 'platform-image-readback-failed'
    | 'decoded-texture-exceeds-pixel-budget'
  >;
  readonly width?: number;
  readonly height?: number;
  readonly maxDecodedTexturePixels?: number;

  constructor(
    code: PlatformTextureDecodeError['code'],
    message: string,
    details?: {
      readonly width?: number;
      readonly height?: number;
      readonly maxDecodedTexturePixels?: number;
    },
  ) {
    super(message);
    this.name = 'PlatformTextureDecodeError';
    this.code = code;
    if (details?.width !== undefined) this.width = details.width;
    if (details?.height !== undefined) this.height = details.height;
    if (details?.maxDecodedTexturePixels !== undefined) {
      this.maxDecodedTexturePixels = details.maxDecodedTexturePixels;
    }
  }
}

export const decodeRawImagePixelsWithPlatform: DecodeGltfTexturePixelsFn = async (handle, context) => {
  const normalized = normalizeRawImageHandleForDecode(handle, context.path);
  preflightRawImagePixelBudget(normalized, context.path, context.maxDecodedTexturePixels);
  const bitmap = await createBitmapFromRawImage(normalized, context.path);
  try {
    const width = Math.max(0, Math.floor(numberProp(bitmap, 'width') ?? 0));
    const height = Math.max(0, Math.floor(numberProp(bitmap, 'height') ?? 0));
    if (width <= 0 || height <= 0) {
      throw new PlatformTextureDecodeError(
        'platform-image-decode-failed',
        `[vitrum/gltf-adapter] ${context.path} decoded to invalid image dimensions ${width}x${height}. Texture left unchanged.`,
      );
    }
    assertDecodedPixelBudget(width, height, context.path, context.maxDecodedTexturePixels);
    const ctx = createReadback2dContext(width, height, context.path);
    try {
      ctx.drawImage(bitmap, 0, 0, width, height);
      const imageData = ctx.getImageData(0, 0, width, height);
      return {
        width,
        height,
        data: imageData.data,
        channels: 4,
        dataType: 'uint8',
        colorSpace: context.colorSpace,
      };
    } catch (err) {
      throw new PlatformTextureDecodeError(
        'platform-image-readback-failed',
        `[vitrum/gltf-adapter] ${context.path} decoded through browser image APIs, but canvas pixel readback failed: ` +
          `${err instanceof Error ? err.message : String(err)}. Texture left unchanged.`,
      );
    }
  } finally {
    closeBitmap(bitmap);
  }
};

export function canDecodeRawImagePixelsWithPlatform(): boolean {
  try {
    return typeof globalThis.createImageBitmap === 'function' && typeof globalThis.Blob === 'function';
  } catch {
    return false;
  }
}

export function canDecodeRawPngPixelsDeterministically(handle: RawImageHandle): boolean {
  try {
    const raw = rawImageBytesAndMime(handle);
    return raw?.mimeType === 'image/png' || readRawImageDimensions(handle)?.format === 'png';
  } catch {
    return rawImageBytesAndMime(handle)?.mimeType === 'image/png';
  }
}

export function canDecodeRawJpegPixelsDeterministically(handle: RawImageHandle): boolean {
  try {
    const raw = rawImageBytesAndMime(handle);
    return raw?.mimeType === 'image/jpeg' ||
      raw?.mimeType === 'image/jpg' ||
      readRawImageDimensions(handle)?.format === 'jpeg';
  } catch {
    const mimeType = rawImageBytesAndMime(handle)?.mimeType;
    return mimeType === 'image/jpeg' || mimeType === 'image/jpg';
  }
}

export function canDecodeRawWebpPixelsWithNode(handle: RawImageHandle): boolean {
  try {
    return isNodeLikeHost() && isWebpRawImageHandle(handle);
  } catch {
    return false;
  }
}

function isNodeLikeHost(): boolean {
  try {
    const host = globalThis as typeof globalThis & {
      process?: { versions?: { node?: unknown } };
    };
    return typeof host.process?.versions?.node === 'string';
  } catch {
    return false;
  }
}

function rawImageBytesAndMime(
  handle: RawImageHandle,
): { readonly data: Uint8Array; readonly mimeType: string } | null {
  try {
    const candidate = handle as unknown as {
      readonly data?: unknown;
      readonly mimeType?: unknown;
    };
    const data = localUint8ArrayView(candidate.data);
    if (data === null || typeof candidate.mimeType !== 'string') return null;
    return { data, mimeType: candidate.mimeType.trim().toLowerCase() };
  } catch {
    return null;
  }
}

const normalizedRawImageHandles = new WeakSet<object>();

/** @internal */
export function normalizeRawImageHandleForDecode(
  handle: RawImageHandle,
  path: string,
): RawImageHandle {
  if (
    typeof handle === 'object' &&
    handle !== null &&
    normalizedRawImageHandles.has(handle)
  ) {
    return handle;
  }
  let dataValue: unknown;
  let mimeTypeValue: unknown;
  try {
    const candidate = handle as unknown as {
      readonly data?: unknown;
      readonly mimeType?: unknown;
    };
    dataValue = candidate.data;
    mimeTypeValue = candidate.mimeType;
  } catch (err) {
    throw new PlatformTextureDecodeError(
      'platform-image-decode-failed',
      `[vitrum/gltf-adapter] ${path} raw-image property access failed: ` +
        `${safeErrorMessage(err)}. Texture left unchanged.`,
    );
  }
  return normalizeCapturedRawImageForDecode(dataValue, mimeTypeValue, path);
}

/**
 * Normalize already-captured raw-image properties. Callers that perform
 * resource accounting can charge the exact captured view before this copy,
 * without re-reading hostile getters.
 *
 * @internal
 */
export function normalizeCapturedRawImageForDecode(
  dataValue: unknown,
  mimeTypeValue: unknown,
  path: string,
): RawImageHandle {
  const info = inspectIntrinsicTypedArray(dataValue);
  const view = localUint8ArrayView(dataValue);
  if (view === null) {
    const reason = info?.isShared === true
      ? 'SharedArrayBuffer-backed image bytes are not accepted'
      : 'image data must be an intrinsic Uint8Array';
    throw new PlatformTextureDecodeError(
      'platform-image-decode-failed',
      `[vitrum/gltf-adapter] ${path} ${reason}. Texture left unchanged.`,
    );
  }
  if (typeof mimeTypeValue !== 'string' || mimeTypeValue.trim().length === 0) {
    throw new PlatformTextureDecodeError(
      'platform-image-decode-failed',
      `[vitrum/gltf-adapter] ${path} raw-image mimeType must be a non-empty string. Texture left unchanged.`,
    );
  }
  try {
    const owned = new Uint8Array(view.length);
    owned.set(view);
    const normalized: RawImageHandle = Object.freeze({
      kind: 'raw-image',
      data: owned,
      mimeType: mimeTypeValue,
    });
    normalizedRawImageHandles.add(normalized);
    return normalized;
  } catch (err) {
    throw new PlatformTextureDecodeError(
      'platform-image-decode-failed',
      `[vitrum/gltf-adapter] ${path} raw-image bytes could not be copied safely: ` +
        `${safeErrorMessage(err)}. Texture left unchanged.`,
    );
  }
}

function isWebpRawImageHandle(handle: RawImageHandle): boolean {
  const raw = rawImageBytesAndMime(handle);
  if (raw === null) return false;
  const { data, mimeType } = raw;
  return mimeType === 'image/webp' ||
    (data.length >= 12 &&
      data[0] === 0x52 &&
      data[1] === 0x49 &&
      data[2] === 0x46 &&
      data[3] === 0x46 &&
      data[8] === 0x57 &&
      data[9] === 0x45 &&
      data[10] === 0x42 &&
      data[11] === 0x50);
}

export const decodeRawPngPixelsDeterministically: DecodeGltfTexturePixelsFn = async (
  handle,
  context,
) => {
  try {
    const normalized = normalizeRawImageHandleForDecode(handle, context.path);
    preflightRawImagePixelBudget(normalized, context.path, context.maxDecodedTexturePixels, 'png');
    const [{ PNG }, { Buffer }] = await Promise.all([
      importPngJs(),
      importPortableBuffer(),
    ]);
    const decoded = PNG.sync.read(Buffer.from(normalized.data));
    assertDecodedPixelBudget(
      decoded.width,
      decoded.height,
      context.path,
      context.maxDecodedTexturePixels,
    );
    return {
      width: decoded.width,
      height: decoded.height,
      data: decoded.data,
      channels: 4,
      dataType: 'uint8',
      colorSpace: context.colorSpace,
    };
  } catch (err) {
    if (err instanceof PlatformTextureDecodeError) throw err;
    throw new PlatformTextureDecodeError(
      'platform-image-decode-failed',
      `[vitrum/gltf-adapter] ${context.path} could not be decoded as PNG through the deterministic built-in decoder: ` +
        `${err instanceof Error ? err.message : String(err)}. Texture left unchanged.`,
    );
  }
};

export const decodeRawJpegPixelsDeterministically: DecodeGltfTexturePixelsFn = async (
  handle,
  context,
) => {
  try {
    const normalized = normalizeRawImageHandleForDecode(handle, context.path);
    preflightRawImagePixelBudget(normalized, context.path, context.maxDecodedTexturePixels, 'jpeg');
    const jpeg = await importJpegJs();
    const decode = jpegDecodeFn(jpeg);
    const decoded = decode(
      normalized.data,
      jpegDecodeOptionsForPixelLimit(context.maxDecodedTexturePixels),
    );
    assertDecodedPixelBudget(
      decoded.width,
      decoded.height,
      context.path,
      context.maxDecodedTexturePixels,
    );
    return {
      width: decoded.width,
      height: decoded.height,
      data: decoded.data,
      channels: 4,
      dataType: 'uint8',
      colorSpace: context.colorSpace,
    };
  } catch (err) {
    if (err instanceof PlatformTextureDecodeError) throw err;
    throw new PlatformTextureDecodeError(
      'platform-image-decode-failed',
      `[vitrum/gltf-adapter] ${context.path} could not be decoded as JPEG through the deterministic built-in decoder: ` +
        `${err instanceof Error ? err.message : String(err)}. Texture left unchanged.`,
    );
  }
};

export const decodeRawWebpPixelsWithNode: DecodeGltfTexturePixelsFn = async (handle, context) => {
  try {
    const normalized = normalizeRawImageHandleForDecode(handle, context.path);
    preflightRawImagePixelBudget(normalized, context.path, context.maxDecodedTexturePixels, 'webp');
    const webp = await importWebpWasm();
    const decode = webpDecodeFn(webp);
    const decoded = await decode(arrayBufferFromUint8Array(normalized.data));
    assertDecodedPixelBudget(
      decoded.width,
      decoded.height,
      context.path,
      context.maxDecodedTexturePixels,
    );
    return {
      width: decoded.width,
      height: decoded.height,
      data: decoded.data,
      channels: 4,
      dataType: 'uint8',
      colorSpace: context.colorSpace,
    };
  } catch (err) {
    if (err instanceof PlatformTextureDecodeError) throw err;
    throw new PlatformTextureDecodeError(
      'platform-image-decode-failed',
      `[vitrum/gltf-adapter] ${context.path} could not be decoded as WebP through the built-in Node decoder: ` +
        `${err instanceof Error ? err.message : String(err)}. Texture left unchanged.`,
    );
  }
};

function preflightRawImagePixelBudget(
  handle: RawImageHandle,
  path: string,
  maxDecodedTexturePixels: number,
  expectedFormat?: RawImageFormat,
): RawImageDimensions | null {
  let dimensions: RawImageDimensions | null;
  try {
    dimensions = readRawImageDimensions(handle);
  } catch (err) {
    if (err instanceof RawImageDimensionsError) {
      throw new PlatformTextureDecodeError(
        'platform-image-decode-failed',
        `[vitrum/gltf-adapter] ${path} could not be safely preflighted: ${err.message} Texture left unchanged.`,
      );
    }
    throw err;
  }
  if (dimensions === null) {
    if (maxDecodedTexturePixels === 0 && expectedFormat === undefined) return null;
    throw new PlatformTextureDecodeError(
      'platform-image-decode-failed',
      `[vitrum/gltf-adapter] ${path} has no supported PNG/JPEG/WebP dimension header, so its decoded ` +
        'allocation cannot be safely preflighted. Texture left unchanged.',
    );
  }
  if (expectedFormat !== undefined && dimensions.format !== expectedFormat) {
    throw new PlatformTextureDecodeError(
      'platform-image-decode-failed',
      `[vitrum/gltf-adapter] ${path} was routed to the ${expectedFormat.toUpperCase()} decoder but has a ` +
        `${dimensions.format.toUpperCase()} header. Texture left unchanged.`,
    );
  }
  assertDecodedPixelBudget(
    dimensions.width,
    dimensions.height,
    path,
    maxDecodedTexturePixels,
  );
  return dimensions;
}

function assertDecodedPixelBudget(
  width: number,
  height: number,
  path: string,
  maxDecodedTexturePixels: number,
): void {
  if (
    !Number.isSafeInteger(width) ||
    !Number.isSafeInteger(height) ||
    width <= 0 ||
    height <= 0
  ) {
    throw new PlatformTextureDecodeError(
      'platform-image-decode-failed',
      `[vitrum/gltf-adapter] ${path} decoded to invalid image dimensions ${width}x${height}. Texture left unchanged.`,
    );
  }
  if (width > Math.floor(Number.MAX_SAFE_INTEGER / height)) {
    throw new PlatformTextureDecodeError(
      'decoded-texture-exceeds-pixel-budget',
      `[vitrum/gltf-adapter] ${path} declares ${width}x${height} pixels, whose pixel count exceeds the safe ` +
        'integer range. Texture left unchanged before decoder allocation.',
      { width, height, maxDecodedTexturePixels },
    );
  }
  const pixelCount = width * height;
  if (maxDecodedTexturePixels !== 0 && pixelCount > maxDecodedTexturePixels) {
    throw new PlatformTextureDecodeError(
      'decoded-texture-exceeds-pixel-budget',
      `[vitrum/gltf-adapter] ${path} declares ${width}x${height} (${pixelCount} pixels), which exceeds ` +
        `maxDecodedTexturePixels=${maxDecodedTexturePixels}. Texture left unchanged before decoder allocation.`,
      { width, height, maxDecodedTexturePixels },
    );
  }
}

interface PngJsSyncReader {
  read(data: unknown): {
    readonly width: number;
    readonly height: number;
    readonly data: Uint8Array;
  };
}

interface PngJsModule {
  readonly PNG?: {
    readonly sync: PngJsSyncReader;
  };
  readonly default?: {
    readonly PNG?: {
      readonly sync: PngJsSyncReader;
    };
  };
}

async function importPngJs(): Promise<{ readonly PNG: NonNullable<PngJsModule['PNG']> }> {
  // The browser bundle is the same pure-JS implementation in every host.
  // A literal import keeps it visible to browser bundlers instead of relying
  // on a Node-only runtime resolver.
  const module = await import('pngjs/browser.js' satisfies string) as PngJsModule;
  const PNG = module.PNG ?? module.default?.PNG;
  if (PNG?.sync == null || typeof PNG.sync.read !== 'function') {
    throw new Error('pngjs PNG.sync.read export is unavailable');
  }
  return { PNG };
}

async function importPortableBuffer(): Promise<typeof import('buffer/index.js')> {
  // pngjs's browser build expects the Buffer read/copy surface. This package
  // implementation is pure JavaScript and bundles identically in browser and
  // Node; using the explicit package subpath avoids resolving Node's built-in.
  return await import('buffer/index.js');
}

interface JpegJsDecodedImage {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8Array | Uint8ClampedArray;
}

type JpegJsDecodeFn = (
  data: Uint8Array,
  options: {
    readonly useTArray: true;
    readonly maxResolutionInMP?: number;
    readonly maxMemoryUsageInMB?: number;
  },
) => JpegJsDecodedImage;

interface JpegJsModule {
  readonly decode?: JpegJsDecodeFn;
  readonly default?: {
    readonly decode?: JpegJsDecodeFn;
  };
}

async function importJpegJs(): Promise<JpegJsModule> {
  // jpeg-js is pure JavaScript and accepts Uint8Array directly, so the same
  // decoder and options execute in browser and Node.
  return await import('jpeg-js');
}

function jpegDecodeFn(module: JpegJsModule): JpegJsDecodeFn {
  const decode = module.decode ?? module.default?.decode;
  if (typeof decode !== 'function') {
    throw new Error('jpeg-js decode export is unavailable');
  }
  return decode;
}

const JPEG_JS_PIXELS_PER_MEGAPIXEL = 1_000_000;
const JPEG_JS_BYTES_PER_MIB = 1024 * 1024;
const JPEG_JS_MAX_SAFE_RESOLUTION_MP =
  Number.MAX_SAFE_INTEGER / JPEG_JS_PIXELS_PER_MEGAPIXEL;
const JPEG_JS_MAX_SAFE_MEMORY_USAGE_MIB = Math.floor(
  Number.MAX_SAFE_INTEGER / JPEG_JS_BYTES_PER_MIB,
);

/**
 * Keep jpeg-js's independent guards from becoming stricter than the public
 * adapter policy. Encoded JPEG dimensions are validated and checked against
 * the exact integer pixel cap before jpeg-js is imported.
 *
 * jpeg-js converts `maxResolutionInMP` back to pixels. The upward whole-MP
 * envelope prevents floating-point unit conversion from rejecting a valid
 * boundary image; the adapter preflight remains the exact guard. jpeg-js also
 * tracks opaque temporary allocations, for which the adapter exposes no
 * separate memory policy, so its memory option is set to the largest whole-MiB
 * value whose byte conversion remains a safe integer. Host allocation and
 * typed-array limits still apply normally.
 *
 * @internal
 */
export function jpegDecodeOptionsForPixelLimit(
  maxDecodedTexturePixels: number,
): {
  readonly useTArray: true;
  readonly maxResolutionInMP: number;
  readonly maxMemoryUsageInMB: number;
} {
  if (
    !Number.isSafeInteger(maxDecodedTexturePixels) ||
    maxDecodedTexturePixels < 0
  ) {
    throw new RangeError(
      '[vitrum/gltf-adapter] maxDecodedTexturePixels must be a non-negative safe integer.',
    );
  }
  const maxResolutionInMP = maxDecodedTexturePixels === 0
    ? JPEG_JS_MAX_SAFE_RESOLUTION_MP
    : Math.min(
      JPEG_JS_MAX_SAFE_RESOLUTION_MP,
      Math.ceil(
        maxDecodedTexturePixels / JPEG_JS_PIXELS_PER_MEGAPIXEL,
      ),
    );
  return {
    useTArray: true,
    maxResolutionInMP,
    maxMemoryUsageInMB: JPEG_JS_MAX_SAFE_MEMORY_USAGE_MIB,
  };
}

interface WebpWasmDecodedImage {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8Array | Uint8ClampedArray;
}

type WebpWasmDecodeFn = (data: unknown) => Promise<WebpWasmDecodedImage> | WebpWasmDecodedImage;

interface WebpWasmModule {
  readonly decode?: WebpWasmDecodeFn;
  readonly default?: {
    readonly decode?: WebpWasmDecodeFn;
  };
}

async function importWebpWasm(): Promise<WebpWasmModule> {
  const specifier = 'webp-wasm';
  return await import(/* @vite-ignore */ specifier) as WebpWasmModule;
}

function webpDecodeFn(module: WebpWasmModule): WebpWasmDecodeFn {
  const owner = module.decode !== undefined ? module : module.default;
  const decode = owner?.decode;
  if (typeof decode !== 'function') {
    throw new Error('webp-wasm decode export is unavailable');
  }
  return (data) => decode.call(owner, data);
}

function arrayBufferFromUint8Array(bytes: Uint8Array): ArrayBuffer {
  const info = intrinsicUint8ArrayInfo(bytes);
  if (typeof arrayBufferSlice !== 'function') {
    throw new Error('ArrayBuffer.prototype.slice is unavailable');
  }
  const result = Reflect.apply(arrayBufferSlice, info.buffer, [
    info.byteOffset,
    info.byteOffset + info.byteLength,
  ]) as unknown;
  if (!isIntrinsicArrayBuffer(result)) {
    throw new Error('ArrayBuffer slice did not return an ArrayBuffer');
  }
  return result;
}

function intrinsicUint8ArrayInfo(bytes: Uint8Array): {
  readonly buffer: ArrayBuffer;
  readonly byteOffset: number;
  readonly byteLength: number;
} {
  const info = inspectIntrinsicTypedArray(bytes);
  if (info === null || info.brand !== 'Uint8Array' || info.isShared) {
    throw new Error('image bytes must be a non-shared intrinsic Uint8Array');
  }
  return {
    buffer: info.buffer as ArrayBuffer,
    byteOffset: info.byteOffset,
    byteLength: info.byteLength,
  };
}

export async function createBitmapFromRawImage(handle: RawImageHandle, path: string): Promise<unknown> {
  try {
    const normalized = normalizeRawImageHandleForDecode(handle, path);
    const slice = arrayBufferFromUint8Array(normalized.data);
    const blob = new Blob([slice], { type: normalized.mimeType });
    return await createImageBitmap(blob);
  } catch (err) {
    throw new PlatformTextureDecodeError(
      'platform-image-decode-failed',
      `[vitrum/gltf-adapter] ${path} could not be decoded through browser image APIs: ` +
        `${err instanceof Error ? err.message : String(err)}. Texture left unchanged.`,
    );
  }
}

interface Canvas2dReadbackContext {
  drawImage(image: unknown, dx: number, dy: number, dw: number, dh: number): void;
  getImageData(sx: number, sy: number, sw: number, sh: number): { readonly data: Uint8ClampedArray };
}

export function createReadback2dContext(
  width: number,
  height: number,
  path: string,
): Canvas2dReadbackContext {
  const host = globalThis as typeof globalThis & {
    OffscreenCanvas?: new (width: number, height: number) => { getContext(type: '2d'): unknown };
    document?: { createElement(tag: 'canvas'): { width: number; height: number; getContext(type: '2d'): unknown } };
  };
  const canvas = typeof host.OffscreenCanvas === 'function'
    ? new host.OffscreenCanvas(width, height)
    : host.document?.createElement('canvas');
  if (canvas == null) {
    throw new PlatformTextureDecodeError(
      'platform-image-readback-unavailable',
      `[vitrum/gltf-adapter] ${path} decoded through browser image APIs, but no OffscreenCanvas/document canvas ` +
        'is available for pixel readback. Texture left unchanged.',
    );
  }
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!isCanvas2dReadbackContext(ctx)) {
    throw new PlatformTextureDecodeError(
      'platform-image-readback-unavailable',
      `[vitrum/gltf-adapter] ${path} decoded through browser image APIs, but a 2D canvas readback context ` +
        'could not be created. Texture left unchanged.',
    );
  }
  return ctx;
}

function isCanvas2dReadbackContext(value: unknown): value is Canvas2dReadbackContext {
  try {
    return isRecord(value) &&
      typeof value.drawImage === 'function' &&
      typeof value.getImageData === 'function';
  } catch {
    return false;
  }
}

export function numberProp(value: unknown, key: string): number | undefined {
  try {
    if (!isRecord(value)) return undefined;
    const property = value[key];
    return typeof property === 'number' ? property : undefined;
  } catch {
    return undefined;
  }
}

export function closeBitmap(bitmap: unknown): void {
  try {
    if (!isRecord(bitmap)) return;
    const close = bitmap.close;
    if (typeof close === 'function') {
      Reflect.apply(close, bitmap, []);
    }
  } catch {
    // Bitmap cleanup is observational and must never replace the decode result.
  }
}

function safeErrorMessage(error: unknown): string {
  try {
    if (
      typeof error === 'object' &&
      error !== null &&
      typeof (error as { readonly message?: unknown }).message === 'string'
    ) {
      return (error as { readonly message: string }).message;
    }
    return String(error);
  } catch {
    return '<unprintable error>';
  }
}
