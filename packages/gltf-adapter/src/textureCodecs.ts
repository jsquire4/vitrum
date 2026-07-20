// textureCodecs.ts — raw-image codec detection + decoder shims (D15-6).
// Extracted verbatim from texturePipeline.ts. Owns: the platform/browser bitmap
// decoder, the Node pngjs/jpeg-js/webp-wasm dynamic-import decoders, the raw-image
// format sniffers + host-capability probes, and the low-level buffer/canvas
// readback helpers. texturePipeline.ts imports these back; PlatformTextureDecodeError
// lives here so both modules can throw/catch the same class without a runtime cycle
// (the DecodeSceneTextureDiagnosticCode + DecodeGltfTexturePixelsFn imports below are
// type-only, so there is no runtime import cycle).

import type { RawImageHandle } from './textures.js';
import type { DecodeGltfTexturePixelsFn, DecodeSceneTextureDiagnosticCode } from './texturePipeline.js';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

export class PlatformTextureDecodeError extends Error {
  readonly code: Extract<
    DecodeSceneTextureDiagnosticCode,
    'platform-image-decode-failed' | 'platform-image-readback-unavailable' | 'platform-image-readback-failed'
  >;

  constructor(
    code: PlatformTextureDecodeError['code'],
    message: string,
  ) {
    super(message);
    this.name = 'PlatformTextureDecodeError';
    this.code = code;
  }
}

export const decodeRawImagePixelsWithPlatform: DecodeGltfTexturePixelsFn = async (handle, context) => {
  const bitmap = await createBitmapFromRawImage(handle, context.path);
  try {
    const width = Math.max(0, Math.floor(numberProp(bitmap, 'width') ?? 0));
    const height = Math.max(0, Math.floor(numberProp(bitmap, 'height') ?? 0));
    if (width <= 0 || height <= 0) {
      throw new PlatformTextureDecodeError(
        'platform-image-decode-failed',
        `[vitrum/gltf-adapter] ${context.path} decoded to invalid image dimensions ${width}x${height}. Texture left unchanged.`,
      );
    }
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
  return typeof globalThis.createImageBitmap === 'function' && typeof globalThis.Blob === 'function';
}

export function canDecodeRawPngPixelsWithNode(handle: RawImageHandle): boolean {
  return isNodeLikeHost() && isPngRawImageHandle(handle);
}

export function canDecodeRawJpegPixelsWithNode(handle: RawImageHandle): boolean {
  return isNodeLikeHost() && isJpegRawImageHandle(handle);
}

export function canDecodeRawWebpPixelsWithNode(handle: RawImageHandle): boolean {
  return isNodeLikeHost() && isWebpRawImageHandle(handle);
}

function isNodeLikeHost(): boolean {
  const host = globalThis as typeof globalThis & {
    process?: { versions?: { node?: unknown } };
  };
  return typeof host.process?.versions?.node === 'string';
}

function isPngRawImageHandle(handle: RawImageHandle): boolean {
  const data = handle.data;
  return data.length >= 8 &&
    data[0] === 0x89 &&
    data[1] === 0x50 &&
    data[2] === 0x4e &&
    data[3] === 0x47 &&
    data[4] === 0x0d &&
    data[5] === 0x0a &&
    data[6] === 0x1a &&
    data[7] === 0x0a;
}

function isJpegRawImageHandle(handle: RawImageHandle): boolean {
  const data = handle.data;
  const mimeType = handle.mimeType.toLowerCase();
  return (mimeType === 'image/jpeg' || mimeType === 'image/jpg') ||
    (data.length >= 3 && data[0] === 0xff && data[1] === 0xd8 && data[2] === 0xff);
}

function isWebpRawImageHandle(handle: RawImageHandle): boolean {
  const data = handle.data;
  const mimeType = handle.mimeType.toLowerCase();
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

export const decodeRawPngPixelsWithNode: DecodeGltfTexturePixelsFn = async (handle, context) => {
  try {
    const { PNG } = await importPngJs();
    const bytes = handle.data;
    const decoded = PNG.sync.read(nodeBufferFromUint8Array(bytes));
    return {
      width: decoded.width,
      height: decoded.height,
      data: decoded.data,
      channels: 4,
      dataType: 'uint8',
      colorSpace: context.colorSpace,
    };
  } catch (err) {
    throw new PlatformTextureDecodeError(
      'platform-image-decode-failed',
      `[vitrum/gltf-adapter] ${context.path} could not be decoded as PNG through the built-in Node decoder: ` +
        `${err instanceof Error ? err.message : String(err)}. Texture left unchanged.`,
    );
  }
};

export const decodeRawJpegPixelsWithNode: DecodeGltfTexturePixelsFn = async (handle, context) => {
  try {
    const jpeg = await importJpegJs();
    const decode = jpegDecodeFn(jpeg);
    const decoded = decode(nodeBufferFromUint8Array(handle.data), { useTArray: true });
    return {
      width: decoded.width,
      height: decoded.height,
      data: decoded.data,
      channels: 4,
      dataType: 'uint8',
      colorSpace: context.colorSpace,
    };
  } catch (err) {
    throw new PlatformTextureDecodeError(
      'platform-image-decode-failed',
      `[vitrum/gltf-adapter] ${context.path} could not be decoded as JPEG through the built-in Node decoder: ` +
        `${err instanceof Error ? err.message : String(err)}. Texture left unchanged.`,
    );
  }
};

export const decodeRawWebpPixelsWithNode: DecodeGltfTexturePixelsFn = async (handle, context) => {
  try {
    const webp = await importWebpWasm();
    const decode = webpDecodeFn(webp);
    const decoded = await decode(arrayBufferFromUint8Array(handle.data));
    return {
      width: decoded.width,
      height: decoded.height,
      data: decoded.data,
      channels: 4,
      dataType: 'uint8',
      colorSpace: context.colorSpace,
    };
  } catch (err) {
    throw new PlatformTextureDecodeError(
      'platform-image-decode-failed',
      `[vitrum/gltf-adapter] ${context.path} could not be decoded as WebP through the built-in Node decoder: ` +
        `${err instanceof Error ? err.message : String(err)}. Texture left unchanged.`,
    );
  }
};

interface PngJsSyncReader {
  read(data: unknown): {
    readonly width: number;
    readonly height: number;
    readonly data: Uint8Array;
  };
}

interface PngJsModule {
  readonly PNG: {
    readonly sync: PngJsSyncReader;
  };
}

async function importPngJs(): Promise<PngJsModule> {
  const specifier = 'pngjs';
  return await import(/* @vite-ignore */ specifier) as PngJsModule;
}

interface JpegJsDecodedImage {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8Array | Uint8ClampedArray;
}

type JpegJsDecodeFn = (
  data: unknown,
  options?: { readonly useTArray?: boolean },
) => JpegJsDecodedImage;

interface JpegJsModule {
  readonly decode?: JpegJsDecodeFn;
  readonly default?: {
    readonly decode?: JpegJsDecodeFn;
  };
}

async function importJpegJs(): Promise<JpegJsModule> {
  const specifier = 'jpeg-js';
  return await import(/* @vite-ignore */ specifier) as JpegJsModule;
}

function jpegDecodeFn(module: JpegJsModule): JpegJsDecodeFn {
  const decode = module.decode ?? module.default?.decode;
  if (typeof decode !== 'function') {
    throw new Error('jpeg-js decode export is unavailable');
  }
  return decode;
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

function nodeBufferFromUint8Array(bytes: Uint8Array): unknown {
  const host = globalThis as typeof globalThis & {
    Buffer?: {
      from(buffer: ArrayBufferLike, byteOffset?: number, length?: number): unknown;
    };
  };
  if (host.Buffer == null) {
    throw new Error('Node Buffer is unavailable');
  }
  return host.Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

function arrayBufferFromUint8Array(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

export async function createBitmapFromRawImage(handle: RawImageHandle, path: string): Promise<unknown> {
  try {
    const bytes = handle.data;
    const slice = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const blob = new Blob([slice as ArrayBuffer], { type: handle.mimeType });
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
  return isRecord(value) &&
    typeof value.drawImage === 'function' &&
    typeof value.getImageData === 'function';
}

export function numberProp(value: unknown, key: string): number | undefined {
  return isRecord(value) && typeof value[key] === 'number' ? value[key] : undefined;
}

export function closeBitmap(bitmap: unknown): void {
  if (isRecord(bitmap) && typeof bitmap.close === 'function') bitmap.close();
}
