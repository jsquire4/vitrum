// textures.ts — glTF image → TextureRef bridge.
//
// The adapter exposes images through a pluggable decode callback. Default
// behaviour differs by environment:
//   - Browser: createImageBitmap (returns an ImageBitmap usable by pt-webgpu
//     external-image upload; CPU-atlas backends need the decodeSceneTextures()
//     pixel bridge or a host-supplied CPU-readable handle).
//   - Non-browser: raw bytes are wrapped in { mimeType, data: Uint8Array }.
//     pt-webgpu's materialTextureArray.ts accepts duck-typed objects; see
//     README for the exact shapes accepted per backend.
//
// sRGB vs linear ownership note:
//   The default TextureRef bridge does NOT apply colorspace conversion.
//   decodeSceneTextures() is the opt-in CPU pixel bridge that normalizes
//   sRGB/data map roles into the requested backend color-space payload.
//   - baseColor / emissive map images carry sRGB data; the BACKEND is
//     responsible for sRGB-decode on upload (gl.SRGB8_ALPHA8, GPUTextureFormat
//     'rgba8unorm-srgb', or a manual gamma decode in the shader).
//   - normal / ORM / ao / lightMap / bumpMap images are linear data;
//     backends must NOT apply sRGB-decode to these.
//   The README's support matrix documents this ownership boundary.

import type { GltfJson } from './gltfTypes.js';
import type { GltfTexture } from './gltfTypes.js';
import type {
  TextureFilterMode,
  TextureMipFilterMode,
  TextureRef,
  TextureWrapMode,
  UvTransform,
} from '@vitrum/core';
import {
  DecodedImageHandleOwner,
  ImportResourceLedger,
  createAsyncResourceLimiter,
  gltfArrayBufferByteLength,
  gltfBufferResourceKey,
  gltfImageResourceKey,
  GltfResourceLimitError,
  type GltfImportResourceContext,
} from './importResourceBudget.js';
import { localUint8ArrayView } from './intrinsicTypedArrays.js';
import { readEncodedImageDimensions } from './rawImageDimensions.js';
import { validateDeclaredBufferRange } from './bufferRangeValidation.js';
import { isBasisKtx2Bytes } from './basisKtx2Codec.js';
import { isDdsBytes } from './ddsCodec.js';

export type GltfTextureSourceExtension =
  | 'KHR_texture_basisu'
  | 'EXT_texture_webp'
  | 'MSFT_texture_dds';

export const GLTF_TEXTURE_SOURCE_EXTENSIONS: readonly GltfTextureSourceExtension[] = [
  'KHR_texture_basisu',
  'EXT_texture_webp',
  'MSFT_texture_dds',
];

/**
 * Alternate image sources the adapter can consume without a host codec hook.
 * KHR_texture_basisu is normalized to RGBA8 by the built-in WASM transcoder.
 * MSFT_texture_dds is normalized by the built-in BC1-BC5/masked-pixel decoder.
 * WebP is added by {@link effectiveGltfTextureSourceExtensions} only when this
 * host exposes the browser bitmap bridge or the packaged Node decoder.
 */
export const BUILTIN_GLTF_TEXTURE_SOURCE_EXTENSIONS =
  Object.freeze([
    'KHR_texture_basisu',
    'MSFT_texture_dds',
  ] as const satisfies readonly GltfTextureSourceExtension[]);

/** Whether this runtime has an adapter-owned WebP decode path. */
export function hasBuiltinWebpTextureSourceDecoder(): boolean {
  try {
    const host = globalThis as typeof globalThis & {
      process?: { versions?: { node?: unknown } };
    };
    const browserBridge =
      typeof host.createImageBitmap === 'function' &&
      typeof host.Blob === 'function';
    const packagedNodeDecoder = typeof host.process?.versions?.node === 'string';
    return browserBridge || packagedNodeDecoder;
  } catch {
    return false;
  }
}

export function effectiveGltfTextureSourceExtensions(
  requested: readonly GltfTextureSourceExtension[] | undefined,
): readonly GltfTextureSourceExtension[] {
  return Object.freeze([
    ...new Set<GltfTextureSourceExtension>([
      ...BUILTIN_GLTF_TEXTURE_SOURCE_EXTENSIONS,
      ...(hasBuiltinWebpTextureSourceDecoder()
        ? ['EXT_texture_webp' as const]
        : []),
      ...(requested ?? []),
    ]),
  ]);
}

/**
 * Decode one glTF image into a texture handle.
 *
 * Ownership transfers to the import for every resolved callback result. The
 * adapter closes a closable handle on import rejection, when successful texture
 * normalization supersedes it, or when the host calls `releaseGltfResources()`
 * on the successful result. Within one import, shared identities close once.
 * A host cache that shares a closable identity across imports must retain
 * ownership itself (for example with a non-closable wrapper) or provide
 * idempotent/reference-counted `close()` semantics.
 */
export type DecodeImageFn = (
  bytes: Uint8Array,
  mimeType: string,
) => Promise<unknown>;

export interface GltfImageBytes {
  readonly bytes: Uint8Array;
  readonly mimeType: string;
}

export type GltfImageBytesMap =
  | ReadonlyMap<number, GltfImageBytes>
  | Record<number, GltfImageBytes>;

/**
 * Raw-bytes fallback returned when no decodeImage function is supplied and
 * createImageBitmap is unavailable.
 */
export interface RawImageHandle {
  readonly kind: 'raw-image';
  readonly mimeType: string;
  readonly data: Uint8Array;
  readonly width?: number;
  readonly height?: number;
}

const GLTF_TEXTURE_REF_SOURCE = Symbol('vitrum.gltf.textureRefSource');
// Capture the intrinsic copy operation once. Image byte views can retain a
// host-owned backing ArrayBuffer whose own `slice` property is untrusted.
// eslint-disable-next-line @typescript-eslint/unbound-method -- Invoked only with an explicit typed-array receiver.
const UINT8_ARRAY_SLICE = Uint8Array.prototype.slice;

export interface GltfTextureRefSource {
  readonly path: string;
  readonly imageSourcePath?: string;
  readonly textureIndex?: number;
  readonly imageIndex?: number;
  readonly samplerIndex?: number;
  readonly imageUri?: string;
  readonly imageMimeType?: string;
  readonly textureSourceExtension?: GltfTextureSourceExtension;
}

export type GltfTextureAcquisitionDiagnosticCode =
  | 'image-not-found'
  | 'image-buffer-view-not-found'
  | 'image-buffer-unavailable'
  | 'image-source-missing'
  | 'external-image-uri'
  | 'malformed-data-uri'
  | 'data-uri-atob-unavailable'
  | 'data-uri-decode-failed'
  | 'image-decoder-missing'
  | 'disabled-texture-source-extension';

export interface GltfTextureAcquisitionDiagnostic {
  readonly severity: 'warning';
  readonly code: GltfTextureAcquisitionDiagnosticCode;
  readonly path: string;
  readonly message: string;
  readonly textureIndex?: number;
  readonly imageIndex?: number;
  readonly bufferViewIndex?: number;
  readonly bufferIndex?: number;
  readonly imageUri?: string;
  readonly textureSourceExtensions?: readonly GltfTextureSourceExtension[];
}

export type GltfTextureAcquisitionDiagnosticSink = (
  diagnostic: GltfTextureAcquisitionDiagnostic,
) => void;

type GltfTextureRefWithSource = TextureRef & {
  readonly [GLTF_TEXTURE_REF_SOURCE]?: GltfTextureRefSource;
};

interface SelectedTextureImageSource {
  readonly imageIndex?: number;
  readonly path?: string;
  readonly textureSourceExtension?: GltfTextureSourceExtension;
}

export function attachGltfTextureRefSource(ref: TextureRef, source: GltfTextureRefSource | undefined): TextureRef {
  if (source === undefined) return ref;
  const tagged = { ...ref } as GltfTextureRefWithSource;
  Object.defineProperty(tagged, GLTF_TEXTURE_REF_SOURCE, {
    value: source,
    enumerable: false,
    configurable: false,
    writable: false,
  });
  return tagged;
}

export function gltfTextureRefSource(ref: TextureRef): GltfTextureRefSource | undefined {
  return (ref as GltfTextureRefWithSource)[GLTF_TEXTURE_REF_SOURCE];
}

/**
 * Resolve a glTF image index to bytes + mimeType from a bufferView.
 * Data URI images are decoded locally. External URI images are skipped (the
 * adapter does not fetch).
 */
function getImageBytes(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  imageIndex: number,
  textureIndex: number,
  warnings: string[],
  onDiagnostic?: GltfTextureAcquisitionDiagnosticSink,
  externalImages?: ReadonlyMap<number, GltfImageBytes>,
  selectedSource?: SelectedTextureImageSource,
  resourceContext?: GltfImportResourceContext,
): { bytes: Uint8Array; mimeType: string } | undefined {
  const image = gltf.images?.[imageIndex];
  if (!image) {
    emitTextureAcquisitionDiagnostic(warnings, onDiagnostic, {
      severity: 'warning',
      code: 'image-not-found',
      path: selectedSource?.path ?? `textures[${textureIndex}].source`,
      textureIndex,
      imageIndex,
      ...(selectedSource?.textureSourceExtension !== undefined
        ? { textureSourceExtensions: [selectedSource.textureSourceExtension] }
        : {}),
      message:
        `[vitrum/gltf-adapter] Texture at textures[${textureIndex}] references missing image index ` +
        `${imageIndex}. Texture skipped.`,
    });
    return undefined;
  }

  if (image.bufferView !== undefined) {
    const bv = gltf.bufferViews?.[image.bufferView];
    if (!bv) {
      emitTextureAcquisitionDiagnostic(warnings, onDiagnostic, {
        severity: 'warning',
        code: 'image-buffer-view-not-found',
        path: `images[${imageIndex}].bufferView`,
        textureIndex,
        imageIndex,
        bufferViewIndex: image.bufferView,
        message:
          `[vitrum/gltf-adapter] Image "${image.name ?? imageIndex}" references missing bufferView ` +
          `${image.bufferView}. Image skipped.`,
      });
      return undefined;
    }
    // Preserve the adapter's structured "resource unavailable" path when the
    // host did not supply the referenced buffer at all. Malformed indices still
    // fall through to the strict declared-range validator below.
    if (
      Number.isSafeInteger(bv.buffer) &&
      bv.buffer >= 0 &&
      !buffers.has(bv.buffer)
    ) {
      emitTextureAcquisitionDiagnostic(warnings, onDiagnostic, {
        severity: 'warning',
        code: 'image-buffer-unavailable',
        path: `bufferViews[${image.bufferView}].buffer`,
        textureIndex,
        imageIndex,
        bufferViewIndex: image.bufferView,
        bufferIndex: bv.buffer,
        message:
          `[vitrum/gltf-adapter] Image "${image.name ?? imageIndex}" bufferView ${image.bufferView} ` +
          `references unavailable buffer ${bv.buffer}. Image skipped.`,
      });
      return undefined;
    }
    const range = validateDeclaredBufferRange(
      gltf,
      bv.buffer,
      bv.byteOffset ?? 0,
      bv.byteLength,
      `bufferViews[${image.bufferView}]`,
    );
    const buf = buffers.get(range.bufferIndex);
    if (!buf) {
      emitTextureAcquisitionDiagnostic(warnings, onDiagnostic, {
        severity: 'warning',
        code: 'image-buffer-unavailable',
        path: `bufferViews[${image.bufferView}].buffer`,
        textureIndex,
        imageIndex,
        bufferViewIndex: image.bufferView,
        bufferIndex: range.bufferIndex,
        message:
          `[vitrum/gltf-adapter] Image "${image.name ?? imageIndex}" bufferView ${image.bufferView} ` +
          `references unavailable buffer ${range.bufferIndex}. Image skipped.`,
      });
      return undefined;
    }
    const loadedByteLength = gltfArrayBufferByteLength(buf);
    if (loadedByteLength === undefined) {
      throw new TypeError(
        `[vitrum/gltf-adapter] buffers[${bv.buffer}] must be a non-shared ArrayBuffer.`,
      );
    }
    if (range.end > loadedByteLength) {
      throw new RangeError(
        `[vitrum/gltf-adapter] bufferViews[${image.bufferView}] range ` +
        `[${range.byteOffset}, ${range.end}) exceeds loaded buffers[${range.bufferIndex}] ` +
        `byteLength ${loadedByteLength}.`,
      );
    }
    const bytes = new Uint8Array(
      buf,
      range.byteOffset,
      range.byteLength,
    );
    const mimeType = image.mimeType ?? 'image/png';
    // A bufferView is a non-owning view of the already-observed parent buffer,
    // not a second encoded resource. Re-observe the parent identity so direct
    // internal callers still account for it while public imports deduplicate it.
    resourceContext?.ledger.chargeEncodedBytes(
      gltfBufferResourceKey(range.bufferIndex),
      loadedByteLength,
      `buffers[${range.bufferIndex}]`,
    );
    return { bytes, mimeType };
  }

  if (image.uri) {
    if (image.uri.startsWith('data:')) {
      const decoded = decodeDataUri(
        image.uri,
        warnings,
        imageIndex,
        image.name ?? String(imageIndex),
        onDiagnostic,
        resourceContext,
      );
      if (decoded != null) return decoded;
      const external = externalImages?.get(imageIndex);
      if (external != null) {
        return validatedExternalImageBytes(external, imageIndex, resourceContext);
      }
      return undefined;
    }
    const external = externalImages?.get(imageIndex);
    if (external != null) {
      return validatedExternalImageBytes(external, imageIndex, resourceContext);
    }
    emitTextureAcquisitionDiagnostic(warnings, onDiagnostic, {
      severity: 'warning',
      code: 'external-image-uri',
      path: `images[${imageIndex}].uri`,
      imageIndex,
      imageUri: image.uri,
      message:
        `[vitrum/gltf-adapter] Image "${image.name ?? imageIndex}" has a URI ` +
        `("${image.uri.substring(0, 60)}…"). The adapter does not fetch external image URIs. ` +
        'Embed the image in a bufferView or data: URI. Image skipped.',
    });
    return undefined;
  }

  emitTextureAcquisitionDiagnostic(warnings, onDiagnostic, {
    severity: 'warning',
    code: 'image-source-missing',
    path: `images[${imageIndex}]`,
    textureIndex,
    imageIndex,
    message:
      `[vitrum/gltf-adapter] Image "${image.name ?? imageIndex}" has neither bufferView nor uri. ` +
      'Image skipped.',
  });
  return undefined;
}

function decodeDataUri(
  uri: string,
  warnings: string[],
  imageIndex: number,
  label: string,
  onDiagnostic?: GltfTextureAcquisitionDiagnosticSink,
  resourceContext?: GltfImportResourceContext,
): { bytes: Uint8Array; mimeType: string } | undefined {
  const comma = uri.indexOf(',');
  if (comma < 0) {
    emitTextureAcquisitionDiagnostic(warnings, onDiagnostic, {
      severity: 'warning',
      code: 'malformed-data-uri',
      path: `images[${imageIndex}].uri`,
      imageIndex,
      imageUri: uri,
      message: `[vitrum/gltf-adapter] Image "${label}" has a malformed data: URI. Image skipped.`,
    });
    return undefined;
  }
  const meta = uri.slice(5, comma);
  const payload = uri.slice(comma + 1);
  const parts = meta.split(';').filter(Boolean);
  const mimeType = parts.find((p) => !p.includes('=')) ?? 'application/octet-stream';
  const isBase64 = parts.some((p) => p.toLowerCase() === 'base64');
  const resourceKey = gltfImageResourceKey(imageIndex);
  const path = `images[${imageIndex}].uri`;
  resourceContext?.ledger.ensureEncodedBytes(
    resourceKey,
    dataUriDecodedByteUpperBound(payload, isBase64),
    path,
  );
  let bytes: Uint8Array;
  try {
    if (isBase64) {
      if (typeof globalThis.atob !== 'function') {
        emitTextureAcquisitionDiagnostic(warnings, onDiagnostic, {
          severity: 'warning',
          code: 'data-uri-atob-unavailable',
          path: `images[${imageIndex}].uri`,
          imageIndex,
          imageUri: uri,
          message:
            `[vitrum/gltf-adapter] Image "${label}" uses base64 data: URI, but atob() is unavailable. ` +
            'Image skipped.',
        });
        return undefined;
      }
      const bin = globalThis.atob(payload.replace(/\s+/g, ''));
      bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
    } else {
      bytes = decodePercentEncodedDataUriPayload(payload);
    }
  } catch (err) {
    emitTextureAcquisitionDiagnostic(warnings, onDiagnostic, {
      severity: 'warning',
      code: 'data-uri-decode-failed',
      path: `images[${imageIndex}].uri`,
      imageIndex,
      imageUri: uri,
      message:
        `[vitrum/gltf-adapter] Image "${label}" data: URI could not be decoded: ` +
        `${err instanceof Error ? err.message : String(err)}. Image skipped.`,
    });
    return undefined;
  }
  resourceContext?.ledger.chargeEncodedBytes(
    resourceKey,
    bytes.byteLength,
    path,
  );
  return { bytes, mimeType };
}

function decodePercentEncodedDataUriPayload(payload: string): Uint8Array {
  const bytes = new Uint8Array(percentDecodedByteUpperBound(payload));
  let offset = 0;
  for (let index = 0; index < payload.length; index += 1) {
    const code = payload.charCodeAt(index);
    if (code === 0x25) {
      if (
        index + 2 >= payload.length ||
        !isHexCodeUnit(payload.charCodeAt(index + 1)) ||
        !isHexCodeUnit(payload.charCodeAt(index + 2))
      ) {
        throw new URIError(`URI malformed at payload code-unit ${index}`);
      }
      bytes[offset++] =
        (hexCodeUnitValue(payload.charCodeAt(index + 1)) << 4) |
        hexCodeUnitValue(payload.charCodeAt(index + 2));
      index += 2;
      continue;
    }

    let codePoint = payload.codePointAt(index)!;
    if (codePoint > 0xffff) {
      index += 1;
    } else if (codePoint >= 0xd800 && codePoint <= 0xdfff) {
      codePoint = 0xfffd;
    }
    if (codePoint <= 0x7f) {
      bytes[offset++] = codePoint;
    } else if (codePoint <= 0x7ff) {
      bytes[offset++] = 0xc0 | (codePoint >>> 6);
      bytes[offset++] = 0x80 | (codePoint & 0x3f);
    } else if (codePoint <= 0xffff) {
      bytes[offset++] = 0xe0 | (codePoint >>> 12);
      bytes[offset++] = 0x80 | ((codePoint >>> 6) & 0x3f);
      bytes[offset++] = 0x80 | (codePoint & 0x3f);
    } else {
      bytes[offset++] = 0xf0 | (codePoint >>> 18);
      bytes[offset++] = 0x80 | ((codePoint >>> 12) & 0x3f);
      bytes[offset++] = 0x80 | ((codePoint >>> 6) & 0x3f);
      bytes[offset++] = 0x80 | (codePoint & 0x3f);
    }
  }
  return bytes;
}

function hexCodeUnitValue(code: number): number {
  if (code >= 0x30 && code <= 0x39) return code - 0x30;
  if (code >= 0x41 && code <= 0x46) return code - 0x41 + 10;
  return code - 0x61 + 10;
}

function validatedExternalImageBytes(
  image: GltfImageBytes,
  imageIndex: number,
  resourceContext: GltfImportResourceContext | undefined,
): GltfImageBytes {
  const bytes = localUint8ArrayView(image.bytes);
  if (bytes === null) {
    throw new TypeError(
      `[vitrum/gltf-adapter] imageBytes[${imageIndex}].bytes must be a non-shared Uint8Array.`,
    );
  }
  if (typeof image.mimeType !== 'string' || image.mimeType.length === 0) {
    throw new TypeError(
      `[vitrum/gltf-adapter] imageBytes[${imageIndex}].mimeType must be a non-empty string.`,
    );
  }
  resourceContext?.ledger.chargeEncodedBytes(
    gltfImageResourceKey(imageIndex),
    bytes.byteLength,
    `imageBytes[${imageIndex}].bytes`,
  );
  return { bytes, mimeType: image.mimeType };
}

function dataUriDecodedByteUpperBound(
  payload: string,
  isBase64: boolean,
): number {
  return isBase64
    ? base64DecodedByteUpperBound(payload)
    : percentDecodedByteUpperBound(payload);
}

function base64DecodedByteUpperBound(payload: string): number {
  let encodedLength = 0;
  let previous = -1;
  let last = -1;
  for (let i = 0; i < payload.length; i += 1) {
    const code = payload.charCodeAt(i);
    if (isDataUriWhitespaceCodeUnit(code)) continue;
    encodedLength += 1;
    previous = last;
    last = code;
  }
  const padding = last === 0x3d ? (previous === 0x3d ? 2 : 1) : 0;
  const completeGroups = Math.floor(encodedLength / 4);
  const remainder = encodedLength % 4;
  const upperBound =
    completeGroups * 3 +
    Math.floor((remainder * 6) / 8) -
    padding;
  return Math.max(0, upperBound);
}

function isDataUriWhitespaceCodeUnit(code: number): boolean {
  return (
    (code >= 0x09 && code <= 0x0d) ||
    code === 0x20 ||
    code === 0xa0 ||
    code === 0x1680 ||
    (code >= 0x2000 && code <= 0x200a) ||
    code === 0x2028 ||
    code === 0x2029 ||
    code === 0x202f ||
    code === 0x205f ||
    code === 0x3000 ||
    code === 0xfeff
  );
}

function percentDecodedByteUpperBound(payload: string): number {
  let byteLength = 0;
  for (let i = 0; i < payload.length; i += 1) {
    if (
      payload.charCodeAt(i) === 0x25 &&
      i + 2 < payload.length &&
      isHexCodeUnit(payload.charCodeAt(i + 1)) &&
      isHexCodeUnit(payload.charCodeAt(i + 2))
    ) {
      byteLength = checkedByteLengthAdd(byteLength, 1);
      i += 2;
      continue;
    }
    const codePoint = payload.codePointAt(i)!;
    const width =
      codePoint <= 0x7f ? 1 :
      codePoint <= 0x7ff ? 2 :
      codePoint <= 0xffff ? 3 :
      4;
    byteLength = checkedByteLengthAdd(byteLength, width);
    if (codePoint > 0xffff) i += 1;
  }
  return byteLength;
}

function checkedByteLengthAdd(total: number, additional: number): number {
  if (additional > Number.MAX_SAFE_INTEGER - total) {
    throw new RangeError(
      '[vitrum/gltf-adapter] data: URI decoded byte length exceeds the safe integer range.',
    );
  }
  return total + additional;
}

function isHexCodeUnit(code: number): boolean {
  return (
    (code >= 0x30 && code <= 0x39) ||
    (code >= 0x41 && code <= 0x46) ||
    (code >= 0x61 && code <= 0x66)
  );
}

/** Decode a single glTF image to an opaque handle via the provided callback. */
async function decodeImage(
  bytes: Uint8Array,
  mimeType: string,
  decodeFn: DecodeImageFn | undefined,
  warnings: string[],
  imageIndex: number,
  onDiagnostic?: GltfTextureAcquisitionDiagnosticSink,
  resourceContext?: GltfImportResourceContext,
): Promise<unknown> {
  const path = `images[${imageIndex}]`;
  let handle: unknown;
  if (decodeFn) {
    handle = await decodeFn(bytes, mimeType);
  } else if (
    mimeType.trim().toLowerCase() === 'image/ktx2' ||
    isBasisKtx2Bytes(bytes)
  ) {
    // createImageBitmap does not portably decode KTX2. Preserve the encoded
    // bytes so decodeSceneTextures can run the built-in Basis transcoder in
    // both browser and Node.
    handle = {
      kind: 'raw-image',
      mimeType: 'image/ktx2',
      data: bytes,
    } satisfies RawImageHandle;
  } else if (
    mimeType.trim().toLowerCase() === 'image/vnd-ms.dds' ||
    isDdsBytes(bytes)
  ) {
    handle = {
      kind: 'raw-image',
      mimeType: 'image/vnd-ms.dds',
      data: bytes,
    } satisfies RawImageHandle;
  } else if (typeof createImageBitmap !== 'undefined') {
    // Browser path: createImageBitmap
    const ownedBytes = Reflect.apply(UINT8_ARRAY_SLICE, bytes, []) as Uint8Array;
    // Intrinsic Uint8Array#slice always creates a new, non-shared ArrayBuffer.
    const blob = new Blob([ownedBytes.buffer as ArrayBuffer], { type: mimeType });
    handle = await createImageBitmap(blob);
  } else {
    // Node / non-browser: return raw bytes.
    emitTextureAcquisitionDiagnostic(warnings, onDiagnostic, {
      severity: 'warning',
      code: 'image-decoder-missing',
      path,
      imageIndex,
      message:
        '[vitrum/gltf-adapter] No decodeImage callback provided and createImageBitmap is not ' +
        'available (non-browser environment). Images are returned as { kind: "raw-image", data, mimeType }. ' +
        'pt-webgpu and pt-webgl2 expect ImageBitmap or a canvas-compatible handle; supply ' +
        'opts.decodeImage to convert raw bytes to an appropriate backend handle.',
    });
    handle = { kind: 'raw-image', mimeType, data: bytes } satisfies RawImageHandle;
  }

  resourceContext?.decodedImageHandles.track(handle);
  if (resourceContext !== undefined) {
    const decodedDimensions = decodedHandleDimensions(handle);
    if (decodedDimensions !== null) {
      try {
        const pixels = checkedPixelCount(
          decodedDimensions.width,
          decodedDimensions.height,
          path,
        );
        resourceContext.ledger.chargeDecodedTexturePixels(pixels, path);
      } catch (cause) {
        resourceContext.decodedImageHandles.closeTracked(handle);
        throw cause;
      }
    }
  }
  return handle;
}

interface DecodedHandleDimensions {
  readonly width: number;
  readonly height: number;
}

function decodedHandleDimensions(
  handle: unknown,
): DecodedHandleDimensions | null {
  if (handle === null || (typeof handle !== 'object' && typeof handle !== 'function')) {
    return null;
  }
  let width: unknown;
  let height: unknown;
  try {
    const candidate = handle as { readonly width?: unknown; readonly height?: unknown };
    width = candidate.width;
    height = candidate.height;
  } catch {
    return null;
  }
  if (
    typeof width !== 'number' ||
    !Number.isFinite(width) ||
    typeof height !== 'number' ||
    !Number.isFinite(height)
  ) {
    return null;
  }
  return { width, height };
}

function checkedPixelCount(
  width: number,
  height: number,
  path: string,
): number {
  if (
    !Number.isSafeInteger(width) ||
    width <= 0 ||
    !Number.isSafeInteger(height) ||
    height <= 0
  ) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${path} decoded dimensions must be positive safe integers; ` +
      `received ${String(width)}x${String(height)}.`,
    );
  }
  if (width > Number.MAX_SAFE_INTEGER / height) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${path} decoded pixel count exceeds the safe integer range.`,
    );
  }
  return width * height;
}

function ensureDecodedTexturePixels(
  ledger: ImportResourceLedger,
  pixelCount: number,
  path: string,
  reservedPixelCount = 0,
): void {
  const perTextureLimit = ledger.limits.maxDecodedTexturePixels;
  if (perTextureLimit !== 0 && pixelCount > perTextureLimit) {
    throw new GltfResourceLimitError({
      limitKind: 'decoded-texture-pixels',
      limit: perTextureLimit,
      actual: pixelCount,
      path,
    });
  }
  if (
    reservedPixelCount > Number.MAX_SAFE_INTEGER - ledger.totalDecodedTexturePixels ||
    pixelCount > Number.MAX_SAFE_INTEGER - ledger.totalDecodedTexturePixels - reservedPixelCount
  ) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${path} total decoded texture pixels exceed the safe integer range.`,
    );
  }
  const total =
    ledger.totalDecodedTexturePixels +
    reservedPixelCount +
    pixelCount;
  const totalLimit = ledger.limits.maxTotalDecodedTexturePixels;
  if (totalLimit !== 0 && total > totalLimit) {
    throw new GltfResourceLimitError({
      limitKind: 'total-decoded-texture-pixels',
      limit: totalLimit,
      actual: total,
      path,
    });
  }
}

/**
 * Collect all texture handles referenced by the glTF, decoding images exactly
 * once per unique image index.
 *
 * Returns a Map from glTF texture index → decoded handle (Promise resolved).
 */
export async function buildTextureHandleMap(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  decodeFn: DecodeImageFn | undefined,
  warnings: string[],
  externalImages?: GltfImageBytesMap,
  textureSourceExtensions: readonly GltfTextureSourceExtension[] = [],
  onDiagnostic?: GltfTextureAcquisitionDiagnosticSink,
  textureIndices?: ReadonlySet<number>,
  resourceContext?: GltfImportResourceContext,
): Promise<Map<number, unknown>> {
  const effectiveResourceContext =
    resourceContext ?? createTextureImportResourceContext();
  const textures = gltf.textures ?? [];
  const imageInputs = new Map<number, {
    readonly bytes: Uint8Array;
    readonly mimeType: string;
  }>();
  const textureImageSources = new Map<number, SelectedTextureImageSource>();
  const externalImageMap = normalizeImageBytesMap(externalImages);
  const sourceExtensions = new Set(
    effectiveGltfTextureSourceExtensions(textureSourceExtensions),
  );
  let reservedDecodedPixels = 0;

  // Resolve and preflight every unique image before starting any decoder work.
  for (const [textureIndex, tex] of textures.entries()) {
    if (textureIndices !== undefined && !textureIndices.has(textureIndex)) continue;
    const selectedSource = resolveTextureImageSource(tex, textureIndex, sourceExtensions, warnings, onDiagnostic);
    textureImageSources.set(textureIndex, selectedSource);
    const imageIdx = selectedSource.imageIndex;
    if (imageIdx !== undefined && !imageInputs.has(imageIdx)) {
      const imgData = getImageBytes(
        gltf,
        buffers,
        imageIdx,
        textureIndex,
        warnings,
        onDiagnostic,
        externalImageMap,
        selectedSource,
        effectiveResourceContext,
      );
      if (imgData) {
        const path = `images[${imageIdx}]`;
        // Only signature-recognized containers are preflighted here. Hosts may
        // intentionally use opaque/custom payloads under a conventional MIME
        // label; their returned dimensions are still enforced after decode.
        const encodedDimensions = readEncodedImageDimensions(imgData.bytes);
        if (encodedDimensions !== null) {
          const pixels = checkedPixelCount(
            encodedDimensions.width,
            encodedDimensions.height,
            path,
          );
          ensureDecodedTexturePixels(
            effectiveResourceContext.ledger,
            pixels,
            path,
            reservedDecodedPixels,
          );
          reservedDecodedPixels += pixels;
        }
        imageInputs.set(imageIdx, {
          bytes: imgData.bytes,
          mimeType: imgData.mimeType,
        });
      }
    }
  }

  const imageEntries = [...imageInputs].map(([imageIndex, input]) => [
    imageIndex,
    effectiveResourceContext.limiter.run(() =>
      decodeImage(
        input.bytes,
        input.mimeType,
        decodeFn,
        warnings,
        imageIndex,
        onDiagnostic,
        effectiveResourceContext,
      )),
  ] as const);
  const settledImages = await Promise.allSettled(
    imageEntries.map(([, handle]) => handle),
  );
  const resolvedImages = new Map<number, unknown>();
  let hasFailure = false;
  let firstFailure: unknown;
  for (let i = 0; i < settledImages.length; i += 1) {
    const settled = settledImages[i]!;
    if (settled.status === 'rejected') {
      if (!hasFailure) firstFailure = settled.reason;
      hasFailure = true;
      continue;
    }
    resolvedImages.set(imageEntries[i]![0], settled.value);
  }
  if (hasFailure) {
    effectiveResourceContext.decodedImageHandles.rollback();
    throw firstFailure;
  }

  // Map each glTF texture index back to its deduplicated decoded image.
  const resolved = new Map<number, unknown>();
  for (const [texIdx] of textures.entries()) {
    if (textureIndices !== undefined && !textureIndices.has(texIdx)) continue;
    const imageIdx = textureImageSources.get(texIdx)?.imageIndex;
    if (imageIdx !== undefined) {
      if (resolvedImages.has(imageIdx)) {
        resolved.set(texIdx, resolvedImages.get(imageIdx));
      }
    }
  }

  return resolved;
}

function createTextureImportResourceContext(): GltfImportResourceContext {
  const ledger = new ImportResourceLedger();
  return {
    ledger,
    limiter: createAsyncResourceLimiter(
      ledger.limits.maxConcurrentResourceOperations,
    ),
    decodedImageHandles: new DecodedImageHandleOwner(),
  };
}

function resolveTextureImageSource(
  texture: GltfTexture,
  textureIndex: number,
  enabledExtensions: ReadonlySet<string>,
  warnings: string[],
  onDiagnostic?: GltfTextureAcquisitionDiagnosticSink,
): SelectedTextureImageSource {
  const selected = selectTextureImageSource(texture, textureIndex, enabledExtensions);
  if (selected.imageIndex !== undefined) return selected;

  const available: GltfTextureSourceExtension[] = [];
  for (const extName of GLTF_TEXTURE_SOURCE_EXTENSIONS) {
    if (texture.extensions?.[extName]?.source !== undefined) available.push(extName);
  }
  if (texture.source === undefined && available.length > 0) {
    const path = available.length === 1
      ? `textures[${textureIndex}].extensions.${available[0]}`
      : `textures[${textureIndex}].extensions`;
    emitTextureAcquisitionDiagnostic(warnings, onDiagnostic, {
      severity: 'warning',
      code: 'disabled-texture-source-extension',
      path,
      textureIndex,
      textureSourceExtensions: available,
      message:
        `[vitrum/gltf-adapter] Texture at textures[${textureIndex}] uses ${available.join(', ')} but none of those ` +
        'texture-source extensions were enabled. Pass opts.textureSourceExtensions to select ' +
        'an alternate image source. Texture skipped.',
    });
  }
  return texture.source !== undefined
    ? { imageIndex: texture.source, path: `textures[${textureIndex}].source` }
    : {};
}

function emitTextureAcquisitionDiagnostic(
  warnings: string[],
  onDiagnostic: GltfTextureAcquisitionDiagnosticSink | undefined,
  diagnostic: GltfTextureAcquisitionDiagnostic,
): void {
  warnings.push(diagnostic.message);
  try {
    onDiagnostic?.(diagnostic);
  } catch {
    // Host diagnostic callbacks must not abort texture acquisition.
  }
}

function selectTextureImageSource(
  texture: GltfTexture,
  textureIndex: number,
  enabledExtensions: ReadonlySet<string>,
): SelectedTextureImageSource {
  for (const extName of GLTF_TEXTURE_SOURCE_EXTENSIONS) {
    if (!enabledExtensions.has(extName)) continue;
    const source = texture.extensions?.[extName]?.source;
    if (source !== undefined) {
      return {
        imageIndex: source,
        path: `textures[${textureIndex}].extensions.${extName}.source`,
        textureSourceExtension: extName,
      };
    }
  }
  return texture.source !== undefined
    ? { imageIndex: texture.source, path: `textures[${textureIndex}].source` }
    : {};
}

function normalizeImageBytesMap(
  images: GltfImageBytesMap | undefined,
): ReadonlyMap<number, GltfImageBytes> | undefined {
  if (images == null) return undefined;
  if (isReadonlyImageBytesMap(images)) return images;
  const out = new Map<number, GltfImageBytes>();
  for (const [k, v] of Object.entries(images)) out.set(Number(k), v);
  return out;
}

function isReadonlyImageBytesMap(
  value: GltfImageBytesMap,
): value is ReadonlyMap<number, GltfImageBytes> {
  const candidate = value as unknown as {
    readonly get?: unknown;
    readonly entries?: unknown;
    readonly size?: unknown;
  };
  return typeof candidate.get === 'function' &&
    typeof candidate.entries === 'function' &&
    typeof candidate.size === 'number';
}

/** Extract a UvTransform from a KHR_texture_transform extension block. */
function uvTransformFromExt(
  ext: { offset?: [number, number]; rotation?: number; scale?: [number, number]; texCoord?: number } | undefined,
): UvTransform | undefined {
  if (!ext) return undefined;
  const hasOffset = ext.offset && (ext.offset[0] !== 0 || ext.offset[1] !== 0);
  const hasScale = ext.scale && (ext.scale[0] !== 1 || ext.scale[1] !== 1);
  const hasRot = ext.rotation !== undefined && ext.rotation !== 0;
  if (!hasOffset && !hasScale && !hasRot) return undefined;
  const result: UvTransform = {};
  if (ext.offset) (result as { offset?: readonly [number, number] }).offset = ext.offset;
  if (ext.scale) (result as { scale?: readonly [number, number] }).scale = ext.scale;
  if (ext.rotation !== undefined) (result as { rotation?: number }).rotation = ext.rotation;
  return result;
}

function textureWrapMode(value: number | undefined): TextureWrapMode {
  if (value === 33071) return 'clamp-to-edge';
  if (value === 33648) return 'mirrored-repeat';
  return 'repeat';
}

function textureMagFilterMode(value: number | undefined): TextureFilterMode | undefined {
  if (value === 9728) return 'nearest';
  if (value === 9729) return 'linear';
  return undefined;
}

function textureMinFilterModes(value: number | undefined): {
  readonly minFilter?: TextureFilterMode;
  readonly mipFilter?: TextureMipFilterMode;
} {
  switch (value) {
    case 9728:
      return { minFilter: 'nearest', mipFilter: 'none' };
    case 9729:
      return { minFilter: 'linear', mipFilter: 'none' };
    case 9984:
      return { minFilter: 'nearest', mipFilter: 'nearest' };
    case 9985:
      return { minFilter: 'linear', mipFilter: 'nearest' };
    case 9986:
      return { minFilter: 'nearest', mipFilter: 'linear' };
    case 9987:
      return { minFilter: 'linear', mipFilter: 'linear' };
    default:
      return {};
  }
}

function rawImageHandleMimeType(handle: unknown): string | undefined {
  if (handle === null || typeof handle !== 'object') return undefined;
  const raw = handle as Partial<RawImageHandle>;
  return raw.kind === 'raw-image' && typeof raw.mimeType === 'string'
    ? raw.mimeType
    : undefined;
}

/**
 * Resolve a GltfTextureInfo to a TextureRef, given the decoded handle map.
 * Returns undefined if the texture is missing or its image failed to decode.
 */
export function resolveTextureRef(
  info: { index: number; texCoord?: number; extensions?: Record<string, unknown> } | undefined,
  handleMap: Map<number, unknown>,
  gltf?: Pick<GltfJson, 'textures' | 'samplers' | 'images'>,
  sourcePath?: string,
  textureSourceExtensions: readonly GltfTextureSourceExtension[] = [],
): TextureRef | undefined {
  if (!info) return undefined;
  const handle = handleMap.get(info.index);
  if (handle == null) return undefined;

  const khrTransform = (info.extensions?.['KHR_texture_transform'] as
    | { offset?: [number, number]; rotation?: number; scale?: [number, number]; texCoord?: number }
    | undefined);
  const texCoord = khrTransform?.texCoord ?? info.texCoord ?? 0;
  const transform = uvTransformFromExt(khrTransform);
  const texture = gltf?.textures?.[info.index];
  const samplerIdx = texture?.sampler;
  const sampler = samplerIdx !== undefined ? gltf?.samplers?.[samplerIdx] : undefined;
  const wrapS = textureWrapMode(sampler?.wrapS);
  const wrapT = textureWrapMode(sampler?.wrapT);
  const magFilter = textureMagFilterMode(sampler?.magFilter);
  const { minFilter, mipFilter } = textureMinFilterModes(sampler?.minFilter);
  const selectedSource = texture !== undefined
    ? selectTextureImageSource(texture, info.index, new Set(textureSourceExtensions))
    : {};
  const image = selectedSource.imageIndex !== undefined
    ? gltf?.images?.[selectedSource.imageIndex]
    : undefined;
  const imageMimeType = image?.mimeType ?? rawImageHandleMimeType(handle);

  const ref: TextureRef = {
    handle,
    ...(texCoord !== 0 ? { texCoord } : {}),
    ...(transform ? { transform } : {}),
    ...(wrapS !== 'repeat' ? { wrapS } : {}),
    ...(wrapT !== 'repeat' ? { wrapT } : {}),
    ...(magFilter !== undefined ? { magFilter } : {}),
    ...(minFilter !== undefined ? { minFilter } : {}),
    ...(mipFilter !== undefined ? { mipFilter } : {}),
  };
  const source: GltfTextureRefSource = {
    path: sourcePath ?? `textures[${info.index}]`,
    ...(selectedSource.path !== undefined ? { imageSourcePath: selectedSource.path } : {}),
    textureIndex: info.index,
    ...(selectedSource.imageIndex !== undefined ? { imageIndex: selectedSource.imageIndex } : {}),
    ...(samplerIdx !== undefined ? { samplerIndex: samplerIdx } : {}),
    ...(image?.uri !== undefined ? { imageUri: image.uri } : {}),
    ...(imageMimeType !== undefined ? { imageMimeType } : {}),
    ...(selectedSource.textureSourceExtension !== undefined
      ? { textureSourceExtension: selectedSource.textureSourceExtension }
      : {}),
  };
  return attachGltfTextureRefSource(ref, source);
}
