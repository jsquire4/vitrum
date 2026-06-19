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

export type GltfTextureSourceExtension =
  | 'KHR_texture_basisu'
  | 'EXT_texture_webp'
  | 'MSFT_texture_dds';

export const GLTF_TEXTURE_SOURCE_EXTENSIONS: readonly GltfTextureSourceExtension[] = [
  'KHR_texture_basisu',
  'EXT_texture_webp',
  'MSFT_texture_dds',
];

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

export interface GltfTextureRefSource {
  readonly path: string;
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

export function attachGltfTextureRefSource(ref: TextureRef, source: GltfTextureRefSource | undefined): TextureRef {
  if (source === undefined) return ref;
  return {
    ...ref,
    [GLTF_TEXTURE_REF_SOURCE]: source,
  } as GltfTextureRefWithSource;
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
): { bytes: Uint8Array; mimeType: string } | undefined {
  const image = gltf.images?.[imageIndex];
  if (!image) {
    emitTextureAcquisitionDiagnostic(warnings, onDiagnostic, {
      severity: 'warning',
      code: 'image-not-found',
      path: `textures[${textureIndex}].source`,
      textureIndex,
      imageIndex,
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
    const buf = buffers.get(bv.buffer);
    if (!buf) {
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
    const bytes = new Uint8Array(buf, bv.byteOffset ?? 0, bv.byteLength);
    const mimeType = image.mimeType ?? 'image/png';
    return { bytes, mimeType };
  }

  if (image.uri) {
    if (image.uri.startsWith('data:')) {
      const decoded = decodeDataUri(image.uri, warnings, imageIndex, image.name ?? String(imageIndex), onDiagnostic);
      if (decoded != null) return decoded;
      const external = externalImages?.get(imageIndex);
      if (external != null) return external;
      return undefined;
    }
    const external = externalImages?.get(imageIndex);
    if (external != null) return external;
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
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
      return { bytes, mimeType };
    }
    return { bytes: new TextEncoder().encode(decodeURIComponent(payload)), mimeType };
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
}

/** Decode a single glTF image to an opaque handle via the provided callback. */
async function decodeImage(
  bytes: Uint8Array,
  mimeType: string,
  decodeFn: DecodeImageFn | undefined,
  warnings: string[],
  imageIndex: number,
  onDiagnostic?: GltfTextureAcquisitionDiagnosticSink,
): Promise<unknown> {
  if (decodeFn) {
    return decodeFn(bytes, mimeType);
  }

  // Browser path: createImageBitmap
  if (typeof createImageBitmap !== 'undefined') {
    const slice = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const blob = new Blob([slice as ArrayBuffer], { type: mimeType });
    return createImageBitmap(blob);
  }

  // Node / non-browser: return raw bytes.
  emitTextureAcquisitionDiagnostic(warnings, onDiagnostic, {
    severity: 'warning',
    code: 'image-decoder-missing',
    path: `images[${imageIndex}]`,
    imageIndex,
    message:
      '[vitrum/gltf-adapter] No decodeImage callback provided and createImageBitmap is not ' +
      'available (non-browser environment). Images are returned as { kind: "raw-image", data, mimeType }. ' +
      'pt-webgpu and pt-webgl2 expect ImageBitmap or a canvas-compatible handle; supply ' +
      'opts.decodeImage to convert raw bytes to an appropriate backend handle.',
  });
  return { kind: 'raw-image', mimeType, data: bytes } satisfies RawImageHandle;
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
): Promise<Map<number, unknown>> {
  const textures = gltf.textures ?? [];
  const imageHandles = new Map<number, Promise<unknown>>();
  const textureImageSources = new Map<number, number | undefined>();
  const externalImageMap = normalizeImageBytesMap(externalImages);
  const sourceExtensions = new Set(textureSourceExtensions);

  // Kick off unique image decodes in parallel.
  for (const [textureIndex, tex] of textures.entries()) {
    if (textureIndices !== undefined && !textureIndices.has(textureIndex)) continue;
    const imageIdx = resolveTextureImageSource(tex, textureIndex, sourceExtensions, warnings, onDiagnostic);
    textureImageSources.set(textureIndex, imageIdx);
    if (imageIdx !== undefined && !imageHandles.has(imageIdx)) {
      const imgData = getImageBytes(gltf, buffers, imageIdx, textureIndex, warnings, onDiagnostic, externalImageMap);
      if (imgData) {
        imageHandles.set(
          imageIdx,
          decodeImage(imgData.bytes, imgData.mimeType, decodeFn, warnings, imageIdx, onDiagnostic),
        );
      }
    }
  }

  // Await all.
  const resolved = new Map<number, unknown>();
  for (const [texIdx] of textures.entries()) {
    if (textureIndices !== undefined && !textureIndices.has(texIdx)) continue;
    const imageIdx = textureImageSources.get(texIdx);
    if (imageIdx !== undefined) {
      const p = imageHandles.get(imageIdx);
      if (p) resolved.set(texIdx, await p);
    }
  }

  return resolved;
}

function resolveTextureImageSource(
  texture: GltfTexture,
  textureIndex: number,
  enabledExtensions: ReadonlySet<string>,
  warnings: string[],
  onDiagnostic?: GltfTextureAcquisitionDiagnosticSink,
): number | undefined {
  const selected = selectTextureImageSource(texture, enabledExtensions);
  if (selected.imageIndex !== undefined) return selected.imageIndex;

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
  return texture.source;
}

function emitTextureAcquisitionDiagnostic(
  warnings: string[],
  onDiagnostic: GltfTextureAcquisitionDiagnosticSink | undefined,
  diagnostic: GltfTextureAcquisitionDiagnostic,
): void {
  warnings.push(diagnostic.message);
  onDiagnostic?.(diagnostic);
}

function selectTextureImageSource(
  texture: GltfTexture,
  enabledExtensions: ReadonlySet<string>,
): {
  readonly imageIndex?: number;
  readonly textureSourceExtension?: GltfTextureSourceExtension;
} {
  for (const extName of GLTF_TEXTURE_SOURCE_EXTENSIONS) {
    if (!enabledExtensions.has(extName)) continue;
    const source = texture.extensions?.[extName]?.source;
    if (source !== undefined) return { imageIndex: source, textureSourceExtension: extName };
  }
  return texture.source !== undefined ? { imageIndex: texture.source } : {};
}

function normalizeImageBytesMap(
  images: GltfImageBytesMap | undefined,
): ReadonlyMap<number, GltfImageBytes> | undefined {
  if (images == null) return undefined;
  if (images instanceof Map) return images;
  const out = new Map<number, GltfImageBytes>();
  for (const [k, v] of Object.entries(images)) out.set(Number(k), v);
  return out;
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
    ? selectTextureImageSource(texture, new Set(textureSourceExtensions))
    : {};
  const image = selectedSource.imageIndex !== undefined
    ? gltf?.images?.[selectedSource.imageIndex]
    : undefined;

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
    textureIndex: info.index,
    ...(selectedSource.imageIndex !== undefined ? { imageIndex: selectedSource.imageIndex } : {}),
    ...(samplerIdx !== undefined ? { samplerIndex: samplerIdx } : {}),
    ...(image?.uri !== undefined ? { imageUri: image.uri } : {}),
    ...(image?.mimeType !== undefined ? { imageMimeType: image.mimeType } : {}),
    ...(selectedSource.textureSourceExtension !== undefined
      ? { textureSourceExtension: selectedSource.textureSourceExtension }
      : {}),
  };
  return attachGltfTextureRefSource(ref, source);
}
