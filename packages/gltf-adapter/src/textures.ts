// textures.ts — glTF image → TextureRef bridge.
//
// The adapter exposes images through a pluggable decode callback. Default
// behaviour differs by environment:
//   - Browser: createImageBitmap (returns an ImageBitmap usable by pt-webgpu
//     and pt-webgl2 as a texImage2D source).
//   - Non-browser: raw bytes are wrapped in { mimeType, data: Uint8Array }.
//     pt-webgpu's materialTextureArray.ts accepts duck-typed objects; see
//     README for the exact shapes accepted per backend.
//
// sRGB vs linear ownership note:
//   The ADAPTER does NOT apply colorspace conversion.
//   - baseColor / emissive map images carry sRGB data; the BACKEND is
//     responsible for sRGB-decode on upload (gl.SRGB8_ALPHA8, GPUTextureFormat
//     'rgba8unorm-srgb', or a manual gamma decode in the shader).
//   - normal / ORM / ao / lightMap / bumpMap images are linear data;
//     backends must NOT apply sRGB-decode to these.
//   The README's support matrix documents this ownership boundary.

import type { GltfJson } from './gltfTypes.js';
import type { TextureRef, UvTransform } from '@vitrum/core';

export type DecodeImageFn = (
  bytes: Uint8Array,
  mimeType: string,
) => Promise<unknown>;

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

/**
 * Resolve a glTF image index to bytes + mimeType from a bufferView.
 * URI-based images are skipped (the adapter does not fetch).
 */
function getImageBytes(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  imageIndex: number,
  warnings: string[],
): { bytes: Uint8Array; mimeType: string } | undefined {
  const image = gltf.images?.[imageIndex];
  if (!image) return undefined;

  if (image.bufferView !== undefined) {
    const bv = gltf.bufferViews?.[image.bufferView];
    if (!bv) return undefined;
    const buf = buffers.get(bv.buffer);
    if (!buf) return undefined;
    const bytes = new Uint8Array(buf, bv.byteOffset ?? 0, bv.byteLength);
    const mimeType = image.mimeType ?? 'image/png';
    return { bytes, mimeType };
  }

  if (image.uri) {
    warnings.push(
      `[vitrum/gltf-adapter] Image "${image.name ?? imageIndex}" has a URI ("${image.uri.substring(0, 60)}…"). ` +
        'The adapter does not fetch URIs. Supply pre-loaded buffers via opts.buffers ' +
        'or provide an opts.decodeImage callback that handles data: URIs. Image skipped.',
    );
    return undefined;
  }

  return undefined;
}

/** Decode a single glTF image to an opaque handle via the provided callback. */
async function decodeImage(
  bytes: Uint8Array,
  mimeType: string,
  decodeFn: DecodeImageFn | undefined,
  warnings: string[],
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
  warnings.push(
    '[vitrum/gltf-adapter] No decodeImage callback provided and createImageBitmap is not ' +
      'available (non-browser environment). Images are returned as { kind: "raw-image", data, mimeType }. ' +
      'pt-webgpu and pt-webgl2 expect ImageBitmap or a canvas-compatible handle; supply ' +
      'opts.decodeImage to convert raw bytes to an appropriate backend handle.',
  );
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
): Promise<Map<number, unknown>> {
  const textures = gltf.textures ?? [];
  const imageHandles = new Map<number, Promise<unknown>>();

  // Kick off unique image decodes in parallel.
  for (const tex of textures) {
    if (tex.source !== undefined && !imageHandles.has(tex.source)) {
      const imageIdx = tex.source;
      const imgData = getImageBytes(gltf, buffers, imageIdx, warnings);
      if (imgData) {
        imageHandles.set(
          imageIdx,
          decodeImage(imgData.bytes, imgData.mimeType, decodeFn, warnings),
        );
      }
    }
  }

  // Await all.
  const resolved = new Map<number, unknown>();
  for (const [texIdx, tex] of textures.entries()) {
    if (tex.source !== undefined) {
      const p = imageHandles.get(tex.source);
      if (p) resolved.set(texIdx, await p);
    }
  }

  return resolved;
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

/**
 * Resolve a GltfTextureInfo to a TextureRef, given the decoded handle map.
 * Returns undefined if the texture is missing or its image failed to decode.
 */
export function resolveTextureRef(
  info: { index: number; texCoord?: number; extensions?: Record<string, unknown> } | undefined,
  handleMap: Map<number, unknown>,
): TextureRef | undefined {
  if (!info) return undefined;
  const handle = handleMap.get(info.index);
  if (handle == null) return undefined;

  const khrTransform = (info.extensions?.['KHR_texture_transform'] as
    | { offset?: [number, number]; rotation?: number; scale?: [number, number]; texCoord?: number }
    | undefined);
  const texCoord = khrTransform?.texCoord ?? info.texCoord ?? 0;
  const transform = uvTransformFromExt(khrTransform);

  const ref: TextureRef = {
    handle,
    ...(texCoord !== 0 ? { texCoord } : {}),
    ...(transform ? { transform } : {}),
  };
  return ref;
}
