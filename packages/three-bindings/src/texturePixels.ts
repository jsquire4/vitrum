// texturePixels — extract RGBA-float CPU pixels from ANY THREE material texture, for
// the THREE-free path tracers (`@vitrum/pt-webgl2`, `@vitrum/pt-webgpu`) whose atlas
// packers read raw pixels, not a `THREE.Texture`. This is the material-map analogue of
// the env `equirectTextureToPayload` (G2) — the on-ramp owns the THREE/DOM knowledge so
// the backends stay THREE-free.
//
// Two pixel sources:
//   • DataTexture          — `tex.image.data` typed array (read directly).
//   • Image/ImageBitmap/   — drawn to a 2D canvas, `getImageData` read back (the G3.1
//     Canvas/Video           DOM bridge; returns null in a non-DOM/Node context).
//
// Orientation + colour-space mirror what THREE uploads to the GPU (= what the fork's
// CopyMaterial bakes into its linear texture array), so native matches the fork:
//   • rows reversed iff `tex.flipY` (image textures default true; DataTextures false),
//   • sRGB → linear when `tex.colorSpace === 'srgb'` (the fork's array is linear; the
//     CopyMaterial decodes the sRGB source on sample). Alpha is never sRGB-decoded.

import type * as THREE from 'three';

const SRGB_COLOR_SPACE = 'srgb'; // THREE.SRGBColorSpace string value (avoid a runtime THREE import)

function halfToFloat(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * 2 ** -14 * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * 2 ** (e - 15) * (1 + f / 1024);
}

/** Standard sRGB → linear transfer (per-channel, colour channels only). */
function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

export interface RawTexturePayload {
  readonly width: number;
  readonly height: number;
  readonly data: Float32Array; // RGBA, row-major, linear
}

/** Read a DataTexture's typed-array image into RGBA-float, flipping rows + decoding. */
function fromTypedArray(
  src: ArrayLike<number>, width: number, height: number, flip: boolean, decode: (v: number) => number,
): RawTexturePayload {
  const stride = Math.max(1, Math.round(src.length / (width * height))); // RGBA 4 / RGB 3 / R 1
  const isHalf = src instanceof Uint16Array;
  const isFloat = src instanceof Float32Array;
  const intMax = isHalf || isFloat ? 0 : 2 ** (8 * ((src as { BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT ?? 1)) - 1;
  const raw = (v: number): number => (isHalf ? halfToFloat(v) : isFloat ? v : intMax > 0 ? v / intMax : v);
  const out = new Float32Array(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    const sy = flip ? height - 1 - y : y;
    for (let x = 0; x < width; x += 1) {
      const s = (sy * width + x) * stride;
      const d = (y * width + x) * 4;
      out[d] = decode(raw(Number(src[s] ?? 0)));
      out[d + 1] = decode(raw(Number(src[s + (stride > 1 ? 1 : 0)] ?? 0)));
      out[d + 2] = decode(raw(Number(src[s + (stride > 2 ? 2 : 0)] ?? 0)));
      out[d + 3] = stride >= 4 ? raw(Number(src[s + 3] ?? 1)) : 1;
    }
  }
  return { width, height, data: out };
}

/** Draw a drawable image to a 2D canvas + read back RGBA-float. null without a DOM. */
function fromDrawable(img: CanvasImageSource, flip: boolean, decode: (v: number) => number): RawTexturePayload | null {
  const dims = img as { width?: number; height?: number; naturalWidth?: number; naturalHeight?: number; videoWidth?: number; videoHeight?: number };
  const width = Number(dims.naturalWidth ?? dims.videoWidth ?? dims.width ?? 0);
  const height = Number(dims.naturalHeight ?? dims.videoHeight ?? dims.height ?? 0);
  if (!width || !height || !Number.isFinite(width) || !Number.isFinite(height)) return null;

  let canvas: { getContext(id: '2d'): CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null };
  if (typeof OffscreenCanvas !== 'undefined') {
    canvas = new OffscreenCanvas(width, height);
  } else if (typeof document !== 'undefined') {
    const c = document.createElement('canvas');
    c.width = width; c.height = height;
    canvas = c;
  } else {
    return null; // no DOM (e.g. Node unit tests) → caller falls back to the THREE handle
  }
  const ctx = canvas.getContext('2d');
  if (ctx == null) return null;
  ctx.drawImage(img, 0, 0, width, height);
  let px: Uint8ClampedArray;
  try {
    px = ctx.getImageData(0, 0, width, height).data;
  } catch {
    return null; // CORS-tainted canvas
  }

  const out = new Float32Array(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    const sy = flip ? height - 1 - y : y;
    for (let x = 0; x < width; x += 1) {
      const s = (sy * width + x) * 4;
      const d = (y * width + x) * 4;
      out[d] = decode(px[s]! / 255);
      out[d + 1] = decode(px[s + 1]! / 255);
      out[d + 2] = decode(px[s + 2]! / 255);
      out[d + 3] = px[s + 3]! / 255;
    }
  }
  return { width, height, data: out };
}

/**
 * Extract RGBA-float linear pixels from a THREE material texture, or `null` when no
 * readable source exists (GPU-only texture, CORS-tainted image, or no DOM available).
 */
export function materialTextureToPayload(tex: THREE.Texture): RawTexturePayload | null {
  const img = tex.image as
    | ({ data?: ArrayLike<number>; width?: number; height?: number } & CanvasImageSource)
    | null
    | undefined;
  if (img == null) return null;
  const flip = tex.flipY !== false; // image textures default flipY=true; DataTextures false
  const isSrgb = (tex.colorSpace as unknown as string) === SRGB_COLOR_SPACE;
  const decode = isSrgb ? srgbToLinear : (v: number): number => v;

  if (img.data != null && typeof img.data.length === 'number' && img.width && img.height) {
    return fromTypedArray(img.data, img.width, img.height, flip, decode);
  }
  return fromDrawable(img as CanvasImageSource, flip, decode);
}
