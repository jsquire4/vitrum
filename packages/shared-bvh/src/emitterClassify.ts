/**
 * emitterClassify.ts — emitter-classification + Beer-Lambert + surface-texture
 * helpers extracted from materialEntry.ts (D11.5).
 *
 * These are the `MaterialSpec` field readers used by the ReSTIR/DDGI/RC emitter
 * list and per-triangle color/glow packing
 * (`walkaround-hybrid/src/restir/{packingHelpers,emitterList}.ts`). Every field
 * they read lives on core `MaterialSpec`, with optional backend escape hatches
 * carried through `material.extensions`.
 *
 * `extensions['surfaceTextureId']` and `extensions['skipEmitter']` are explicit
 * core-scene contract lanes. A host feeding a core `Scene` must set those values
 * directly to exercise the surface-texture and emitter-suppression paths.
 */

import type { MaterialSpec, TextureRef } from '@vitrum/core';
import {
  MATERIAL_TRANSMISSIVE_COLOR_THRESHOLD,
  MATERIAL_EMITTER_TRANSMISSION_THRESHOLD,
  MATERIAL_EMITTER_SUN_DOT_THRESHOLD,
  MATERIAL_DEFAULT_TRI_COLOR,
} from './materialEntry.js';

interface ReadableTextureHandle {
  readonly width?: number;
  readonly height?: number;
  readonly data?: ArrayLike<number>;
  readonly image?: { readonly width?: number; readonly height?: number; readonly data?: ArrayLike<number> };
  readonly __vitrum_hint__?: TextureHandleHint;
  readonly channels?: number;
  readonly dataType?: string;
  readonly colorSpace?: string;
}

interface TextureHandleHint {
  readonly channels?: 1 | 2 | 3 | 4;
  readonly dataType?: 'uint8' | 'uint16' | 'float32';
  readonly colorSpace?: 'srgb' | 'linear';
}

type MutableTextureHandleHint = {
  -readonly [K in keyof TextureHandleHint]?: TextureHandleHint[K];
};

export type BarycentricWeights = readonly [number, number, number];

interface TexelClipVertex {
  readonly weights: BarycentricWeights;
  readonly texUv: readonly [number, number];
}

interface TextureCellInterval {
  readonly lo: number;
  readonly hi: number;
  readonly texel: number;
}

function halfToFloat(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * 2 ** -14 * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * 2 ** (e - 15) * (1 + f / 1024);
}

function srgbToLinear(v: number): number {
  const c = Math.max(0, Math.min(1, v));
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function textureHint(handle: ReadableTextureHandle): TextureHandleHint | undefined {
  if (handle.__vitrum_hint__ != null) return handle.__vitrum_hint__;
  if (handle.channels == null && handle.dataType == null && handle.colorSpace == null) return undefined;
  const hint: MutableTextureHandleHint = {};
  if (
    handle.channels === 1 ||
    handle.channels === 2 ||
    handle.channels === 3 ||
    handle.channels === 4
  ) {
    hint.channels = handle.channels;
  }
  if (
    handle.dataType === 'uint8' ||
    handle.dataType === 'uint16' ||
    handle.dataType === 'float32'
  ) {
    hint.dataType = handle.dataType;
  }
  if (handle.colorSpace === 'srgb' || handle.colorSpace === 'linear') {
    hint.colorSpace = handle.colorSpace;
  }
  return hint;
}

function readableTextureDimensions(ref: TextureRef | undefined): { width: number; height: number } | null {
  const handle = ref?.handle as ReadableTextureHandle | null | undefined;
  if (handle == null) return null;
  const src = handle.data ?? handle.image?.data;
  const width = Math.floor(Number(handle.width ?? handle.image?.width ?? 0));
  const height = Math.floor(Number(handle.height ?? handle.image?.height ?? 0));
  if (src == null || typeof src.length !== 'number' || width <= 0 || height <= 0) {
    return null;
  }
  return { width, height };
}

function averageReadableTextureRgb(
  ref: TextureRef | undefined,
  fieldColorSpace: 'srgb' | 'linear',
): [number, number, number] | null {
  const handle = ref?.handle as ReadableTextureHandle | null | undefined;
  if (handle == null) return null;

  const src = handle.data ?? handle.image?.data;
  const width = Number(handle.width ?? handle.image?.width ?? 0);
  const height = Number(handle.height ?? handle.image?.height ?? 0);
  if (src == null || typeof src.length !== 'number' || width <= 0 || height <= 0) {
    return null;
  }

  const pixelCount = Math.floor(width) * Math.floor(height);
  if (pixelCount <= 0) return null;
  const hint = textureHint(handle);
  const heuristicStride = Math.max(1, Math.round(src.length / pixelCount));
  const stride = hint?.channels ?? heuristicStride;
  if (stride < 1 || stride > 4 || src.length < pixelCount * stride) {
    return null;
  }

  const isHalf = src instanceof Uint16Array;
  const isFloat = src instanceof Float32Array;
  const useHalf = hint?.dataType != null ? hint.dataType === 'uint16' : isHalf;
  const useFloat = hint?.dataType != null ? hint.dataType === 'float32' : isFloat;
  const bpe = (src as { readonly BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT ?? 1;
  const intMax = useHalf || useFloat ? 0 : 2 ** (8 * bpe) - 1;
  const decode = (v: number): number => (
    useHalf ? halfToFloat(v) : useFloat ? v : intMax > 0 ? v / intMax : v
  );
  const needsSrgbDecode = fieldColorSpace === 'srgb' && hint?.colorSpace !== 'linear';

  let r = 0;
  let g = 0;
  let b = 0;
  let n = 0;
  for (let p = 0; p < pixelCount; p += 1) {
    const base = p * stride;
    let pr = decode(Number(src[base] ?? 0));
    let pg = decode(Number(src[base + (stride > 1 ? 1 : 0)] ?? 0));
    let pb = decode(Number(src[base + (stride > 2 ? 2 : 0)] ?? 0));
    if (needsSrgbDecode) {
      pr = srgbToLinear(pr);
      pg = srgbToLinear(pg);
      pb = srgbToLinear(pb);
    }
    if (!Number.isFinite(pr) || !Number.isFinite(pg) || !Number.isFinite(pb)) continue;
    r += pr;
    g += pg;
    b += pb;
    n += 1;
  }

  return n > 0 ? [r / n, g / n, b / n] : null;
}

function wrapUv1(value: number, mode: TextureRef['wrapS']): number {
  if (mode === 'clamp-to-edge') return Math.min(1, Math.max(0, value));
  if (mode === 'mirrored-repeat') {
    const f = value * 0.5 - Math.floor(value * 0.5);
    return 1 - Math.abs(f * 2 - 1);
  }
  return value - Math.floor(value);
}

function transformTextureUv(ref: TextureRef, uv: readonly [number, number]): [number, number] {
  const raw = transformTextureUvUnwrapped(ref, uv);
  return [wrapUv1(raw[0], ref.wrapS), wrapUv1(raw[1], ref.wrapT)];
}

function transformTextureUvUnwrapped(ref: TextureRef, uv: readonly [number, number]): [number, number] {
  const sx = ref.transform?.scale?.[0] ?? 1;
  const sy = ref.transform?.scale?.[1] ?? 1;
  const x = uv[0] * sx;
  const y = uv[1] * sy;
  const rot = ref.transform?.rotation ?? 0;
  const c = Math.cos(rot);
  const s = Math.sin(rot);
  const ox = ref.transform?.offset?.[0] ?? 0;
  const oy = ref.transform?.offset?.[1] ?? 0;
  return [
    x * c - y * s + ox,
    x * s + y * c + oy,
  ];
}

function readTextureRgbAtTexel(
  ref: TextureRef | undefined,
  x: number,
  y: number,
  fieldColorSpace: 'srgb' | 'linear',
): [number, number, number] | null {
  const handle = ref?.handle as ReadableTextureHandle | null | undefined;
  if (handle == null) return null;

  const src = handle.data ?? handle.image?.data;
  const width = Math.floor(Number(handle.width ?? handle.image?.width ?? 0));
  const height = Math.floor(Number(handle.height ?? handle.image?.height ?? 0));
  if (src == null || typeof src.length !== 'number' || width <= 0 || height <= 0) {
    return null;
  }

  const pixelCount = width * height;
  const hint = textureHint(handle);
  const heuristicStride = Math.max(1, Math.round(src.length / pixelCount));
  const stride = hint?.channels ?? heuristicStride;
  if (stride < 1 || stride > 4 || src.length < pixelCount * stride) {
    return null;
  }

  const isHalf = src instanceof Uint16Array;
  const isFloat = src instanceof Float32Array;
  const useHalf = hint?.dataType != null ? hint.dataType === 'uint16' : isHalf;
  const useFloat = hint?.dataType != null ? hint.dataType === 'float32' : isFloat;
  const bpe = (src as { readonly BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT ?? 1;
  const intMax = useHalf || useFloat ? 0 : 2 ** (8 * bpe) - 1;
  const decode = (v: number): number => (
    useHalf ? halfToFloat(v) : useFloat ? v : intMax > 0 ? v / intMax : v
  );
  const needsSrgbDecode = fieldColorSpace === 'srgb' && hint?.colorSpace !== 'linear';

  const ix = Math.min(width - 1, Math.max(0, Math.floor(x)));
  const iy = Math.min(height - 1, Math.max(0, Math.floor(y)));
  const base = (iy * width + ix) * stride;
  let r = decode(Number(src[base] ?? 0));
  let g = decode(Number(src[base + (stride > 1 ? 1 : 0)] ?? 0));
  let b = decode(Number(src[base + (stride > 2 ? 2 : 0)] ?? 0));
  if (needsSrgbDecode) {
    r = srgbToLinear(r);
    g = srgbToLinear(g);
    b = srgbToLinear(b);
  }
  if (!Number.isFinite(r) || !Number.isFinite(g) || !Number.isFinite(b)) return null;
  return [r, g, b];
}

/**
 * Sample a CPU-readable `TextureRef` at a resolved UV, using the same KHR texture
 * transform + repeat/clamp/mirror wrap convention as the GPU material samplers.
 *
 * This helper intentionally uses nearest `textureLoad`-style addressing because
 * the walkaround atlas path also samples readable maps through texel loads. It is
 * used only for CPU-side emitter-power estimates; GPU shading still samples the
 * actual map payload where that backend supports it.
 */
export function sampleReadableTextureRgbAtUv(
  ref: TextureRef | undefined,
  uv: readonly [number, number],
  fieldColorSpace: 'srgb' | 'linear',
): [number, number, number] | null {
  const handle = ref?.handle as ReadableTextureHandle | null | undefined;
  if (handle == null || ref == null) return null;

  const src = handle.data ?? handle.image?.data;
  const width = Math.floor(Number(handle.width ?? handle.image?.width ?? 0));
  const height = Math.floor(Number(handle.height ?? handle.image?.height ?? 0));
  if (src == null || typeof src.length !== 'number' || width <= 0 || height <= 0) {
    return null;
  }

  const pixelCount = width * height;
  const hint = textureHint(handle);
  const heuristicStride = Math.max(1, Math.round(src.length / pixelCount));
  const stride = hint?.channels ?? heuristicStride;
  if (stride < 1 || stride > 4 || src.length < pixelCount * stride) {
    return null;
  }

  const isHalf = src instanceof Uint16Array;
  const isFloat = src instanceof Float32Array;
  const useHalf = hint?.dataType != null ? hint.dataType === 'uint16' : isHalf;
  const useFloat = hint?.dataType != null ? hint.dataType === 'float32' : isFloat;
  const bpe = (src as { readonly BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT ?? 1;
  const intMax = useHalf || useFloat ? 0 : 2 ** (8 * bpe) - 1;
  const decode = (v: number): number => (
    useHalf ? halfToFloat(v) : useFloat ? v : intMax > 0 ? v / intMax : v
  );
  const needsSrgbDecode = fieldColorSpace === 'srgb' && hint?.colorSpace !== 'linear';

  const wrapped = transformTextureUv(ref, uv);
  const x = Math.min(width - 1, Math.max(0, Math.floor(wrapped[0] * width)));
  const y = Math.min(height - 1, Math.max(0, Math.floor(wrapped[1] * height)));
  const base = (y * width + x) * stride;
  let r = decode(Number(src[base] ?? 0));
  let g = decode(Number(src[base + (stride > 1 ? 1 : 0)] ?? 0));
  let b = decode(Number(src[base + (stride > 2 ? 2 : 0)] ?? 0));
  if (needsSrgbDecode) {
    r = srgbToLinear(r);
    g = srgbToLinear(g);
    b = srgbToLinear(b);
  }
  if (!Number.isFinite(r) || !Number.isFinite(g) || !Number.isFinite(b)) return null;
  return [r, g, b];
}

export function materialSpecScalarEmissiveLe(
  material: MaterialSpec,
): [number, number, number] | null {
  const em = material.emissive;
  if (!em) return null;
  const ei = material.emissiveIntensity ?? 1;
  if (!(ei > 0)) return null;
  if (em[0] <= 0 && em[1] <= 0 && em[2] <= 0) return null;
  const out: [number, number, number] = [em[0] * ei, em[1] * ei, em[2] * ei];
  return (out[0] <= 0 && out[1] <= 0 && out[2] <= 0) ? null : out;
}

export function materialSpecEmissiveLeAtUv(
  material: MaterialSpec,
  uv0: readonly [number, number],
  uv1?: readonly [number, number],
): [number, number, number] | null {
  const scalar = materialSpecScalarEmissiveLe(material);
  if (scalar == null) return null;
  const ref = material.emissiveMap;
  if (ref == null) return scalar;
  const texCoord = Math.max(0, Math.floor(ref.texCoord ?? 0));
  const uv = texCoord === 1 && uv1 != null ? uv1 : uv0;
  const map = sampleReadableTextureRgbAtUv(ref, uv, 'srgb') ?? [1, 1, 1];
  const out: [number, number, number] = [
    scalar[0] * map[0],
    scalar[1] * map[1],
    scalar[2] * map[2],
  ];
  return (out[0] <= 0 && out[1] <= 0 && out[2] <= 0) ? null : out;
}

const TRIANGLE_EMISSIVE_QUADRATURE: readonly [number, number, number][] = [
  [1 / 3, 1 / 3, 1 / 3],
  [0.6, 0.2, 0.2],
  [0.2, 0.6, 0.2],
  [0.2, 0.2, 0.6],
  [0.5, 0.5, 0],
  [0.5, 0, 0.5],
  [0, 0.5, 0.5],
];

/**
 * Conservative subdivision level for CPU-readable emissive-map triangle lights.
 * A value of `1` means no split. Higher values split one source triangle into
 * `level²` barycentric micro-triangles with per-cell radiance estimates.
 */
export function emissiveMapTriangleSubdivisionLevel(
  material: MaterialSpec,
  maxSubdivision = 4,
): number {
  if (material.emissiveMap == null) return 1;
  const dims = readableTextureDimensions(material.emissiveMap);
  if (dims == null) return 1;
  const cappedMax = Math.max(1, Math.floor(maxSubdivision));
  const texelMajor = Math.max(dims.width, dims.height);
  if (texelMajor <= 1) return 1;
  return Math.max(2, Math.min(cappedMax, texelMajor));
}

/**
 * Visit the `level²` micro-triangles of a reference triangle in barycentric
 * coordinates. Orientation is consistent with the parent triangle.
 */
export function forEachBarycentricSubTriangle(
  level: number,
  visit: (a: BarycentricWeights, b: BarycentricWeights, c: BarycentricWeights) => void,
): void {
  const n = Math.max(1, Math.floor(level));
  const weightAt = (i: number, j: number): BarycentricWeights => {
    const u = i / n;
    const v = j / n;
    return [1 - u - v, u, v];
  };

  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n - i; j += 1) {
      const a = weightAt(i, j);
      const b = weightAt(i + 1, j);
      const c = weightAt(i, j + 1);
      visit(a, b, c);

      if (i + j < n - 1) {
        const d = weightAt(i + 1, j + 1);
        visit(b, d, c);
      }
    }
  }
}

function clipTexelPolygonHalfPlane(
  poly: readonly TexelClipVertex[],
  axis: 0 | 1,
  bound: number,
  keepGreater: boolean,
): TexelClipVertex[] {
  if (poly.length === 0) return [];
  const out: TexelClipVertex[] = [];
  const isInside = (v: TexelClipVertex): boolean => (
    keepGreater ? v.texUv[axis] >= bound - 1e-12 : v.texUv[axis] <= bound + 1e-12
  );
  const interpolate = (
    a: TexelClipVertex,
    b: TexelClipVertex,
  ): TexelClipVertex => {
    const av = a.texUv[axis];
    const bv = b.texUv[axis];
    const denom = bv - av;
    const t = Math.abs(denom) < 1e-20 ? 0 : (bound - av) / denom;
    const clamped = Math.min(1, Math.max(0, t));
    return {
      weights: [
        a.weights[0] + (b.weights[0] - a.weights[0]) * clamped,
        a.weights[1] + (b.weights[1] - a.weights[1]) * clamped,
        a.weights[2] + (b.weights[2] - a.weights[2]) * clamped,
      ],
      texUv: [
        a.texUv[0] + (b.texUv[0] - a.texUv[0]) * clamped,
        a.texUv[1] + (b.texUv[1] - a.texUv[1]) * clamped,
      ],
    };
  };

  for (let i = 0; i < poly.length; i += 1) {
    const current = poly[i]!;
    const previous = poly[(i + poly.length - 1) % poly.length]!;
    const currentInside = isInside(current);
    const previousInside = isInside(previous);
    if (currentInside !== previousInside) {
      out.push(interpolate(previous, current));
    }
    if (currentInside) out.push(current);
  }
  return out;
}

function clipTexelPolygonToCell(
  poly: readonly TexelClipVertex[],
  x0: number,
  x1: number,
  y0: number,
  y1: number,
): TexelClipVertex[] {
  let clipped = clipTexelPolygonHalfPlane(poly, 0, x0, true);
  clipped = clipTexelPolygonHalfPlane(clipped, 0, x1, false);
  clipped = clipTexelPolygonHalfPlane(clipped, 1, y0, true);
  clipped = clipTexelPolygonHalfPlane(clipped, 1, y1, false);
  return clipped.length >= 3 ? clipped : [];
}

function buildTextureCellIntervals(
  minValue: number,
  maxValue: number,
  texelCount: number,
  mode: TextureRef['wrapS'],
): TextureCellInterval[] | null {
  if (texelCount <= 0 || !Number.isFinite(minValue) || !Number.isFinite(maxValue)) return null;
  if (mode === 'mirrored-repeat') return null;
  if (texelCount === 1) {
    return [{ lo: minValue, hi: maxValue, texel: 0 }];
  }

  if (mode === 'clamp-to-edge') {
    const out: TextureCellInterval[] = [];
    for (let texel = 0; texel < texelCount; texel += 1) {
      const lo = texel === 0 ? minValue : texel / texelCount;
      const hi = texel === texelCount - 1 ? maxValue : (texel + 1) / texelCount;
      const clippedLo = Math.max(lo, minValue);
      const clippedHi = Math.min(hi, maxValue);
      if (clippedHi > clippedLo + 1e-12) out.push({ lo: clippedLo, hi: clippedHi, texel });
    }
    return out;
  }

  const first = Math.floor(minValue * texelCount);
  const last = Math.floor(maxValue * texelCount);
  const out: TextureCellInterval[] = [];
  for (let cell = first; cell <= last; cell += 1) {
    const texel = ((cell % texelCount) + texelCount) % texelCount;
    const lo = Math.max(cell / texelCount, minValue);
    const hi = Math.min((cell + 1) / texelCount, maxValue);
    if (hi > lo + 1e-12) out.push({ lo, hi, texel });
  }
  return out;
}

function texUvArea2(
  a: readonly [number, number],
  b: readonly [number, number],
  c: readonly [number, number],
): number {
  return Math.abs(
    (b[0] - a[0]) * (c[1] - a[1]) -
    (b[1] - a[1]) * (c[0] - a[0]),
  );
}

/**
 * Splits a CPU-readable emissive-map triangle along exact texel-cell boundaries
 * in transformed texture space and visits constant-radiance barycentric
 * sub-triangles. This gives emitter CDF/PDF construction a texel-space support
 * instead of the older fixed quadrature estimate.
 *
 * Returns `false` when an exact split is not representable or bounded here
 * (unreadable map, mirrored repeat, degenerate UVs, or too many covered cells);
 * callers should then fall back to the conservative barycentric subdivision.
 * Returns `true` when the texture was handled, even if all covered texels were
 * black and no sub-triangles were emitted.
 */
export function forEachEmissiveMapTexelSubTriangle(
  material: MaterialSpec,
  uv0A: readonly [number, number],
  uv0B: readonly [number, number],
  uv0C: readonly [number, number],
  uv1A: readonly [number, number] | undefined,
  uv1B: readonly [number, number] | undefined,
  uv1C: readonly [number, number] | undefined,
  visit: (
    a: BarycentricWeights,
    b: BarycentricWeights,
    c: BarycentricWeights,
    radiance: readonly [number, number, number],
    texelX: number,
    texelY: number,
    ordinal: number,
  ) => void,
  maxCoveredCells = 4096,
): boolean {
  const scalar = materialSpecScalarEmissiveLe(material);
  const ref = material.emissiveMap;
  if (scalar == null || ref == null) return false;
  const dims = readableTextureDimensions(ref);
  if (dims == null) return false;

  const texCoord = Math.max(0, Math.floor(ref.texCoord ?? 0));
  const useUv1 = texCoord === 1 && uv1A != null && uv1B != null && uv1C != null;
  const srcA = useUv1 ? uv1A : uv0A;
  const srcB = useUv1 ? uv1B : uv0B;
  const srcC = useUv1 ? uv1C : uv0C;
  const texA = transformTextureUvUnwrapped(ref, srcA);
  const texB = transformTextureUvUnwrapped(ref, srcB);
  const texC = transformTextureUvUnwrapped(ref, srcC);
  if (texUvArea2(texA, texB, texC) < 1e-14) return false;

  const minX = Math.min(texA[0], texB[0], texC[0]);
  const maxX = Math.max(texA[0], texB[0], texC[0]);
  const minY = Math.min(texA[1], texB[1], texC[1]);
  const maxY = Math.max(texA[1], texB[1], texC[1]);
  const xIntervals = buildTextureCellIntervals(minX, maxX, dims.width, ref.wrapS);
  const yIntervals = buildTextureCellIntervals(minY, maxY, dims.height, ref.wrapT);
  if (xIntervals == null || yIntervals == null) return false;
  if (xIntervals.length * yIntervals.length > maxCoveredCells) return false;

  const initial: TexelClipVertex[] = [
    { weights: [1, 0, 0], texUv: texA },
    { weights: [0, 1, 0], texUv: texB },
    { weights: [0, 0, 1], texUv: texC },
  ];

  let ordinal = 0;
  for (const xi of xIntervals) {
    for (const yi of yIntervals) {
      const clipped = clipTexelPolygonToCell(initial, xi.lo, xi.hi, yi.lo, yi.hi);
      if (clipped.length < 3) continue;
      const texelRgb = readTextureRgbAtTexel(ref, xi.texel, yi.texel, 'srgb');
      if (texelRgb == null) return false;
      const radiance: [number, number, number] = [
        scalar[0] * texelRgb[0],
        scalar[1] * texelRgb[1],
        scalar[2] * texelRgb[2],
      ];
      if (radiance[0] <= 0 && radiance[1] <= 0 && radiance[2] <= 0) continue;
      const anchor = clipped[0]!;
      for (let i = 1; i + 1 < clipped.length; i += 1) {
        const b = clipped[i]!;
        const c = clipped[i + 1]!;
        if (texUvArea2(anchor.texUv, b.texUv, c.texUv) < 1e-16) continue;
        visit(anchor.weights, b.weights, c.weights, radiance, xi.texel, yi.texel, ordinal);
        ordinal += 1;
      }
    }
  }
  return true;
}

function baryUv(
  a: readonly [number, number],
  b: readonly [number, number],
  c: readonly [number, number],
  w: readonly [number, number, number],
): [number, number] {
  return [
    a[0] * w[0] + b[0] * w[1] + c[0] * w[2],
    a[1] * w[0] + b[1] * w[1] + c[1] * w[2],
  ];
}

/**
 * Deterministic UV-local emissive estimate for one triangle. This is not a full
 * texel alias table, but it prevents large UV-varying emissive maps from being
 * selected solely by a whole-material average when building CPU emitter CDFs.
 */
export function estimateMaterialSpecEmissiveLeOverTriangle(
  material: MaterialSpec,
  uv0A: readonly [number, number],
  uv0B: readonly [number, number],
  uv0C: readonly [number, number],
  uv1A?: readonly [number, number],
  uv1B?: readonly [number, number],
  uv1C?: readonly [number, number],
): [number, number, number] | null {
  const scalar = materialSpecScalarEmissiveLe(material);
  if (scalar == null) return null;
  if (material.emissiveMap == null) return scalar;

  let r = 0;
  let g = 0;
  let b = 0;
  for (const weights of TRIANGLE_EMISSIVE_QUADRATURE) {
    const uv0 = baryUv(uv0A, uv0B, uv0C, weights);
    const uv1 = uv1A != null && uv1B != null && uv1C != null
      ? baryUv(uv1A, uv1B, uv1C, weights)
      : undefined;
    const le = materialSpecEmissiveLeAtUv(material, uv0, uv1);
    r += le?.[0] ?? 0;
    g += le?.[1] ?? 0;
    b += le?.[2] ?? 0;
  }
  const inv = 1 / TRIANGLE_EMISSIVE_QUADRATURE.length;
  const out: [number, number, number] = [r * inv, g * inv, b * inv];
  return (out[0] <= 0 && out[1] <= 0 && out[2] <= 0) ? null : out;
}

/**
 * Emissive radiance Le (`emissive.rgb · emissiveIntensity · emissiveMap`) of a core
 * `MaterialSpec`, or `null` when the surface is not self-emissive.
 *
 * Uses the same reject conditions as the rest of the core emitter pipeline
 * (absent emissive, non-positive intensity, all-non-positive final channels),
 * so the camera-visible glow Le and the NEE-sampled emitter radiance share one
 * source. When `emissiveMap` exposes readable CPU pixels, the average linear RGB
 * map value modulates Le; opaque/GPU-only handles fall back to the scalar factor.
 * Deliberately EXCLUDES the transmissive "sun-attenuated secondary emitter"
 * branch — that lives in {@link classifyTriangleEmitterCore}.
 * A missing `emissiveIntensity` defaults to ×1, matching the core material
 * entry adapter and the path-tracing backends.
 *
 * @param material a core `MaterialSpec`.
 * @returns `[r, g, b]` HDR radiance, or `null` for a non-emissive surface.
 */
export function materialSpecEmissiveLe(
  material: MaterialSpec,
): [number, number, number] | null {
  const scalar = materialSpecScalarEmissiveLe(material);
  if (scalar == null) return null;
  const map = averageReadableTextureRgb(material.emissiveMap, 'srgb') ?? [1, 1, 1];
  const out: [number, number, number] = [
    scalar[0] * map[0],
    scalar[1] * map[1],
    scalar[2] * map[2],
  ];
  if (out[0] <= 0 && out[1] <= 0 && out[2] <= 0) return null;
  return out;
}

/**
 * Apply RGB Beer-Lambert absorption to a tint color given a sample
 * thickness / attenuation-distance pair: `c' = c^(thickness/attDist)`
 * (per channel, with a 1e-6 floor). Returns the input color unchanged when any
 * required parameter is missing / non-finite / non-positive.
 *
 * Tuple in/out helper used by per-triangle color and Beer-lane packing.
 */
export function applyBeerLambertColor(
  attCol: readonly [number, number, number],
  thickness: number | undefined,
  attDist: number | undefined,
): [number, number, number] {
  if (thickness === undefined || attDist === undefined) {
    return [attCol[0], attCol[1], attCol[2]];
  }
  if (!Number.isFinite(thickness) || !Number.isFinite(attDist)) {
    return [attCol[0], attCol[1], attCol[2]];
  }
  if (thickness <= 0 || attDist <= 0) {
    return [attCol[0], attCol[1], attCol[2]];
  }
  const k = thickness / attDist;
  return [
    Math.pow(Math.max(1e-6, attCol[0]), k),
    Math.pow(Math.max(1e-6, attCol[1]), k),
    Math.pow(Math.max(1e-6, attCol[2]), k),
  ];
}

/**
 * Resolve a triangle's visible RGB color from a core `MaterialSpec`: the
 * attenuation color (optionally Beer-Lambert-tinted) for a transmissive surface,
 * else the base color, else the warm-gray fallback.
 *
 * Core triangle-color resolver:
 *  - `isTransmissive` ⇔ `transmission > {@link MATERIAL_TRANSMISSIVE_COLOR_THRESHOLD}`
 *    (0.01).
 *  - transmissive → the attenuation color ({@link applyBeerLambertColor}-tinted
 *    iff `applyBeer`).
 *  - otherwise → `baseColor`, falling back to {@link MATERIAL_DEFAULT_TRI_COLOR}.
 *
 * A transmissive material with no explicit `attenuationColor` is treated as
 * white `(1,1,1)`, and an absent `attenuationDistance` behaves like Infinity
 * (→ {@link applyBeerLambertColor} passthrough). A core `MaterialSpec`'s
 * `baseColor` is required and non-null, so the warm-gray fallback only fires for
 * the no-material case (`packBVH*Tri` passes the literal default when
 * `materials[matId]` is missing) or for defensively-empty loose inputs.
 *
 * @param material  a core `MaterialSpec`.
 * @param applyBeer when true, Beer-Lambert-tint the transmissive attenuation
 *                  color (the `bvh_beer` lane); when false, use it raw (the
 *                  `bvhIndex.w` RGBA8 lane).
 */
export function materialSpecTriColor(
  material: MaterialSpec,
  applyBeer: boolean,
): [number, number, number] {
  const transmission = material.transmission ?? 0;
  const isTransmissive = transmission > MATERIAL_TRANSMISSIVE_COLOR_THRESHOLD;
  if (isTransmissive) {
    // The transmissive branch always uses attenuation color (never baseColor).
    // Missing attenuation data maps to no-tint / no-falloff defaults.
    const attenColor = material.attenuationColor ?? [1, 1, 1];
    if (applyBeer) {
      return applyBeerLambertColor(
        attenColor,
        material.thickness,
        material.attenuationDistance, // undefined → Infinity-equivalent passthrough
      );
    }
    return [attenColor[0], attenColor[1], attenColor[2]];
  }
  // `MaterialSpec.baseColor` is a required Vec3, so the per-material path always
  // has a base color. The guard is a defensive runtime check for loosely-typed
  // callers and keeps the warm-gray fallback reachable.
  const base = material.baseColor;
  if (Array.isArray(base) && base.length >= 3) return [base[0], base[1], base[2]];
  return [
    MATERIAL_DEFAULT_TRI_COLOR[0],
    MATERIAL_DEFAULT_TRI_COLOR[1],
    MATERIAL_DEFAULT_TRI_COLOR[2],
  ];
}

/**
 * Read the surface-texture id (the `bvhIndex.w` low-byte `texType` lane, 3 bits)
 * from a core `MaterialSpec`'s `extensions['surfaceTextureId']`.
 *
 * Returns `0` when absent / non-numeric.
 */
export function materialSpecSurfaceTextureId(material: MaterialSpec): number {
  const raw = material.extensions?.['surfaceTextureId'];
  return (typeof raw === 'number' ? raw : 0) & 0x7;
}

/**
 * Read the `skipEmitter` override from a core `MaterialSpec`'s
 * `extensions['skipEmitter']`. Strict `=== true` (any other value, including
 * absent, means "do not skip").
 */
export function materialSpecSkipEmitter(material: MaterialSpec): boolean {
  return material.extensions?.['skipEmitter'] === true;
}

/**
 * Classify a core `MaterialSpec` + face normal as a ReSTIR-DI emitter, or `null`
 * when the face is not selected. Priority order:
 *
 *  1. **Emissive** (`emissive.rgb · emissiveIntensity` positive) → direct
 *     emitter with `color = Le`, `intensity = emissiveIntensity` (default 1).
 *     Shares {@link materialSpecEmissiveLe} with the camera-glow packer so the
 *     NEE radiance and the camera glow Le are identical.
 *  2. **Transmissive** (`transmission > {@link MATERIAL_EMITTER_TRANSMISSION_THRESHOLD}`
 *     (0.1), not `skipEmitter`, `|dot(lightDir, normal)| > {@link
 *     MATERIAL_EMITTER_SUN_DOT_THRESHOLD}` (0.05)) → sun-attenuated secondary
 *     emitter:
 *       `color   = baseColor ⊙ attenuationColor · transmission · primaryIntensity · sunDot`
 *       `intensity = primaryIntensity · transmission · sunDot`
 *     `baseColor` / `attenuationColor` default to (1,1,1) when absent.
 *  3. otherwise → `null` (skipped).
 *
 * `lightDir` is the configured primary-light direction; `primaryIntensity` is
 * its irradiance — both passed as plain numbers/tuples. The caller computes
 * power (`luminance(color) · area`) and the < 1e-8 drop.
 *
 * @param material         a core `MaterialSpec`.
 * @param normal           the face normal (world-space, unit length).
 * @param lightDir         the primary-light direction (world-space, unit length).
 * @param primaryIntensity the primary-light irradiance.
 * @returns `{ color, intensity }` for a selected emitter, else `null`.
 */
export function classifyTriangleEmitterCore(
  material: MaterialSpec,
  normal: { x: number; y: number; z: number },
  lightDir: { x: number; y: number; z: number },
  primaryIntensity: number,
): { color: [number, number, number]; intensity: number } | null {
  // 1. Emissive surface → direct emitter (shares the camera-glow Le source).
  const emissiveLe = materialSpecEmissiveLe(material);
  if (emissiveLe != null) {
    return { color: emissiveLe, intensity: material.emissiveIntensity ?? 1 };
  }

  // 2. Transmissive → sun-attenuated secondary emitter.
  const trans = material.transmission ?? 0;
  if (trans <= MATERIAL_EMITTER_TRANSMISSION_THRESHOLD) return null;
  if (materialSpecSkipEmitter(material)) return null;

  const sunDot = Math.abs(
    lightDir.x * normal.x + lightDir.y * normal.y + lightDir.z * normal.z,
  );
  if (sunDot <= MATERIAL_EMITTER_SUN_DOT_THRESHOLD) return null;

  const baseColor = material.baseColor ?? [1, 1, 1];
  const attenColor = material.attenuationColor ?? [1, 1, 1];
  return {
    color: [
      baseColor[0] * attenColor[0] * trans * primaryIntensity * sunDot,
      baseColor[1] * attenColor[1] * trans * primaryIntensity * sunDot,
      baseColor[2] * attenColor[2] * trans * primaryIntensity * sunDot,
    ],
    intensity: primaryIntensity * trans * sunDot,
  };
}
