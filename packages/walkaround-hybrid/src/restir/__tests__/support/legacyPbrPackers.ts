/**
 * Legacy structural-PBR (`PbrMaterialLike`) per-triangle packers — TEST ORACLES.
 *
 * These `@deprecated` packers predate the core-`MaterialSpec` `*FromCore`
 * production family in `restir/packingHelpers.ts`. They are retained as
 * INDEPENDENT reference oracles: `roughMetalPacking.test.ts` /
 * `emissiveLePacking.test.ts` assert that the production `*FromCore` packers
 * produce byte-identical output to these structural counterparts. Because they
 * have ZERO production consumers (verified by grep, 2026-07-20), they were
 * relocated here out of the production module (D6-4 / I3) so `restir/` no longer
 * ships legacy test-only code.
 *
 * The genuinely-shared pure helpers (`resolveRoughMetal`, `packRoughMetalIorBytes`,
 * the warm-gray missing-material default, and the IOR constants) still live in
 * `restir/packingHelpers.ts` — the production `*FromCore` packers use them too —
 * and are imported here so there is a single source of truth for the shared math.
 */

import {
  WARM_GRAY_DEFAULT_R,
  WARM_GRAY_DEFAULT_G,
  WARM_GRAY_DEFAULT_B,
  ROUGH_DEFAULT,
  IOR_RANGE_MIN,
  resolveRoughMetal,
  packRoughMetalIorBytes,
} from '../../packingHelpers.js';

interface ColorLike {
  readonly r: number;
  readonly g: number;
  readonly b: number;
}

export interface PbrMaterialLike {
  readonly color?: ColorLike;
  readonly emissive?: ColorLike;
  readonly emissiveIntensity?: number;
  readonly roughness?: number;
  readonly metalness?: number;
  readonly transmission?: number;
  readonly ior?: number;
  readonly attenuationColor?: ColorLike;
  readonly attenuationDistance?: number;
  readonly thickness?: number;
  readonly userData?: {
    readonly surfaceTextureId?: number;
  };
}

/**
 * Apply Beer-Lambert absorption to an attenuation color given a sample
 * thickness / attenuation-distance pair. Returns the input color unchanged
 * if any required parameter is missing or non-finite.
 */
function applyBeerLambert(
  attCol: ColorLike,
  thickness: number | undefined,
  attDist: number | undefined,
): ColorLike {
  if (thickness === undefined || attDist === undefined) return attCol;
  if (!Number.isFinite(thickness) || !Number.isFinite(attDist)) return attCol;
  if (thickness <= 0 || attDist <= 0) return attCol;
  const k = thickness / attDist;
  return {
    r: Math.pow(Math.max(1e-6, attCol.r), k),
    g: Math.pow(Math.max(1e-6, attCol.g), k),
    b: Math.pow(Math.max(1e-6, attCol.b), k),
  };
}

/**
 * Resolve a triangle's RGB color for packing. Shared between packBVHIndexW
 * (raw attenuation color) and packBVHBeerColors (Beer-Lambert tinted).
 */
function resolveTriColor(mat: PbrMaterialLike, applyBeer: boolean): ColorLike {
  const transmission = (mat.transmission ?? 0);
  const isTransmissive = transmission > 0.01;
  const attenColor = mat.attenuationColor;
  if (isTransmissive && attenColor) {
    if (applyBeer) {
      return applyBeerLambert(
        attenColor,
        mat.thickness,
        mat.attenuationDistance,
      );
    }
    return attenColor;
  }
  return mat.color ?? { r: 0.6, g: 0.58, b: 0.55 };
}

/**
 * Pack one triangle's index lanes + material byte into an existing vec4u buffer.
 * @deprecated Superseded by `packBVHIndexWFromCore` (core `MaterialSpec[]`).
 */
function packBVHIndexWTri(
  indexBuf: Uint32Array,
  indices: Uint32Array,
  triMaterialId: Uint32Array,
  materials: readonly PbrMaterialLike[],
  tri: number,
): void {
  const base4 = tri * 4;
  indexBuf[base4 + 0] = indices[tri * 3 + 0]!;
  indexBuf[base4 + 1] = indices[tri * 3 + 1]!;
  indexBuf[base4 + 2] = indices[tri * 3 + 2]!;

  const matId = triMaterialId[tri]!;
  const mat = materials[matId];
  let r = WARM_GRAY_DEFAULT_R, g = WARM_GRAY_DEFAULT_G, b = WARM_GRAY_DEFAULT_B;
  let transmission = 0;
  let texTypeId = 0;
  let isMetal = 0;
  if (mat) {
    transmission = (mat.transmission ?? 0);
    const color = resolveTriColor(mat, /* applyBeer */ false);
    r = Math.round(color.r * 255) & 0xFF;
    g = Math.round(color.g * 255) & 0xFF;
    b = Math.round(color.b * 255) & 0xFF;
    const surfTex = mat.userData?.surfaceTextureId;
    texTypeId = (typeof surfTex === 'number' ? surfTex : 0) & 0x7;
    const metalness = (mat.metalness ?? 0);
    isMetal = metalness > 1e-4 ? 1 : 0;
  }
  const trans4 = Math.min(15, Math.round(transmission * 15)) & 0xF;
  const lowByte = ((trans4 << 4) | (isMetal << 3) | (texTypeId & 0x7)) & 0xFF;
  indexBuf[base4 + 3] = (r << 24) | (g << 16) | (b << 8) | lowByte;
}

/**
 * Pack one triangle's roughness+metalness+IOR into a parallel u32 buffer.
 * @deprecated Superseded by `packBVHRoughMetalFromCore` (core `MaterialSpec[]`).
 */
function packBVHRoughMetalTri(
  rmBuf: Uint32Array,
  triMaterialId: Uint32Array,
  materials: readonly PbrMaterialLike[],
  tri: number,
): void {
  const matId = triMaterialId[tri]!;
  const mat = materials[matId];
  const rm = mat
    ? resolveRoughMetal(mat.roughness, mat.metalness, mat.transmission, mat.ior)
    : { rough: ROUGH_DEFAULT, metal: 0, ior: IOR_RANGE_MIN };
  rmBuf[tri] = packRoughMetalIorBytes(rm.rough, rm.metal, rm.ior);
}

/**
 * Pack one triangle's Beer-Lambert visible color into a parallel u32 buffer.
 * @deprecated Superseded by `packBVHBeerColorsFromCore` (core `MaterialSpec[]`).
 */
function packBVHBeerColorTri(
  beerBuf: Uint32Array,
  triMaterialId: Uint32Array,
  materials: readonly PbrMaterialLike[],
  tri: number,
): void {
  const matId = triMaterialId[tri]!;
  const mat = materials[matId];
  let r = WARM_GRAY_DEFAULT_R, g = WARM_GRAY_DEFAULT_G, b = WARM_GRAY_DEFAULT_B;
  if (mat) {
    const color = resolveTriColor(mat, /* applyBeer */ true);
    r = Math.round(Math.min(1, color.r) * 255) & 0xFF;
    g = Math.round(Math.min(1, color.g) * 255) & 0xFF;
    b = Math.round(Math.min(1, color.b) * 255) & 0xFF;
  }
  beerBuf[tri] = (r << 24) | (g << 16) | (b << 8);
}

/**
 * Re-pack `bvhIndex.w` and `bvh_beer` for a contiguous triangle subrange.
 * @deprecated Structural `PbrMaterialLike`-based family; legacy adapter.
 */
export function repackBVHMaterialRange(
  indexBuf: Uint32Array,
  beerBuf: Uint32Array,
  indices: Uint32Array,
  triMaterialId: Uint32Array,
  materials: readonly PbrMaterialLike[],
  triStart: number,
  triCount: number,
  rmBuf?: Uint32Array,
): void {
  const triEnd = triStart + triCount;
  for (let t = triStart; t < triEnd; t++) {
    packBVHIndexWTri(indexBuf, indices, triMaterialId, materials, t);
    packBVHBeerColorTri(beerBuf, triMaterialId, materials, t);
    if (rmBuf) packBVHRoughMetalTri(rmBuf, triMaterialId, materials, t);
  }
}

/**
 * Pack vertex indices + RGBA8 baseColor + (trans4|texType4) into vec4u
 * per-triangle (4 u32 = 16 bytes per triangle).
 * @deprecated Superseded by `packBVHIndexWFromCore` (core `MaterialSpec[]`).
 */
export function packBVHIndexW(
  indices: Uint32Array,
  triMaterialId: Uint32Array,
  materials: readonly PbrMaterialLike[],
  triCount: number,
): Uint32Array<ArrayBuffer> {
  const indexBuf = new Uint32Array(triCount * 4);
  for (let t = 0; t < triCount; t++) {
    packBVHIndexWTri(indexBuf, indices, triMaterialId, materials, t);
  }
  return indexBuf;
}

/**
 * Pack per-triangle roughness+metalness+IOR into a parallel u32 buffer
 * (bits[31:24]=rough×255, bits[23:16]=metal×255, bits[15:8]=ior_quantized).
 * @deprecated Superseded by `packBVHRoughMetalFromCore` (core `MaterialSpec[]`).
 */
export function packBVHRoughMetal(
  triMaterialId: Uint32Array,
  materials: readonly PbrMaterialLike[],
  triCount: number,
): Uint32Array<ArrayBuffer> {
  const rmBuf = new Uint32Array(triCount);
  for (let t = 0; t < triCount; t++) {
    packBVHRoughMetalTri(rmBuf, triMaterialId, materials, t);
  }
  return rmBuf;
}

/**
 * Pack the Beer-Lambert visible color per triangle into a parallel u32 buffer.
 * @deprecated Superseded by `packBVHBeerColorsFromCore` (core `MaterialSpec[]`).
 */
export function packBVHBeerColors(
  triMaterialId: Uint32Array,
  materials: readonly PbrMaterialLike[],
  triCount: number,
): Uint32Array<ArrayBuffer> {
  const beerBuf = new Uint32Array(triCount);
  for (let t = 0; t < triCount; t++) {
    packBVHBeerColorTri(beerBuf, triMaterialId, materials, t);
  }
  return beerBuf;
}

/**
 * Emissive radiance Le (HDR, `emissive.rgb * emissiveIntensity`) of a material,
 * or `null` when the material is not a self-emissive surface. Mirrors the
 * EMISSIVE branch of `classifyTriangleEmitter` (emitterList.ts) EXACTLY but
 * deliberately EXCLUDES the transmissive "sun-attenuated secondary emitter"
 * branch (`lo_emit` already handles glass self-emission — packing it here would
 * double-count).
 * @deprecated Structural `PbrMaterialLike` family; production code uses
 *   `materialSpecEmissiveLe` from `@vitrum/shared-bvh` via the `*FromCore` packers.
 */
export function materialEmissiveLe(mat: PbrMaterialLike): [number, number, number] | null {
  const em = mat.emissive;
  if (!em) return null;
  const ei = mat.emissiveIntensity;
  if (!(ei && ei > 0)) return null;
  if (em.r <= 0 && em.g <= 0 && em.b <= 0) return null;
  return [em.r * ei, em.g * ei, em.b * ei];
}

/**
 * Pack per-triangle emissive radiance Le into a parallel rgba32float buffer
 * (stride 4: rgb + 0 pad). Non-emissive triangles are zero.
 * @deprecated Superseded by `packBVHEmissiveLeFromCore` (core `MaterialSpec[]`).
 */
export function packBVHEmissiveLe(
  triMaterialId: Uint32Array,
  materials: readonly PbrMaterialLike[],
  triCount: number,
): Float32Array<ArrayBuffer> {
  const out = new Float32Array(triCount * 4);
  for (let t = 0; t < triCount; t++) {
    const mat = materials[triMaterialId[t]!];
    if (!mat) continue;
    const le = materialEmissiveLe(mat);
    if (le == null) continue;
    out[t * 4 + 0] = le[0];
    out[t * 4 + 1] = le[1];
    out[t * 4 + 2] = le[2];
  }
  return out;
}
