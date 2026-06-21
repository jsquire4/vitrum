/**
 * Per-engine vertex / index / color packing helpers for ReSTIR.
 *
 * Sits on top of `@vitrum/shared-bvh`'s `buildSceneBVH` shared-core output
 * and re-packs the layouts ReSTIR's WGSL shaders consume:
 *   - position.w slot ← packed raw UV pair
 *   - bvhIndex.w slot ← RGBA8 baseColor | (trans4 | texType4)
 *   - bvh_beer    u32 ← Beer-Lambert visible color per triangle
 *
 * Extracted from the legacy `legacy/three/restirBvhCompute.ts` mixed builder
 * (was ~270 lines of inline packing).
 */

import type { MaterialSpec } from '@vitrum/core';
import {
  materialSpecTriColor,
  materialSpecSurfaceTextureId,
} from '@vitrum/shared-bvh';

interface ColorLike {
  readonly r: number;
  readonly g: number;
  readonly b: number;
}

export interface BufferAttributeLike {
  readonly array: ArrayLike<number>;
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

/** Default warm-gray fallback color (sRGB byte values) when a triangle has
 *  no material or unrecognised material type. Matches the old in-file
 *  `WARM_GRAY_DEFAULT_*` constants. File-local — no external consumers
 *  (2026-05-18 dead-code sweep). */
const WARM_GRAY_DEFAULT_R = 153;
const WARM_GRAY_DEFAULT_G = 148;
const WARM_GRAY_DEFAULT_B = 140;

/**
 * Apply Beer-Lambert absorption to an attenuation color given a sample
 * thickness / attenuation-distance pair. Returns the input color unchanged
 * if any required parameter is missing or non-finite.
 *
 * File-local — no external consumers (2026-05-18 dead-code sweep).
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
 * Pack UV (two f16 values) into the .w slot of every vec4f position.
 * See the ReSTIR BVH builders for the rationale (single storage buffer per stage).
 */
export function packUVIntoPositionW(
  positions: Float32Array,
  uvAttr: BufferAttributeLike | undefined,
  vertCount: number,
): Float32Array<ArrayBuffer> {
  return packUVIntoVec4W(positions, uvAttr, vertCount);
}

/**
 * Pack UV (two f16 values) into the .w slot of a vec4f-strided stream.
 * The xyz lanes are preserved verbatim. Used for position.w (uv0, consumed by
 * traversal) and normal.w (uv1, consumed by material texture sampling).
 */
export function packUVIntoVec4W(
  values: Float32Array,
  uvAttr: BufferAttributeLike | undefined,
  vertCount: number,
): Float32Array<ArrayBuffer> {
  const out = new Float32Array(values.length);
  out.set(values);
  const u32View = new Uint32Array(out.buffer);
  const sourceUvs = uvAttr?.array;

  for (let i = 0; i < vertCount; i++) {
    const u16 = floatToHalfBits(sourceUvs?.[i * 2 + 0] ?? 0);
    const v16 = floatToHalfBits(sourceUvs?.[i * 2 + 1] ?? 0);
    u32View[i * 4 + 3] = (v16 << 16) | u16;
  }
  return out;
}

function floatToHalfBits(value: number): number {
  const input = Number.isFinite(value) ? Math.fround(value) : 0;
  const sign = input < 0 || Object.is(input, -0) ? 0x8000 : 0;
  const abs = Math.abs(input);
  if (abs === 0) return sign;
  if (abs >= 65504) return sign | 0x7bff;
  if (abs < 2 ** -24) return sign;
  if (abs < 2 ** -14) {
    return sign | Math.min(0x03ff, Math.round(abs / (2 ** -24)));
  }

  let exp = Math.floor(Math.log2(abs));
  let mant = Math.round((abs / (2 ** exp) - 1) * 1024);
  if (mant === 1024) {
    mant = 0;
    exp += 1;
  }
  const halfExp = exp + 15;
  if (halfExp >= 31) return sign | 0x7bff;
  return sign | ((halfExp & 0x1f) << 10) | (mant & 0x03ff);
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
 * @deprecated Superseded by {@link packBVHIndexWFromCore} (core `MaterialSpec[]`).
 *   The structural `PbrMaterialLike`-based family is legacy; production code uses
 *   the `*FromCore` counterparts. Retained for tests and legacy adapters only.
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

// ──────────────────────────────────────────────────────────────────────────
// B1 — per-triangle roughness + metalness lane.
// B1-ior-per-tri — per-triangle IOR lane (2026-06-10).
//
// The bvhIndex.w payload (RGBA8 baseColor | trans4 | isMetal1 | texId3) is full;
// authored roughness/metalness never reached the BRDF (shade hardcoded
// rough = select(0.85, 0.05, isGlass), metal = 0). B1 (road-to-100) needs the
// real per-triangle values so the GGX BRDF in lo_direct / lo_analyticNEE and the
// glossy/metal GI target are physically driven.
//
// Layout: one u32 per triangle (parallel to bvh_beer / bvh_emissive, same texel
// addressing):
//   bits[31:24] = roughness × 255
//   bits[23:16] = metalness × 255
//   bits[15:8]  = IOR quantized: (ior − 1) / 2 * 255, range [1.0, 3.0]
//                 covering water (1.33) → glass (1.5) → diamond (2.42) → TiO₂ (≈2.9).
//                 Decode: ior = 1 + (byte / 255) * 2.  Quantization step = 2/255 ≈ 0.0078.
//                 IOR_GLASS = 1.5 encodes to byte 63.75 → rounds to 64 → decodes to
//                 1 + 64/255 * 2 ≈ 1.502 (error < 0.003 — within glass dispersion spread).
//   bits[7:0]   = material flags + low-precision scalar sidecar:
//                 bit 0 = castShadowDisabled
//                 bit 1 = unlit shading model
//                 bit 2 = scalarAlphaDiscarded
//                 bits 3-7 = aoMapIntensity × 31 (glTF occlusion strength).
//
// DIFFUSE DEFAULT INVARIANT (B1): a material with no authored roughness packs
// ROUGH_DEFAULT = 0.85 — the EXACT value shade/ris/cast hardcoded for non-glass
// before B1 — so a default-roughness diffuse scene (metal 0) is numerically
// unchanged. Glass (transmission > 0.5) packs 0.05 to match the prior glass
// hardcode. Metalness 0 default.
//
// IOR DEFAULT INVARIANT (B1-ior-per-tri): glass materials with no authored ior
// default to IOR_DEFAULT_GLASS = 1.5, which encodes to byte 64 and decodes
// back to 1.502 — preserving the previous hard-wired IOR_GLASS=1.5 behaviour
// to within 0.003. Opaque/non-glass materials pack IOR_DEFAULT_OPAQUE = 1.0
// (byte 0) — decodes to exactly 1.0 (air/vacuum); the WGSL consumers skip the
// IOR lane for non-glass surfaces so this value has no visual impact.
const ROUGH_DEFAULT = 0.85;
const ROUGH_GLASS = 0.05;
/** Default IOR for glass (crown glass). Packs to byte 64, decodes to 1.502. */
export const IOR_DEFAULT_GLASS = 1.5;
/** IOR range minimum (air/vacuum). Byte 0. */
export const IOR_RANGE_MIN = 1.0;
/** IOR range maximum (slightly above diamond 2.42, up to TiO₂ ≈ 2.9). */
export const IOR_RANGE_MAX = 3.0;

/** Quantize an IOR value to a u8 byte using the [IOR_RANGE_MIN, IOR_RANGE_MAX] linear mapping.
 *  Maps ior → clamp((ior − 1) / 2 * 255, 0, 255). */
export function quantizeIor(ior: number): number {
  return Math.min(255, Math.max(0, Math.round((ior - IOR_RANGE_MIN) / (IOR_RANGE_MAX - IOR_RANGE_MIN) * 255))) & 0xFF;
}

/** Dequantize a u8 IOR byte back to a float. Inverse of {@link quantizeIor}. */
export function dequantizeIor(byte: number): number {
  return IOR_RANGE_MIN + (byte / 255) * (IOR_RANGE_MAX - IOR_RANGE_MIN);
}

const BVH_MATERIAL_CAST_SHADOW_DISABLED_BIT = 1 << 0;
const BVH_MATERIAL_UNLIT_BIT = 1 << 1;
const BVH_MATERIAL_SCALAR_ALPHA_DISCARDED_BIT = 1 << 2;

function quantizeAoMapIntensity(value: number | undefined): number {
  const strength = Number.isFinite(value) ? Math.min(1, Math.max(0, value ?? 1)) : 1;
  return (Math.round(strength * 31) & 0x1F) << 3;
}

function packRoughMetalIorBytes(roughness: number, metalness: number, ior: number): number {
  const r8 = Math.min(255, Math.max(0, Math.round(roughness * 255))) & 0xFF;
  const m8 = Math.min(255, Math.max(0, Math.round(metalness * 255))) & 0xFF;
  const i8 = quantizeIor(ior);
  return ((r8 << 24) | (m8 << 16) | (i8 << 8)) >>> 0;
}

function scalarAlphaDiscarded(mat: MaterialSpec): boolean {
  const mode = mat.alphaMode ?? 'opaque';
  if (mode === 'opaque') return false;
  const opacity = Number.isFinite(mat.opacity) ? Math.min(1, Math.max(0, mat.opacity ?? 1)) : 1;
  if (mode === 'mask') {
    const cutoff = Number.isFinite(mat.alphaCutoff) ? Math.min(1, Math.max(0, mat.alphaCutoff ?? 0.5)) : 0.5;
    return opacity < cutoff;
  }
  // Fractional blend is a composition problem, not a traversal problem. The
  // realtime backend approximates only the fully-transparent endpoint here.
  return opacity <= 0;
}

/** Resolve a triangle's (roughness, metalness, ior) for packing, applying the
 *  B1 diffuse-default invariant (no authored roughness → 0.85; glass → 0.05)
 *  and the B1-ior-per-tri IOR default invariant (no authored ior on glass → 1.5;
 *  opaque → 1.0 — not consumed for opaque surfaces). */
function resolveRoughMetal(
  roughness: number | undefined,
  metalness: number | undefined,
  transmission: number | undefined,
  ior?: number,
): { rough: number; metal: number; ior: number } {
  const isGlass = (transmission ?? 0) > 0.5;
  let rough: number;
  if (roughness === undefined || !Number.isFinite(roughness)) {
    rough = isGlass ? ROUGH_GLASS : ROUGH_DEFAULT;
  } else {
    rough = Math.min(1, Math.max(0, roughness));
  }
  const metal = Math.min(1, Math.max(0, metalness ?? 0));
  // IOR: glass defaults to 1.5, opaque defaults to 1.0 (irrelevant — not consumed).
  const resolvedIor = (ior !== undefined && Number.isFinite(ior) && ior > 0)
    ? Math.min(IOR_RANGE_MAX, Math.max(IOR_RANGE_MIN, ior))
    : (isGlass ? IOR_DEFAULT_GLASS : IOR_RANGE_MIN);
  return { rough, metal, ior: resolvedIor };
}

/**
 * Pack one triangle's roughness+metalness+IOR into a parallel u32 buffer.
 * @deprecated Superseded by {@link packBVHRoughMetalFromCore} (core `MaterialSpec[]`).
 *   The structural `PbrMaterialLike`-based family is legacy. Retained for tests only.
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
 * @deprecated Superseded by {@link packBVHBeerColorsFromCore} (core `MaterialSpec[]`).
 *   The structural `PbrMaterialLike`-based family is legacy. Retained for tests only.
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
 * Used by the material-only `updatePrimitive` fast path.
 *
 * @deprecated Structural `PbrMaterialLike`-based family; legacy adapter.
 *   The canonical location of this function is also re-exported from
 *   `restir/bvhCore.ts` (D6.7, R6 E sweep, 2026-06-11) for subsystem-local
 *   access. Retained here for back-compatibility.
 */
function _repackBVHMaterialRange(
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
 * @deprecated Superseded by {@link packBVHIndexWFromCore} (core `MaterialSpec[]`).
 *   The structural `PbrMaterialLike`-based family is legacy; production code uses
 *   the `*FromCore` counterparts. Retained for tests only.
 */
function _packBVHIndexW(
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
 * Read by the ReSTIR/shade WGSL via decodeRoughMetal+decodeIor(triIndex).
 * See packBVHRoughMetalTri for the B1 diffuse-default invariant and the
 * B1-ior-per-tri IOR default invariant (glass → 1.5; opaque → 1.0).
 * @deprecated Superseded by {@link packBVHRoughMetalFromCore} (core `MaterialSpec[]`).
 *   The structural `PbrMaterialLike`-based family is legacy. Retained for tests only.
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
 * Read by shade.wgsl Lo_emit on a primary glass hit.
 * @deprecated Superseded by {@link packBVHBeerColorsFromCore} (core `MaterialSpec[]`).
 *   The structural `PbrMaterialLike`-based family is legacy. Retained for tests only.
 */
function _packBVHBeerColors(
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
 * or `null` when the material is not a self-emissive surface. This
 * mirrors the EMISSIVE branch of `classifyTriangleEmitter` (emitterList.ts) EXACTLY
 * — so the camera-visible glow Le equals the radiance ReSTIR-DI samples for that
 * emitter — but deliberately EXCLUDES the transmissive "sun-attenuated secondary
 * emitter" branch: glass self-emission to the camera is already handled by
 * shade.wgsl `lo_emit` (Beer-Lambert), so packing it here would double-count.
 * @deprecated Structural `PbrMaterialLike` family; legacy adapter. Production code
 *   uses `materialSpecEmissiveLe` from `@vitrum/shared-bvh` via the `*FromCore` packers.
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
 * (stride 4: rgb + 0 pad). Read by shade.wgsl `lo_emitterGlow` on a primary hit
 * so emissive-mesh surfaces are CAMERA-VISIBLE (the real-time analogue of the
 * pt-webgpu camera-visible-emitters fix). Non-emissive triangles are zero. HDR
 * (emissiveIntensity may exceed 1), hence float — not the LDR `bvh_beer` u32.
 * @deprecated Superseded by {@link packBVHEmissiveLeFromCore} (core `MaterialSpec[]`).
 *   The structural `PbrMaterialLike`-based family is legacy. Retained for tests only.
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

// ──────────────────────────────────────────────────────────────────────────
// Core-material per-triangle packers.
//
// These are the core-`MaterialSpec` counterparts to the three `*Tri` packers
// above. They delegate the per-material RGB resolution to the canonical
// `materialEntry.ts` mirrors in `@vitrum/shared-bvh`
// (`materialSpecTriColor` / scalar emissive Le /
// `materialSpecSurfaceTextureId`) — the same functions the DDGI/emitter
// decouples already use — and reproduce the EXACT RGBA8 / trans4 / isMetal
// bit-packing + warm-gray missing-material default of the structural PBR packers
// byte-for-byte. The caller drives them with a parallel `coreMaterials[]` built
// in material-slot order, so output is per-triangle stable with the production
// packers, not merely set-equivalent.
// ──────────────────────────────────────────────────────────────────────────

/**
 * Apply the production emissive convention to a core material before reading
 * its emissive Le: treat `emissive` as the FINAL radiance-space colour and force
 * `emissiveIntensity = 1`, so `materialSpecEmissiveLe` yields
 * `Le = emissive * 1`. This is the SAME ei-collapse fix the ReSTIR-DI
 * emitter decouple (`restir/bvhCore.ts:toProductionEmissiveRadiance`, commit
 * `46a0078`) and the DDGI material decouple (`probeUpdateMaterials.ts`, commit
 * `15070cd`) needed: a raw `materialSpecEmissiveLe` computes
 * `emissive * emissiveIntensity`, so a core emitter with `ei = 4` would pack 4x
 * the intended radiance — the exact divergence those GPU A/Bs caught.
 * A material with no `emissive` is returned unchanged (not an emitter either way).
 */
function toProductionEmissiveRadiance(m: MaterialSpec): MaterialSpec {
  if (m.emissive === undefined) return m;
  if (m.emissiveIntensity === 1) return m; // already the production convention
  return { ...m, emissiveIntensity: 1 };
}

/**
 * Scalar production Le for the camera-visible emissive buffer.
 *
 * Readable emissiveMap energy is intentionally NOT folded in here: shade.wgsl
 * samples the emissive atlas at the hit UV, so averaging the map into this
 * buffer would apply readable emissive maps twice. ReSTIR emitter selection
 * still uses map-averaged power through `materialSpecEmissiveLe`.
 */
function scalarProductionEmissiveLe(m: MaterialSpec): [number, number, number] | null {
  const production = toProductionEmissiveRadiance(m);
  const em = production.emissive;
  if (!em) return null;
  const ei = production.emissiveIntensity ?? 1;
  if (!(ei > 0)) return null;
  if (em[0] <= 0 && em[1] <= 0 && em[2] <= 0) return null;
  const out: [number, number, number] = [
    em[0] * ei,
    em[1] * ei,
    em[2] * ei,
  ];
  if (out[0] <= 0 && out[1] <= 0 && out[2] <= 0) return null;
  return out;
}

/**
 * Core-material counterpart to {@link packBVHIndexW}: pack vertex indices + RGBA8
 * baseColor + (trans4 | isMetal | texType) per triangle from a parallel
 * `MaterialSpec[]`. Mirrors {@link packBVHIndexWTri} field-for-field:
 *  - RGB ← `materialSpecTriColor(mat, /*applyBeer*\/ false)` × 255 & 0xFF
 *    (the RAW attenuation color for a transmissive surface, else baseColor).
 *  - trans4 ← `min(15, round(transmission · 15)) & 0xF`.
 *  - isMetal ← `metallic > 1e-4 ? 1 : 0`.
 *  - texType ← `materialSpecSurfaceTextureId(mat) & 0x7`.
 *  - low byte ← `((trans4 << 4) | (isMetal << 3) | (texType & 0x7)) & 0xFF`.
 * A missing material slot falls back to the warm-gray default + zero
 * transmission/texType/metal.
 */
export function packBVHIndexWFromCore(
  indices: Uint32Array,
  triMaterialId: Uint32Array,
  materials: readonly MaterialSpec[],
  triCount: number,
): Uint32Array<ArrayBuffer> {
  const indexBuf = new Uint32Array(triCount * 4);
  for (let t = 0; t < triCount; t++) {
    const base4 = t * 4;
    indexBuf[base4 + 0] = indices[t * 3 + 0]!;
    indexBuf[base4 + 1] = indices[t * 3 + 1]!;
    indexBuf[base4 + 2] = indices[t * 3 + 2]!;

    const mat = materials[triMaterialId[t]!];
    let r = WARM_GRAY_DEFAULT_R, g = WARM_GRAY_DEFAULT_G, b = WARM_GRAY_DEFAULT_B;
    let transmission = 0;
    let texTypeId = 0;
    let isMetal = 0;
    if (mat) {
      transmission = mat.transmission ?? 0;
      const color = materialSpecTriColor(mat, /* applyBeer */ false);
      r = Math.round(color[0] * 255) & 0xFF;
      g = Math.round(color[1] * 255) & 0xFF;
      b = Math.round(color[2] * 255) & 0xFF;
      texTypeId = materialSpecSurfaceTextureId(mat) & 0x7;
      const metalness = mat.metallic ?? 0;
      isMetal = metalness > 1e-4 ? 1 : 0;
    }
    const trans4 = Math.min(15, Math.round(transmission * 15)) & 0xF;
    const lowByte = ((trans4 << 4) | (isMetal << 3) | (texTypeId & 0x7)) & 0xFF;
    indexBuf[base4 + 3] = (r << 24) | (g << 16) | (b << 8) | lowByte;
  }
  return indexBuf;
}

/**
 * Core-material counterpart to {@link packBVHRoughMetal}: pack per-triangle
 * roughness+metalness+IOR (bits[31:24]=rough×255, bits[23:16]=metal×255,
 * bits[15:8]=ior_quantized) from a `MaterialSpec[]`. Applies the SAME B1
 * diffuse-default invariant as the structural packer: no authored `roughness` →
 * 0.85 (0.05 for glass, transmission > 0.5); metalness from `mat.metallic ?? 0`.
 * IOR from `mat.ior ?? 1.5` (glass) / `1.0` (opaque). Missing slot →
 * (0.85, 0, opaque-1.0). Mirrors {@link packBVHRoughMetalTri} byte-for-byte so
 * the core and structural paths produce identical per-triangle output.
 *
 * SHADOW-01 (2026-06-11) — material-flag bit 0 carries the
 * source PRIMITIVE's castShadow flag (1 ⟺ castShadow:false — "does NOT cast
 * shadows"). The flag rides the material slot because both walkaround BVH paths
 * give castShadow:false primitives distinct slots (the TLAS path's
 * `materialResolver` is per-primitive; the merged path opts into
 * `splitMaterialsByCastShadow`). Default (true/undefined) packs 0 —
 * byte-identical to the pre-SHADOW-01 lane. Consumed by the shared-bvh
 * cast-shadow-masked any-hit traversal in the ReSTIR DI shadow predicates.
 *
 * GLTF-unlit (2026-06-11) — bit 1 carries `MaterialSpec.shadingModel ===
 * 'unlit'`. Shade consumes it as a lighting-independent base-color output.
 * Default PBR materials keep bit 1 clear, preserving the pre-unlit lane.
 *
 * Scalar alpha cutout (2026-06-13) — bit 2 carries the backend's scalar
 * coverage decision: `alphaMode:'mask'` discards when `opacity < alphaCutoff`;
 * `alphaMode:'blend'` discards only the fully transparent endpoint
 * (`opacity <= 0`). Readable `alphaMap` handles are evaluated by the material
 * atlas traversal wrapper; fractional blend is warned as approximate by
 * HybridEngine.setScene.
 *
 * AO map strength (2026-06-14) — bits 3-7 carry `aoMapIntensity` quantized to
 * 5 bits. Shade applies the glTF occlusion formula `mix(1, aoMap.r, strength)`
 * and multiplies it into the runtime GTAO factor.
 */
export function packBVHRoughMetalFromCore(
  triMaterialId: Uint32Array,
  materials: readonly MaterialSpec[],
  triCount: number,
): Uint32Array<ArrayBuffer> {
  const rmBuf = new Uint32Array(triCount);
  for (let t = 0; t < triCount; t++) {
    const mat = materials[triMaterialId[t]!];
    let rough = ROUGH_DEFAULT;
    let metal = 0;
    let ior = IOR_RANGE_MIN;
    let castShadowDisabled = 0;
    let unlit = 0;
    let scalarAlphaDiscardedFlag = 0;
    let aoStrengthBits = 0;
    if (mat) {
      const rm = resolveRoughMetal(mat.roughness, mat.metallic, mat.transmission, mat.ior);
      rough = rm.rough;
      metal = rm.metal;
      ior = rm.ior;
      castShadowDisabled =
        (mat as MaterialSpec & { castShadow?: boolean }).castShadow === false
          ? BVH_MATERIAL_CAST_SHADOW_DISABLED_BIT
          : 0;
      unlit = mat.shadingModel === 'unlit' ? BVH_MATERIAL_UNLIT_BIT : 0;
      scalarAlphaDiscardedFlag = scalarAlphaDiscarded(mat)
        ? BVH_MATERIAL_SCALAR_ALPHA_DISCARDED_BIT
        : 0;
      aoStrengthBits = mat.aoMap != null ? quantizeAoMapIntensity(mat.aoMapIntensity) : 0;
    }
    rmBuf[t] = (
      packRoughMetalIorBytes(rough, metal, ior) |
      castShadowDisabled |
      unlit |
      scalarAlphaDiscardedFlag |
      aoStrengthBits
    ) >>> 0;
  }
  return rmBuf;
}

/**
 * THREE-free counterpart to {@link packBVHBeerColors}: pack the Beer-Lambert
 * visible color per triangle into a parallel u32 buffer from a `MaterialSpec[]`.
 * Mirrors {@link packBVHBeerColorTri}: RGB ←
 * `materialSpecTriColor(mat, /*applyBeer*\/ true)`, `min(1, c) · 255 & 0xFF`,
 * packed `(r << 24) | (g << 16) | (b << 8)`. Warm-gray default for a missing slot.
 */
export function packBVHBeerColorsFromCore(
  triMaterialId: Uint32Array,
  materials: readonly MaterialSpec[],
  triCount: number,
): Uint32Array<ArrayBuffer> {
  const beerBuf = new Uint32Array(triCount);
  for (let t = 0; t < triCount; t++) {
    const mat = materials[triMaterialId[t]!];
    let r = WARM_GRAY_DEFAULT_R, g = WARM_GRAY_DEFAULT_G, b = WARM_GRAY_DEFAULT_B;
    if (mat) {
      const color = materialSpecTriColor(mat, /* applyBeer */ true);
      r = Math.round(Math.min(1, color[0]) * 255) & 0xFF;
      g = Math.round(Math.min(1, color[1]) * 255) & 0xFF;
      b = Math.round(Math.min(1, color[2]) * 255) & 0xFF;
    }
    beerBuf[t] = (r << 24) | (g << 16) | (b << 8);
  }
  return beerBuf;
}

/**
 * Pack per-triangle HDR emissive radiance Le (stride-4 f32, rgb + 0 pad) from
 * a `MaterialSpec[]`. Non-emissive / missing triangles stay zero; emissive
 * triangles get scalar production Le. Readable emissive maps are sampled in
 * shade.wgsl, not averaged into this buffer.
 */
export function packBVHEmissiveLeFromCore(
  triMaterialId: Uint32Array,
  materials: readonly MaterialSpec[],
  triCount: number,
): Float32Array<ArrayBuffer> {
  const out = new Float32Array(triCount * 4);
  for (let t = 0; t < triCount; t++) {
    const mat = materials[triMaterialId[t]!];
    if (!mat) continue;
    const le = scalarProductionEmissiveLe(mat);
    if (le == null) continue;
    out[t * 4 + 0] = le[0];
    out[t * 4 + 1] = le[1];
    out[t * 4 + 2] = le[2];
  }
  return out;
}
