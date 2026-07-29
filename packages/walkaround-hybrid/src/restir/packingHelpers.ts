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
  quantizePackedMaterialTransmission,
  toProductionEmissiveRadiance,
} from '@vitrum/shared-bvh';
import {
  SURFACE_TEXTURE_ID,
  validateSurfaceTextureId,
  type SurfaceTextureId,
} from '@vitrum/stained-glass-extensions';

// Generic vertex-stream UV packing (packUVIntoVec4W / packUVIntoPositionW /
// BufferAttributeLike) moved to `../bvh/bvhPacking.ts` (I3-1: restir/ was a
// de-facto shared-foundation sink). Re-exported here so existing test imports
// (`from '../packingHelpers.js'`) keep resolving.
export {
  packUVIntoPositionW,
  packUVIntoVec4W,
  type BufferAttributeLike,
} from '../bvh/bvhPacking.js';

// The legacy structural-PBR (`PbrMaterialLike`) per-triangle packers
// (packBVHIndexWTri/packBVHRoughMetalTri/packBVHBeerColorTri and their
// public wrappers + `materialEmissiveLe`/`packBVHEmissiveLe`) were retained here
// as byte-identity test ORACLES for the `*FromCore` production packers. They had
// zero production consumers, so they moved to
// `restir/__tests__/support/legacyPbrPackers.ts` (D6-4 / I3). The shared pure
// helpers they depend on (`WARM_GRAY_DEFAULT_*`, `resolveRoughMetal`,
// `packRoughMetalIorBytes`) stay here — the production `*FromCore` packers use
// them too — and are `@internal`-exported for that test-support file so the
// shared math has one source of truth.

/** @internal Default warm-gray fallback color (sRGB byte values) for a triangle
 *  with no material / unrecognised material type. Shared by the production
 *  `*FromCore` packers and the legacy test-oracle packers. */
export const WARM_GRAY_DEFAULT_R = 153;
/** @internal @see WARM_GRAY_DEFAULT_R */
export const WARM_GRAY_DEFAULT_G = 148;
/** @internal @see WARM_GRAY_DEFAULT_R */
export const WARM_GRAY_DEFAULT_B = 140;

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
//   bits[15:8]  = IOR quantized into finite codes [1,255], with byte 0 reserved
//                 for the infinite-IOR sentinel. Finite values use 254 intervals
//                 over [1.0, 3.0].
//                 covering water (1.33) → glass (1.5) → diamond (2.42) → TiO₂ (≈2.9).
//                 Decode: ior = 1 + ((byte − 1) / 254) * 2.
//                 IOR_GLASS = 1.5 encodes to byte 65 and decodes exactly to 1.5.
//   bits[7:0]   = material flags + low-precision scalar sidecar:
//                 bit 0 = castShadowDisabled
//                 bit 1 = unlit shading model
//                 bit 2 = scalarAlphaDiscarded
//                 bits 3-7 = aoMapIntensity × 31 (glTF occlusion strength).
//
// DIFFUSE DEFAULT INVARIANT (B1): a material with no authored roughness packs
// ROUGH_DEFAULT = 0.85 — the EXACT value shade/ris/cast hardcoded for non-glass
// before B1 — so a default-roughness diffuse scene (metal 0) is numerically
// unchanged. Glass (any physical transmission > 0) packs 0.05 to match the prior glass
// hardcode. Metalness 0 default.
//
// IOR DEFAULT INVARIANT (B1-ior-per-tri): glass materials with no authored ior
// default to IOR_DEFAULT_GLASS = 1.5, which encodes to byte 65 and decodes
// exactly to 1.5. Opaque/non-glass materials pack IOR_DEFAULT_OPAQUE = 1.0
// (byte 1) — decodes to exactly 1.0 (air/vacuum); the WGSL consumers skip the
// IOR lane for non-glass surfaces so this value has no visual impact.
/** @internal Default roughness for a non-glass material with no authored value.
 *  Shared by the production `*FromCore` packers and the legacy test-oracle packers. */
export const ROUGH_DEFAULT = 0.85;
const ROUGH_GLASS = 0.05;
/** Default IOR for glass (crown glass). Packs to byte 65 and decodes exactly. */
export const IOR_DEFAULT_GLASS = 1.5;
/** Finite IOR range minimum (air/vacuum). Byte 1; byte 0 is the IOR=0 sentinel. */
export const IOR_RANGE_MIN = 1.0;
/** IOR range maximum (slightly above diamond 2.42, up to TiO₂ ≈ 2.9). */
export const IOR_RANGE_MAX = 3.0;

/** Quantize an IOR value to a u8 byte. Byte 0 preserves KHR's IOR=0
 *  infinite-IOR mode; finite [1,3] values use the remaining [1,255] codes. */
export function quantizeIor(ior: number): number {
  if (ior === 0) return 0;
  return (
    1 +
    Math.min(
      254,
      Math.max(
        0,
        Math.round(
          (ior - IOR_RANGE_MIN) /
          (IOR_RANGE_MAX - IOR_RANGE_MIN) *
          254,
        ),
      ),
    )
  ) & 0xFF;
}

/** Dequantize a u8 IOR byte back to a float. Inverse of {@link quantizeIor}. */
export function dequantizeIor(byte: number): number {
  if ((byte & 0xFF) === 0) return 0;
  return IOR_RANGE_MIN +
    (((byte & 0xFF) - 1) / 254) *
    (IOR_RANGE_MAX - IOR_RANGE_MIN);
}

const BVH_MATERIAL_CAST_SHADOW_DISABLED_BIT = 1 << 0;
const BVH_MATERIAL_UNLIT_BIT = 1 << 1;
const BVH_MATERIAL_SCALAR_ALPHA_DISCARDED_BIT = 1 << 2;

function quantizeAoMapIntensity(value: number | undefined): number {
  const strength = Number.isFinite(value) ? Math.min(1, Math.max(0, value ?? 1)) : 1;
  return (Math.round(strength * 31) & 0x1F) << 3;
}

/** @internal Pack (roughness, metalness, ior) into the u32 rough/metal/ior lane.
 *  Shared by the production `*FromCore` packers and the legacy test-oracle packers. */
export function packRoughMetalIorBytes(roughness: number, metalness: number, ior: number): number {
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

/** @internal Resolve a triangle's (roughness, metalness, ior) for packing, applying the
 *  B1 diffuse-default invariant (no authored roughness → 0.85; glass → 0.05)
 *  and the B1-ior-per-tri IOR default invariant (no authored ior on glass → 1.5;
 *  opaque → 1.0 — not consumed for opaque surfaces). Shared by the production
 *  `*FromCore` packers and the legacy test-oracle packers. */
export function resolveRoughMetal(
  roughness: number | undefined,
  metalness: number | undefined,
  transmission: number | undefined,
  ior?: number,
): { rough: number; metal: number; ior: number } {
  const isGlass = (transmission ?? 0) > 0;
  let rough: number;
  if (roughness === undefined || !Number.isFinite(roughness)) {
    rough = isGlass ? ROUGH_GLASS : ROUGH_DEFAULT;
  } else {
    rough = Math.min(1, Math.max(0, roughness));
  }
  const metal = Math.min(1, Math.max(0, metalness ?? 0));
  // IOR: glass defaults to 1.5, opaque defaults to 1.0 (irrelevant — not consumed).
  const resolvedIor = ior === 0
    ? 0
    : (ior !== undefined && Number.isFinite(ior) && ior > 0)
      ? Math.min(IOR_RANGE_MAX, Math.max(IOR_RANGE_MIN, ior))
      : (isGlass ? IOR_DEFAULT_GLASS : IOR_RANGE_MIN);
  return { rough, metal, ior: resolvedIor };
}

// The legacy structural-PBR (`PbrMaterialLike`) per-triangle packers —
// packBVHRoughMetalTri / packBVHBeerColorTri / packBVHIndexW /
// packBVHRoughMetal / packBVHBeerColors / materialEmissiveLe /
// packBVHEmissiveLe / repackBVHMaterialRange — moved to
// `restir/__tests__/support/legacyPbrPackers.ts` (D6-4 / I3). They were
// byte-identity test oracles for the `*FromCore` production packers below, with
// zero production consumers. The shared math (`resolveRoughMetal`,
// `packRoughMetalIorBytes`, `WARM_GRAY_DEFAULT_*`) is `@internal`-exported above
// for that test-support file.

// ──────────────────────────────────────────────────────────────────────────
// Core-material per-triangle packers.
//
// These are the core-`MaterialSpec` counterparts to the three `*Tri` packers
// above. They delegate the per-material RGB resolution to the canonical
// `materialEntry.ts` mirrors in `@vitrum/shared-bvh`
// (`materialSpecTriColor` / scalar emissive Le) and the canonical stained-glass
// surface-texture validator. They reproduce the EXACT RGBA8 / trans4 / isMetal
// bit-packing + warm-gray missing-material default of the structural PBR
// packers byte-for-byte for valid inputs. The caller drives them with a parallel
// `coreMaterials[]` built in material-slot order, so output is per-triangle
// stable with the production packers, not merely set-equivalent.
// ──────────────────────────────────────────────────────────────────────────

function materialSurfaceTextureId(
  material: MaterialSpec,
  materialIndex: number,
): SurfaceTextureId {
  const raw = material.extensions?.['surfaceTextureId'];
  if (raw === undefined) return SURFACE_TEXTURE_ID.smooth;
  return validateSurfaceTextureId(
    raw,
    `materials[${materialIndex}].extensions.surfaceTextureId`,
  );
}

/**
 * Scalar production Le for the camera-visible emissive buffer.
 *
 * The production emissive convention (`emissive` is the FINAL radiance-space
 * colour, `emissiveIntensity` collapsed to 1) is applied via the shared
 * `toProductionEmissiveRadiance` from `@vitrum/shared-bvh` (hoisted from the
 * four byte-identical subsystem copies — D6-8).
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
 *  - trans4 ← canonical nonzero-preserving 4-bit transmission quantization.
 *  - isMetal ← `metallic > 0 ? 1 : 0`.
 *  - texType ← the canonical validated `extensions.surfaceTextureId`.
 *  - low byte ← `(trans4 << 4) | (isMetal << 3) | texType`.
 * Defined invalid ids throw instead of aliasing through a low-three-bit mask.
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
    let texTypeId: SurfaceTextureId = SURFACE_TEXTURE_ID.smooth;
    let isMetal = 0;
    if (mat) {
      transmission = mat.transmission ?? 0;
      const color = materialSpecTriColor(mat, /* applyBeer */ false);
      r = Math.round(color[0] * 255) & 0xFF;
      g = Math.round(color[1] * 255) & 0xFF;
      b = Math.round(color[2] * 255) & 0xFF;
      texTypeId = materialSurfaceTextureId(mat, triMaterialId[t]!);
      const metalness = mat.metallic ?? 0;
      isMetal = metalness > 0 ? 1 : 0;
    }
    const trans4 = quantizePackedMaterialTransmission(transmission);
    const lowByte = (trans4 << 4) | (isMetal << 3) | texTypeId;
    indexBuf[base4 + 3] = (r << 24) | (g << 16) | (b << 8) | lowByte;
  }
  return indexBuf;
}

/**
 * Core-material counterpart to {@link packBVHRoughMetal}: pack per-triangle
 * roughness+metalness+IOR (bits[31:24]=rough×255, bits[23:16]=metal×255,
 * bits[15:8]=ior_quantized) from a `MaterialSpec[]`. Applies the SAME B1
 * diffuse-default invariant as the structural packer: no authored `roughness` →
 * 0.85 (0.05 for glass, transmission > 0); metalness from `mat.metallic ?? 0`.
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
 * visible color plus a dedicated per-triangle sidedness byte into a parallel
 * u32 buffer from a `MaterialSpec[]`.
 * Mirrors {@link packBVHBeerColorTri}: RGB ←
 * `materialSpecTriColor(mat, /*applyBeer*\/ true)`, `min(1, c) · 255 & 0xFF`,
 * packed `(r << 24) | (g << 16) | (b << 8) | sideFlags`, where sideFlags bit 0
 * is `MaterialSpec.doubleSided`. The low byte was previously zero and is not
 * part of the Beer RGB decode, so AO keeps its full independent 5-bit lane.
 * Warm-gray default for a missing slot.
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
    const sideFlags = mat?.doubleSided === true ? 1 : 0;
    beerBuf[t] = ((r << 24) | (g << 16) | (b << 8) | sideFlags) >>> 0;
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
