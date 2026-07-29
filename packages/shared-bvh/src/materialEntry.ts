/**
 * Canonical scene-material packing for vitrum's per-material GPU buffers
 * (W2-C5 sweep finding).
 *
 * One layout, one packer, one WGSL struct — consumed by every engine that
 * uploads per-material PBR scalars to a compute/raster shader as a struct
 * array (DDGI probe update; RC cascade cast; future passes).
 *
 * Why this lives in `@vitrum/shared-bvh`:
 *   The per-material struct is bound alongside the per-triangle BVH +
 *   materialId LUT emitted by the shared scene packers. The whole
 *   "scene → GPU storage" handoff is BVH-scoped infrastructure, so the
 *   canonical packer co-locates with the BVH builder rather than living
 *   inside one of the consuming engines.
 *
 * Pre-W2-C5 history (and what the unification cleans up):
 *   - `walkaround-hybrid/src/ddgi/probeUpdatePass.ts` packed 16 floats with
 *     baseColor·_pad0·emissive·roughness·metalness·ior·transmission·_pad1
 *     ·attenuationColor·flags(u32). It carried NEITHER attenuationDistance
 *     nor thickness, so the per-cell Beer-Lambert tint was a simplified
 *     `attenColor * transmission`.
 *   - `walkaround-rc/src/bvhCompute.ts` (formerly `walkaround-hybrid/src/rc/` —
 *     extracted to `@vitrum/walkaround-rc` 2026-05-18 W8 follow-up) packed 16 floats with
 *     colorR·G·B·colorA(=1)·transmission·ior·attenColorR·G·B·attenDist·
 *     roughness·metalness·emissiveR·G·B·thickness. Same logical content but
 *     a completely different field order.
 *   - `walkaround-hybrid/src/restir/bvhCompute.ts` does NOT pack a per-
 *     material struct buffer at all — it packs RGBA8 base color into
 *     `bvhIndex.w` and a parallel `bvhBeerColors` u32-per-tri buffer.
 *     That's a fundamentally different mechanism (per-triangle u32 vs
 *     per-material 64-byte struct) and CANNOT be unified into this layout.
 *     ReSTIR's color-packing remains an engine-specific concern.
 *
 * Canonical 16-float / 64-byte layout (std140-compatible — every vec3f
 * field is followed by an f32 so each vec3 lands on a 16-byte boundary):
 *
 *   slot  offset  field             type   purpose
 *   0     0       baseColor.r       f32    diffuse / albedo RGB
 *   1     4       baseColor.g       f32
 *   2     8       baseColor.b       f32
 *   3     12      roughness         f32    GGX roughness (clamped)
 *   4     16      emissive.r        f32    emissive RGB (pre-multiplied by
 *   5     20      emissive.g        f32    emissiveIntensity at pack time)
 *   6     24      emissive.b        f32
 *   7     28      metalness         f32    GGX metallic
 *   8     32      ior               f32    refractive index
 *   9     36      transmission      f32    [0,1] — fraction of light passing
 *   10    40      attenuationDist   f32    Beer-Lambert distance (Infinity →
 *                                          no attenuation)
 *   11    44      thickness         f32    sample thickness for Beer-Lambert
 *   12    48      attenuationCol.r  f32    Beer-Lambert tint RGB
 *   13    52      attenuationCol.g  f32
 *   14    56      attenuationCol.b  f32
 *   15    60      flags             u32    bit 0 = isGlass (transmission > 0)
 *                                          bit 1 = castShadow disabled
 *
 * Defaults (when a field is absent on the input):
 *   baseColor = (1,1,1), emissive = (0,0,0), roughness = 1.0, metalness = 0,
 *   ior = 1.5, transmission = 0, attenuationColor = (1,1,1),
 *   attenuationDistance = Infinity (packed as IEEE-754 +Infinity), thickness = 0,
 *   flags = 0.
 *
 * Default roughness: 1.0 — picked because (a) Lambertian fallback is the
 * safest "I have no idea what this surface is" behaviour and (b) common PBR
 * host material constructors default roughness to 1.0. The pre-W2-C5 DDGI
 * scalar helper default was 0.5 but only fired when a material was missing the
 * field, making the divergence a code-level artefact rather than a runtime one.
 *
 * @since W2-C5 (premium-grade-refactor-20260517.md §W2 sub-task 5).
 */

import type { MaterialSpec } from '@vitrum/core';

/** Floats per entry. 16 × 4 = 64 bytes. */
export const MATERIAL_ENTRY_FLOATS = 16;

/**
 * Default visible color (linear-ish RGB) used when a triangle's material is
 * absent or carries neither a usable base nor attenuation color. Exported for
 * consumers that want the same warm-gray fallback. */
export const MATERIAL_DEFAULT_TRI_COLOR: readonly [number, number, number] = [
  0.6, 0.58, 0.55,
];

/**
 * Transmission threshold above which a surface is treated as "transmissive"
 * for triangle-color resolution (Beer-Lambert tint vs. base color). */
export const MATERIAL_TRANSMISSIVE_COLOR_THRESHOLD = 0;

/** Byte stride per entry. */
export const MATERIAL_ENTRY_STRIDE_BYTES = MATERIAL_ENTRY_FLOATS * 4;

/** Flag bit: surface is transmissive (transmission > 0). */
export const MATERIAL_FLAG_IS_GLASS = 0x1;

/** Flag bit: source primitive explicitly set `castShadow:false`. */
export const MATERIAL_FLAG_CAST_SHADOW_DISABLED = 0x2;

/** Flag bit: both winding orientations are authored surface sides. */
export const MATERIAL_FLAG_DOUBLE_SIDED = 0x4;

/** Library-canonical default roughness — used when an input lacks one.
 *  See module docstring for the choice rationale. */
export const MATERIAL_DEFAULT_ROUGHNESS = 1.0;

/** Canonical no-absorption-distance value. IEEE-754 +Infinity is preserved. */
export const MATERIAL_ATTEN_DIST_INFINITE = Number.POSITIVE_INFINITY;

function isRuntimeArray(value: unknown): boolean {
  return Array.isArray(value);
}

/**
 * Engine-independent PBR-scalar bag accepted by {@link packMaterials}.
 *
 * Engines build their `MaterialEntryInput[]` from whatever source-of-truth
 * they use (core `MaterialSpec` records, structural PBR material bags, host
 * adapters, ...). The packer is pure data.
 */
export interface MaterialEntryInput {
  /** Linear-sRGB diffuse / albedo color. Defaults to (1, 1, 1). */
  baseColor?: readonly [number, number, number];
  /** GGX roughness, [0, 1]. Defaults to {@link MATERIAL_DEFAULT_ROUGHNESS}. */
  roughness?: number;
  /** Metallic factor, [0, 1]. Defaults to 0. */
  metalness?: number;
  /** Emissive radiance (pre-multiplied by emissiveIntensity).
   *  Defaults to (0, 0, 0). */
  emissive?: readonly [number, number, number];
  /** Refractive index. Defaults to 1.5 (common-glass). */
  ior?: number;
  /** [0, 1] fraction of light passing through. Defaults to 0. */
  transmission?: number;
  /** Beer-Lambert tint color. Defaults to (1, 1, 1) (no tint). */
  attenuationColor?: readonly [number, number, number];
  /** Beer-Lambert distance. Defaults to Infinity (re-packed as 1e9). */
  attenuationDistance?: number;
  /** Surface thickness for Beer-Lambert. Defaults to 0. */
  thickness?: number;
  /**
   * Explicit flag override. When omitted, {@link packMaterials} derives
   * `flags = (transmission > 0) ? MATERIAL_FLAG_IS_GLASS : 0`.
   * Pass an explicit value if the caller needs additional bits (e.g.
   * `castShadow:false` at bit 1 or `doubleSided:true` at bit 2).
   */
  flags?: number;
}

/**
 * Map a `@vitrum/core` `MaterialSpec` directly into the canonical
 * {@link MaterialEntryInput} bag — the core counterpart to the structural
 * `extractPbrScalars` adapters in `walkaround-hybrid`.
 *
 * When an engine's geometry comes from a core `Scene`, the material is already
 * a `MaterialSpec`, so it maps straight to {@link MaterialEntryInput}. This is the canonical
 * "core material → GPU material struct" bridge; feed its output to
 * {@link packMaterials} for the 64-byte SSBO layout.
 *
 * Field mapping (mirrors `extractPbrScalars` → the RC/DDGI adapters):
 *  - `baseColor` / `roughness` / `ior` / `transmission` / `attenuationColor` /
 *    `attenuationDistance` / `thickness` pass through 1:1.
 *  - `metallic` → `metalness` (the core field uses the glTF spelling; the
 *    material-entry field keeps the established shader spelling).
 *  - `emissive` is **pre-multiplied** by `emissiveIntensity` so the GPU side
 *    sees a single radiance triple — exactly as the RC adapter does
 *    (`rc/bvhCompute.ts`). A missing `emissiveIntensity` defaults to ×1
 *    (matching `extractPbrScalars`' `PBR_DEFAULTS.emissiveIntensity`).
 *
 * Any field absent on the spec is left `undefined`, so {@link packMaterials}
 * applies the library-canonical defaults (`roughness = 1.0`, `ior = 1.5`,
 * `attenuationDistance → 1e9`, etc.).
 *
 * **Deliberate non-policy:** this adapter does NOT apply RC's `thickness → 0.1`
 * floor (RC's per-tri Beer-Lambert needs a non-zero numerator; DDGI does not).
 * That floor is an RC-side policy, not a property of the material, so RC applies
 * it on top of this adapter's faithful pass-through — exactly as it does today
 * on the `extractPbrScalars` output. Keeping it out here preserves the
 * intentional RC-vs-DDGI divergence and keeps this function a pure field map.
 *
 * @param material a core `MaterialSpec` (the `@vitrum/core` PBR record).
 * @returns the engine-independent {@link MaterialEntryInput} (no defaults
 *          applied — that is {@link packMaterials}' job).
 */
export function coreMaterialToMaterialEntry(material: MaterialSpec): MaterialEntryInput {
  const out: {
    -readonly [K in keyof MaterialEntryInput]: MaterialEntryInput[K];
  } = {
    baseColor: material.baseColor,
    roughness: material.roughness,
    metalness: material.metallic,
  };
  if (material.emissive !== undefined) {
    const ei = material.emissiveIntensity ?? 1;
    out.emissive = [
      material.emissive[0] * ei,
      material.emissive[1] * ei,
      material.emissive[2] * ei,
    ];
  }
  if (material.ior !== undefined) out.ior = material.ior;
  if (material.transmission !== undefined) out.transmission = material.transmission;
  if (material.attenuationColor !== undefined) out.attenuationColor = material.attenuationColor;
  if (material.attenuationDistance !== undefined) {
    out.attenuationDistance = material.attenuationDistance;
  }
  if (material.thickness !== undefined) out.thickness = material.thickness;
  const flags =
    (material.transmission !== undefined && material.transmission > 0 ? MATERIAL_FLAG_IS_GLASS : 0) |
    ((material as MaterialSpec & { castShadow?: boolean }).castShadow === false
      ? MATERIAL_FLAG_CAST_SHADOW_DISABLED
      : 0) |
    (material.doubleSided === true ? MATERIAL_FLAG_DOUBLE_SIDED : 0);
  if (flags !== 0) out.flags = flags;
  return out;
}

/**
 * Apply the production emissive convention to a core material: treat `emissive`
 * as the FINAL radiance-space colour and force `emissiveIntensity = 1`, so any
 * downstream `emissive * emissiveIntensity` read yields `Le = emissive * 1`.
 *
 * This is the ei-collapse guard the ReSTIR-DI emitter decouple
 * (`restir/bvhCore.ts`, commit `46a0078`), the DDGI material decouple
 * (`ddgi/probeUpdateMaterials.ts`, commit `15070cd`), the RC cascade material
 * path (`rc/bvhCore.ts`), and the camera-visible-emitter packer
 * (`restir/packingHelpers.ts`) each needed independently: a raw
 * `coreMaterialToMaterialEntry` / `materialSpecEmissiveLe` computes
 * `emissive · emissiveIntensity`, so a core emitter with `ei = 4` would pack 4×
 * the intended radiance — the exact divergence those GPU A/Bs caught. Hoisting
 * the three-line byte-identical guard here gives every walkaround subsystem one
 * source of truth (D6-8).
 *
 * A material with no `emissive` is returned unchanged (not an emitter either
 * way). A material already at `emissiveIntensity === 1` is returned unchanged
 * (already the production convention — no spread allocation).
 */
export function toProductionEmissiveRadiance(m: MaterialSpec): MaterialSpec {
  if (m.emissive === undefined) return m;
  if (m.emissiveIntensity === 1) return m;
  return { ...m, emissiveIntensity: 1 };
}

// ──────────────────────────────────────────────────────────────────────────
// Core emitter / Beer-Lambert classification
//
// Extracted to emitterClassify.ts (D11.5). Re-exported here for back-compat
// so all existing `from '@vitrum/shared-bvh'` or `from './materialEntry.js'`
// imports keep working without change.
// ──────────────────────────────────────────────────────────────────────────
export {
  materialSpecEmissiveLe,
  applyBeerLambertColor,
  materialSpecTriColor,
  materialSpecSkipEmitter,
  classifyTriangleEmitterCore,
} from './emitterClassify.js';

/**
 * Pack a list of canonical material inputs into a tightly-packed
 * `Float32Array` of `entries.length * 16` floats.
 *
 * @param mats        Input materials. May be shorter than `maxCount`; an
 *                    oversize input is rejected rather than truncated.
 * @param maxCount    When provided, the output is exactly
 *                    `maxCount * MATERIAL_ENTRY_FLOATS` floats long.
 *                    Remaining slots are zero-padded so the consumer
 *                    sees a stable buffer size regardless of scene
 *                    population. When omitted the output is exactly
 *                    `mats.length` entries (with a single zeroed entry
 *                    for an empty input — every WGSL `array<T>` storage
 *                    binding needs at least one element).
 *
 * The returned buffer's underlying `ArrayBuffer` is aliased by an internal
 * `Uint32Array` so the `flags` slot is written as a real `u32` (writing
 * `1.0` as `f32` would land 0x3F800000 ≠ 1u in the GPU read).
 */
export function packMaterials(
  mats: readonly MaterialEntryInput[],
  maxCount?: number,
): Float32Array {
  const ENTRY = MATERIAL_ENTRY_FLOATS;
  if (!isRuntimeArray(mats)) {
    throw new TypeError('packMaterials: mats must be an array.');
  }
  if (maxCount !== undefined && (!Number.isSafeInteger(maxCount) || maxCount < 1)) {
    throw new RangeError(`packMaterials: maxCount must be a positive safe integer (got ${String(maxCount)}).`);
  }
  const count = maxCount ?? Math.max(1, mats.length);
  if (mats.length > count) {
    throw new RangeError(
      `packMaterials: ${mats.length} materials exceed the exact maxCount capacity ${count}.`,
    );
  }
  const knownKeys = new Set([
    'baseColor', 'roughness', 'metalness', 'emissive', 'ior', 'transmission',
    'attenuationColor', 'attenuationDistance', 'thickness', 'flags',
  ]);
  const assertF32 = (value: unknown, label: string, allowInfinity = false): number => {
    if (
      typeof value !== 'number' ||
      (!Number.isFinite(value) && !(allowInfinity && value === Number.POSITIVE_INFINITY)) ||
      (Number.isFinite(value) && !Number.isFinite(Math.fround(value)))
    ) {
      throw new RangeError(`packMaterials: ${label} must be representable as float32${allowInfinity ? ' or +Infinity' : ''}.`);
    }
    return value;
  };
  const assertVec3 = (value: unknown, label: string): readonly [number, number, number] => {
    if (!Array.isArray(value) || value.length !== 3 || Reflect.ownKeys(value).some((key) => (
      key !== 'length' && (typeof key !== 'string' || !/^[0-2]$/.test(key))
    ))) {
      throw new TypeError(`packMaterials: ${label} must be an exact 3-element array.`);
    }
    return [
      assertF32(value[0], `${label}[0]`),
      assertF32(value[1], `${label}[1]`),
      assertF32(value[2], `${label}[2]`),
    ];
  };
  for (let i = 0; i < mats.length; i += 1) {
    const material = mats[i];
    if (material == null || typeof material !== 'object' || Array.isArray(material)) {
      throw new TypeError(`packMaterials: mats[${i}] must be an object.`);
    }
    for (const key of Reflect.ownKeys(material)) {
      if (typeof key !== 'string' || !knownKeys.has(key)) {
        throw new RangeError(`packMaterials: mats[${i}] has unknown field ${String(key)}.`);
      }
    }
    if (material.baseColor !== undefined) assertVec3(material.baseColor, `mats[${i}].baseColor`);
    if (material.emissive !== undefined) assertVec3(material.emissive, `mats[${i}].emissive`);
    if (material.attenuationColor !== undefined) assertVec3(material.attenuationColor, `mats[${i}].attenuationColor`);
    for (const key of ['roughness', 'metalness', 'transmission'] as const) {
      if (material[key] !== undefined) {
        const value = assertF32(material[key], `mats[${i}].${key}`);
        if (value < 0 || value > 1) throw new RangeError(`packMaterials: mats[${i}].${key} must be in [0, 1].`);
      }
    }
    if (material.ior !== undefined && !(assertF32(material.ior, `mats[${i}].ior`) > 0)) {
      throw new RangeError(`packMaterials: mats[${i}].ior must be > 0.`);
    }
    if (material.thickness !== undefined && assertF32(material.thickness, `mats[${i}].thickness`) < 0) {
      throw new RangeError(`packMaterials: mats[${i}].thickness must be >= 0.`);
    }
    if (material.attenuationDistance !== undefined) {
      const value = assertF32(material.attenuationDistance, `mats[${i}].attenuationDistance`, true);
      if (!(value > 0)) throw new RangeError(`packMaterials: mats[${i}].attenuationDistance must be > 0 or +Infinity.`);
    }
    if (material.flags !== undefined && (
      !Number.isSafeInteger(material.flags) || material.flags < 0 || material.flags > 0xffff_ffff
    )) {
      throw new RangeError(`packMaterials: mats[${i}].flags must be an unsigned 32-bit integer.`);
    }
  }
  const out = new Float32Array(count * ENTRY);
  const u32view = new Uint32Array(out.buffer);
  const n = mats.length;

  for (let i = 0; i < n; i++) {
    const m = mats[i]!;
    const base = i * ENTRY;

    const baseColor = m.baseColor ?? [1, 1, 1];
    const emissive = m.emissive ?? [0, 0, 0];
    const attenuationColor = m.attenuationColor ?? [1, 1, 1];
    const roughness = m.roughness ?? MATERIAL_DEFAULT_ROUGHNESS;
    const metalness = m.metalness ?? 0;
    const ior = m.ior ?? 1.5;
    const transmission = m.transmission ?? 0;
    const attenuationDistance = m.attenuationDistance;
    const attenDistF = attenuationDistance ?? MATERIAL_ATTEN_DIST_INFINITE;
    const thickness = m.thickness ?? 0;
    const flags =
      m.flags ?? (transmission > 0 ? MATERIAL_FLAG_IS_GLASS : 0);

    out[base + 0] = baseColor[0];
    out[base + 1] = baseColor[1];
    out[base + 2] = baseColor[2];
    out[base + 3] = roughness;
    out[base + 4] = emissive[0];
    out[base + 5] = emissive[1];
    out[base + 6] = emissive[2];
    out[base + 7] = metalness;
    out[base + 8] = ior;
    out[base + 9] = transmission;
    out[base + 10] = attenDistF;
    out[base + 11] = thickness;
    out[base + 12] = attenuationColor[0];
    out[base + 13] = attenuationColor[1];
    out[base + 14] = attenuationColor[2];
    // flags is a true u32 — write through the aliased view so the GPU sees
    // an integer, not the IEEE-754 bit pattern of an f32.
    u32view[base + 15] = flags >>> 0;
  }

  return out;
}
