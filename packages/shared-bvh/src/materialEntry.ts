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
 *   materialId LUT that `buildSceneBVH` already produces. The whole
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
 *
 * Defaults (when a field is absent on the input):
 *   baseColor = (1,1,1), emissive = (0,0,0), roughness = 1.0, metalness = 0,
 *   ior = 1.5, transmission = 0, attenuationColor = (1,1,1),
 *   attenuationDistance = Infinity (packed as 1e9 — `Infinity` is not
 *   round-trip-safe through f32 in some legacy WGSL paths), thickness = 0,
 *   flags = 0.
 *
 * Default roughness: 1.0 — picked because (a) Lambertian fallback is the
 * safest "I have no idea what this surface is" behaviour and (b) THREE's
 * `MeshStandardMaterial` constructor sets `roughness = 1.0` by default, so
 * the canonical default lines up with the runtime-observed default in every
 * Three.js scene we currently consume. The pre-W2-C5 ddgi `extractThreePbr-
 * Scalars` default was 0.5 but only fired when a THREE material was
 * **missing** the field — which Three's constructor never lets happen —
 * making the divergence a code-level artefact rather than a runtime one.
 *
 * @since W2-C5 (premium-grade-refactor-20260517.md §W2 sub-task 5).
 */

/** Floats per entry. 16 × 4 = 64 bytes. */
export const MATERIAL_ENTRY_FLOATS = 16;

/** Byte stride per entry. */
export const MATERIAL_ENTRY_STRIDE_BYTES = MATERIAL_ENTRY_FLOATS * 4;

/** Flag bit: surface is transmissive (transmission > 0). */
export const MATERIAL_FLAG_IS_GLASS = 0x1;

/** Library-canonical default roughness — used when an input lacks one.
 *  See module docstring for the choice rationale. */
export const MATERIAL_DEFAULT_ROUGHNESS = 1.0;

/** Sentinel used in place of `Infinity` for attenuationDistance — finite f32
 *  values are safer across f32 round-trip through WGSL. Same value the
 *  pre-W2-C5 RC packer used. */
export const MATERIAL_ATTEN_DIST_INFINITE = 1e9;

/**
 * Engine-independent PBR-scalar bag accepted by {@link packMaterials}.
 *
 * Engines build their `MaterialEntryInput[]` from whatever source-of-truth
 * they use (THREE materials via `extractThreePbrScalars`, vitrum
 * `Material` records, a path-tracer's `MaterialBag`, …). The packer never
 * touches THREE — it is pure data.
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
   * future `isLight` bit at position 1).
   */
  flags?: number;
}

/**
 * Pack a list of canonical material inputs into a tightly-packed
 * `Float32Array` of `entries.length * 16` floats.
 *
 * @param mats        Input materials. May be shorter or longer than
 *                    `maxCount` — see the `maxCount` parameter for
 *                    truncation / padding behaviour.
 * @param maxCount    When provided, the output is exactly
 *                    `maxCount * MATERIAL_ENTRY_FLOATS` floats long.
 *                    Inputs beyond `maxCount` are silently dropped;
 *                    remaining slots are zero-padded so the consumer
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
  const count = maxCount ?? Math.max(1, mats.length);
  const out = new Float32Array(count * ENTRY);
  const u32view = new Uint32Array(out.buffer);
  const n = Math.min(mats.length, count);

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
    // Treat undefined / non-finite / negative as "no attenuation".
    const attenDistF =
      attenuationDistance === undefined ||
      !Number.isFinite(attenuationDistance) ||
      attenuationDistance <= 0
        ? MATERIAL_ATTEN_DIST_INFINITE
        : attenuationDistance;
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
