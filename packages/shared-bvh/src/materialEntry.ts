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

import type { MaterialSpec } from '@vitrum/core';

/** Floats per entry. 16 × 4 = 64 bytes. */
export const MATERIAL_ENTRY_FLOATS = 16;

/**
 * Default visible color (linear-ish RGB) used when a triangle's material is
 * absent or carries neither a usable base nor attenuation color. Mirrors the
 * THREE-side `resolveTriColor` fallback `new THREE.Color(0.6, 0.58, 0.55)`
 * (walkaround `restir/packingHelpers.ts`) so the THREE-free path resolves the
 * identical warm-gray when no color is available. Exported for the equivalence
 * test (and any consumer that wants the same fallback). */
export const MATERIAL_DEFAULT_TRI_COLOR: readonly [number, number, number] = [
  0.6, 0.58, 0.55,
];

/**
 * Transmission threshold above which a surface is treated as "transmissive"
 * for triangle-color resolution (Beer-Lambert tint vs. base color). Mirrors the
 * `transmission > 0.01` gate in the THREE-side `resolveTriColor`. */
export const MATERIAL_TRANSMISSIVE_COLOR_THRESHOLD = 0.01;

/**
 * Transmission threshold above which an opaque-but-transmissive face is eligible
 * to become a "sun-attenuated secondary emitter" in the ReSTIR emitter list.
 * Mirrors the `transmission > 0.1` gate in the THREE-side
 * `classifyTriangleEmitter` (walkaround `restir/emitterList.ts`). Strictly
 * higher than {@link MATERIAL_TRANSMISSIVE_COLOR_THRESHOLD} — a faintly
 * transmissive surface tints but does not emit. */
export const MATERIAL_EMITTER_TRANSMISSION_THRESHOLD = 0.1;

/**
 * Minimum |dot(lightDir, faceNormal)| for a transmissive face to register as a
 * sun-attenuated emitter. Mirrors the `sunDot <= 0.05` reject in the THREE-side
 * `classifyTriangleEmitter`. */
export const MATERIAL_EMITTER_SUN_DOT_THRESHOLD = 0.05;

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
 * Map a `@vitrum/core` `MaterialSpec` directly into the canonical
 * {@link MaterialEntryInput} bag — the THREE-free counterpart to the
 * `extractThreePbrScalars`-based `threeToMaterialEntryInput` adapters in
 * `walkaround-hybrid` (RC: `rc/bvhCompute.ts`; DDGI: `ddgi/probeUpdateMaterials.ts`).
 *
 * When an engine's geometry comes from a core `Scene` (the THREE-decoupling
 * path — see `plan/three-decouple-analysis-2026-06-03.md`), the material is
 * already a `MaterialSpec`, so it maps straight to {@link MaterialEntryInput}
 * with no `THREE.Material` round-trip. This is the canonical
 * "core material → GPU material struct" bridge; feed its output to
 * {@link packMaterials} for the 64-byte SSBO layout.
 *
 * Field mapping (mirrors `extractThreePbrScalars` → the RC/DDGI adapters):
 *  - `baseColor` / `roughness` / `ior` / `transmission` / `attenuationColor` /
 *    `attenuationDistance` / `thickness` pass through 1:1.
 *  - `metallic` → `metalness` (the `MaterialEntryInput` field is spelled the
 *    THREE way; the core field is spelled the glTF way).
 *  - `emissive` is **pre-multiplied** by `emissiveIntensity` so the GPU side
 *    sees a single radiance triple — exactly as the RC adapter does
 *    (`rc/bvhCompute.ts`). A missing `emissiveIntensity` defaults to ×1
 *    (matching `extractThreePbrScalars`' `PBR_DEFAULTS.emissiveIntensity`).
 *
 * Any field absent on the spec is left `undefined`, so {@link packMaterials}
 * applies the library-canonical defaults (`roughness = 1.0`, `ior = 1.5`,
 * `attenuationDistance → 1e9`, etc.) — identical to packing a THREE material
 * whose constructor left those at their defaults.
 *
 * **Deliberate non-policy:** this adapter does NOT apply RC's `thickness → 0.1`
 * floor (RC's per-tri Beer-Lambert needs a non-zero numerator; DDGI does not).
 * That floor is an RC-side policy, not a property of the material, so RC applies
 * it on top of this adapter's faithful pass-through — exactly as it does today
 * on the `extractThreePbrScalars` output. Keeping it out here preserves the
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
  return out;
}

// ──────────────────────────────────────────────────────────────────────────
// THREE-free emitter / Beer-Lambert / surface-texture classification
//
// `MaterialSpec` counterparts to the THREE-material field readers used by the
// ReSTIR/DDGI/RC emitter list + per-triangle color/glow packing
// (`walkaround-hybrid/src/restir/{packingHelpers,emitterList}.ts`). Every field
// these read exists 1:1 on the core `MaterialSpec`, so the THREE round-trip is
// not needed once the geometry is ingested from a core `Scene` (the
// THREE-decoupling path — see `plan/three-decouple-analysis-2026-06-03.md`,
// §7 next-increment 1).
//
// Each function mirrors its THREE sibling's logic EXACTLY so a CPU equivalence
// test can pin THREE-variant ≡ core-variant for matching materials, and so the
// golden-pinned downstream bytes (emitter CDF, `bvhIndex.w`, `bvh_beer`,
// `bvh_emissive_le`) are unchanged when the producer is swapped.
//
// **R3 caveat (`extensions['surfaceTextureId']` / `['skipEmitter']`).** The
// THREE path reads `mat.userData.surfaceTextureId` and `mat.userData.skipEmitter`
// off the live THREE material. On core these belong in `material.extensions`
// (the backend escape hatch). VERIFIED (2026-06-03, by code-read of
// `three-bindings/src/material.ts:convertMaterial` + `userDataKeys.ts`): the
// current THREE→core converter does **NOT** copy these two userData keys into
// `extensions` — `VITRUM_USER_DATA_KEYS` omits them and `convertMaterial` only
// ever writes `extensions.dichroicLUTs`. So for a core scene produced by today's
// `sceneFromThreeJS`, both read as `undefined` here (→ texType 0 / not-skipped),
// which is the same as a THREE material whose host never stamped them. These
// readers are therefore CONTRACT-CORRECT (they consult the documented location);
// the converter wiring that would populate `extensions` from `userData` is a
// separate, out-of-scope increment. Until that lands, a host feeding a core
// `Scene` must set `material.extensions['surfaceTextureId' | 'skipEmitter']`
// directly to exercise these lanes.
// ──────────────────────────────────────────────────────────────────────────

/**
 * Emissive radiance Le (`emissive.rgb · emissiveIntensity`) of a core
 * `MaterialSpec`, or `null` when the surface is not self-emissive.
 *
 * THREE-free counterpart to `materialEmissiveLe`
 * (`walkaround-hybrid/src/restir/packingHelpers.ts`). Mirrors it EXACTLY,
 * including the three reject conditions (absent emissive, non-positive
 * intensity, all-non-positive emissive channels), so the camera-visible glow Le
 * and the NEE-sampled emitter radiance stay byte-identical when fed from a core
 * scene. Deliberately EXCLUDES the transmissive "sun-attenuated secondary
 * emitter" branch — that lives in {@link classifyTriangleEmitterCore}.
 *
 * @param material a core `MaterialSpec`.
 * @returns `[r, g, b]` HDR radiance, or `null` for a non-emissive surface.
 */
export function materialSpecEmissiveLe(
  material: MaterialSpec,
): [number, number, number] | null {
  const em = material.emissive;
  if (!em) return null;
  const ei = material.emissiveIntensity;
  if (!(ei !== undefined && ei > 0)) return null;
  if (em[0] <= 0 && em[1] <= 0 && em[2] <= 0) return null;
  return [em[0] * ei, em[1] * ei, em[2] * ei];
}

/**
 * Apply RGB Beer-Lambert absorption to a tint color given a sample
 * thickness / attenuation-distance pair: `c' = c^(thickness/attDist)`
 * (per channel, with a 1e-6 floor). Returns the input color unchanged when any
 * required parameter is missing / non-finite / non-positive.
 *
 * THREE-free counterpart to the file-local `applyBeerLambert`
 * (`walkaround-hybrid/src/restir/packingHelpers.ts`); same math, tuple in/out.
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
 * THREE-free counterpart to `resolveTriColor`
 * (`walkaround-hybrid/src/restir/packingHelpers.ts`). Mirrors it EXACTLY:
 *  - `isTransmissive` ⇔ `transmission > {@link MATERIAL_TRANSMISSIVE_COLOR_THRESHOLD}`
 *    (0.01).
 *  - transmissive → the attenuation color ({@link applyBeerLambertColor}-tinted
 *    iff `applyBeer`).
 *  - otherwise → `baseColor`, falling back to {@link MATERIAL_DEFAULT_TRI_COLOR}.
 *
 * **Critical THREE-default parity.** The THREE `resolveTriColor` gates on
 * `isTransmissive && attenColor`, but a `THREE.MeshPhysicalMaterial` ALWAYS has
 * an `attenuationColor` — its constructor defaults it to white `(1,1,1)` (and
 * `attenuationDistance` to `Infinity`). So in the THREE path the `&& attenColor`
 * guard is always true for a transmissive surface: it returns the (possibly
 * default-white) attenuation color, NEVER the base color. On a core
 * `MaterialSpec`, `attenuationColor` is genuinely OPTIONAL and may be absent —
 * so this function treats an absent `attenuationColor` on a transmissive
 * material as white `(1,1,1)`, reproducing the THREE constructor default. Were
 * it to fall through to `baseColor` instead, the golden-pinned `bvhIndex.w` /
 * `bvh_beer` bytes (produced by the THREE path) would drift. Likewise an absent
 * `attenuationDistance` is `Infinity` here (→ {@link applyBeerLambertColor}
 * passthrough), matching the THREE default.
 *
 * Parity note on the warm-gray fallback: the THREE `resolveTriColor` ends in
 * `physMat.color ?? stdMat?.color ?? warmGray`. A core `MaterialSpec`'s
 * `baseColor` is required and non-null, so the warm-gray fallback only fires for
 * the no-material case (`packBVH*Tri` passes the literal default when
 * `materials[matId]` is missing) — the per-material call always has a base
 * color. The default is still exported + honored here for a defensively-empty
 * `baseColor` tuple, keeping behavior identical to the THREE path's final `??`.
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
    // THREE's MeshPhysicalMaterial defaults attenuationColor → white and
    // attenuationDistance → Infinity, and the transmissive branch always uses
    // the attenuation color (never baseColor). Mirror those defaults so the
    // core variant produces the same bytes for an absent-field transmissive
    // material as the THREE path that pinned the goldens.
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
  // has a base color — the warm-gray fallback mirrors the THREE path's final
  // `?? new THREE.Color(0.6, 0.58, 0.55)`, which only fires for a missing
  // material (handled caller-side). The `Array.isArray` guard is a defensive
  // runtime check for loosely-typed callers and keeps the fallback reachable.
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
 * THREE-free counterpart to the `mat.userData.surfaceTextureId` read in
 * `packBVHIndexWTri` (`walkaround-hybrid/src/restir/packingHelpers.ts`). Returns
 * `0` when absent / non-numeric (same default as the THREE path).
 *
 * See the R3 caveat block above: today's THREE→core converter does not populate
 * this from `userData`, so a host must set `extensions['surfaceTextureId']`
 * directly until that wiring lands.
 */
export function materialSpecSurfaceTextureId(material: MaterialSpec): number {
  const raw = material.extensions?.['surfaceTextureId'];
  return (typeof raw === 'number' ? raw : 0) & 0x7;
}

/**
 * Read the `skipEmitter` override from a core `MaterialSpec`'s
 * `extensions['skipEmitter']`. THREE-free counterpart to the
 * `mat.userData.skipEmitter === true` read in `classifyTriangleEmitter`
 * (`walkaround-hybrid/src/restir/emitterList.ts`). Strict `=== true` (any other
 * value, including absent, means "do not skip"). See the R3 caveat block above.
 */
export function materialSpecSkipEmitter(material: MaterialSpec): boolean {
  return material.extensions?.['skipEmitter'] === true;
}

/**
 * Classify a core `MaterialSpec` + face normal as a ReSTIR-DI emitter, or `null`
 * when the face is not selected. THREE-free counterpart to
 * `classifyTriangleEmitter` (`walkaround-hybrid/src/restir/emitterList.ts`),
 * implementing the same priority order:
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
 *     `baseColor` / `attenuationColor` default to (1,1,1) when absent (matching
 *     the THREE branch's `?? new THREE.Color(1,1,1)`).
 *  3. otherwise → `null` (skipped).
 *
 * `lightDir` is the configured primary-light direction; `primaryIntensity` is
 * its irradiance — both passed as plain numbers/tuples (no `THREE.Vector3`).
 * The caller computes power (`luminance(color) · area`) and the < 1e-8 drop, as
 * `buildEmitterList` does for the THREE path.
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
