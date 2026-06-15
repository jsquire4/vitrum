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

import type { MaterialSpec } from '@vitrum/core';
import {
  MATERIAL_TRANSMISSIVE_COLOR_THRESHOLD,
  MATERIAL_EMITTER_TRANSMISSION_THRESHOLD,
  MATERIAL_EMITTER_SUN_DOT_THRESHOLD,
  MATERIAL_DEFAULT_TRI_COLOR,
} from './materialEntry.js';

/**
 * Emissive radiance Le (`emissive.rgb · emissiveIntensity`) of a core
 * `MaterialSpec`, or `null` when the surface is not self-emissive.
 *
 * Uses the same reject conditions as the rest of the core emitter pipeline
 * (absent emissive, non-positive intensity, all-non-positive emissive channels),
 * so the camera-visible glow Le and the NEE-sampled emitter radiance share one
 * source. Deliberately EXCLUDES the transmissive "sun-attenuated secondary
 * emitter" branch — that lives in {@link classifyTriangleEmitterCore}.
 * A missing `emissiveIntensity` defaults to ×1, matching the core material
 * entry adapter and the path-tracing backends.
 *
 * @param material a core `MaterialSpec`.
 * @returns `[r, g, b]` HDR radiance, or `null` for a non-emissive surface.
 */
export function materialSpecEmissiveLe(
  material: MaterialSpec,
): [number, number, number] | null {
  const em = material.emissive;
  if (!em) return null;
  const ei = material.emissiveIntensity ?? 1;
  if (!(ei > 0)) return null;
  if (em[0] <= 0 && em[1] <= 0 && em[2] <= 0) return null;
  return [em[0] * ei, em[1] * ei, em[2] * ei];
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
