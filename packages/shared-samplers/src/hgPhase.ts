/**
 * hgPhase.ts — Henyey-Greenstein phase function.
 *
 * Public CPU reference implementation and test/oracle surface. Runtime shader
 * integrations own their language-native kernels; for example pt-webgl2's
 * `sampleHG_glsl` explicitly mirrors `sampleHG`.
 *
 * The HG phase function describes the angular distribution of scattered light
 * in a participating medium:
 *
 *   p(cosθ, g) = (1 - g²) / (4π (1 + g² - 2g·cosθ)^(3/2))
 *
 * where:
 *   cosθ = dot(wi, wo) — cosine of the angle between incoming and outgoing directions
 *   g ∈ (-1, 1)        — anisotropy parameter
 *     g = 0:  isotropic (equal scatter in all directions, reduces to 1/4π)
 *     g > 0:  forward-scattering (glass, water — majority forward)
 *     g < 0:  back-scattering (milk, fog — minority back)
 *
 * Conventions:
 *   - wi: incoming direction (pointing TOWARD the scatter point, as is
 *     standard in path tracers; cosθ = dot(wi, wo))
 *   - wo: outgoing direction (pointing AWAY from the scatter point)
 *   - The function is normalized: ∫_{S²} p(cosθ, g) dω = 1
 *
 * Sampling:
 *   The HG distribution has a closed-form inversion. The implementation uses
 *   an algebraically exact rational form near g=0 so the same random variate
 *   maps continuously through isotropy without a cancellation-prone 0/0.
 *
 * References:
 *   Henyey, Greenstein 1941, "Diffuse radiation in the galaxy",
 *   Astrophysical Journal 93:70–83.
 *
 *   Pharr, Jakob, Humphreys "Physically Based Rendering" 4th ed., §11.4
 *   "Henyey-Greenstein Phase Function".
 *
 *   Phase 6 Sprint 7 spec: plan/archive/phase-6-roadmap.md §Sprint 7.
 *   Fork GLSL mirror: plan/archive/sprint-7-pt-fork-patch.md §bsdf.glsl.js.
 */

// ────────────────────────────────────────────────────────────────────────────
// Public API
// ────────────────────────────────────────────────────────────────────────────
import { requireFinite, requireUnitRandom } from './numericGuards.js';


const INV_4PI = 1 / (4 * Math.PI);
/** Shared CPU/shader stability cap; authored values remain distinct up to this bound. */
export const HG_G_STABILITY_LIMIT = 0.999999;

/**
 * Evaluate the Henyey-Greenstein phase function p(cosθ, g).
 *
 * @param cosTheta - dot(wi, wo): cosine of scattering angle.  In [-1, 1].
 * @param g        - anisotropy parameter in (-1, 1). Numerically capped to
 *                   ±{@link HG_G_STABILITY_LIMIT}.
 * @returns        Differential probability density [sr⁻¹].
 *                 Integrates to 1 over the full sphere.
 */
export function evaluateHG(cosTheta: number, g: number): number {
  requireFinite(cosTheta, 'evaluateHG.cosTheta');
  requireFinite(g, 'evaluateHG.g');
  cosTheta = Math.max(-1, Math.min(1, cosTheta));
  // Clamp g away from exact ±1 to avoid division by zero in the denominator.
  g = Math.max(-HG_G_STABILITY_LIMIT, Math.min(HG_G_STABILITY_LIMIT, g));
  // Evaluate around the signed lobe axis without subtracting nearly equal O(1)
  // terms. This stays accurate at the shared ±0.999999 stability boundary and
  // mirrors the WebGPU and WebGL2 kernels.
  const a = Math.abs(g);
  const alignedCos = g >= 0 ? cosTheta : -cosTheta;
  const oneMinusA = 1 - a;
  const denom =
    oneMinusA * oneMinusA + 2 * a * (1 - alignedCos);
  return INV_4PI * (oneMinusA * (1 + a)) /
    (denom * Math.sqrt(denom));
}

/**
 * Sample a direction from the Henyey-Greenstein phase function.
 *
 * Returns a direction in local space where +Z is aligned with the incoming
 * direction wi (i.e., the forward direction).  The caller is responsible for
 * transforming the result to world space via an ONB.
 *
 * @param u1 - Uniform random in [0, 1) for azimuthal sampling.
 * @param u2 - Uniform random in [0, 1) for polar sampling.
 * @param g  - Anisotropy parameter in (-1, 1).
 * @returns  Outgoing direction in local space (unit vector; wo where +Z = wi).
 */
export function sampleHG(
  u1: number,
  u2: number,
  g: number,
): readonly [number, number, number] {
  requireUnitRandom(u1, 'sampleHG.u1');
  requireUnitRandom(u2, 'sampleHG.u2');
  requireFinite(g, 'sampleHG.g');
  g = Math.max(-HG_G_STABILITY_LIMIT, Math.min(HG_G_STABILITY_LIMIT, g));

  const q = 1 - 2 * u2;
  let cosTheta: number;
  if (Math.abs(g) < 0.125) {
    // Exact rational rearrangement of the HG inverse. At g=0 this reduces
    // exactly to q, preserving the random-variate mapping continuously.
    const d = 1 + g * q;
    const numerator =
      2 * q +
      g * (q * q + 3) +
      2 * g * g * q +
      g * g * g * (q * q - 1);
    cosTheta = numerator / (2 * d * d);
  } else {
    const ratio = (1 - g * g) / (1 + g * q);
    cosTheta = (1 + g * g - ratio * ratio) / (2 * g);
  }

  // Clamp to valid range (floating-point safety)
  cosTheta = Math.max(-1, Math.min(1, cosTheta));
  const sinTheta = Math.sqrt(Math.max(0, 1 - cosTheta * cosTheta));

  // Azimuthal angle
  const phi = 2 * Math.PI * u1;
  const cosPhi = Math.cos(phi);
  const sinPhi = Math.sin(phi);

  // Local direction: (x, y, z) where z = cosTheta (the forward axis is +Z)
  return [sinTheta * cosPhi, sinTheta * sinPhi, cosTheta] as const;
}

/**
 * PDF of sampling direction wo (characterized by its cosine with wi) from
 * the Henyey-Greenstein distribution with anisotropy g.
 *
 * Equal to evaluateHG(cosTheta, g) since HG is already a probability density
 * on the unit sphere.
 *
 * @param cosTheta - dot(wi, wo).
 * @param g        - anisotropy parameter in (-1, 1).
 * @returns        PDF value [sr⁻¹].
 */
export function pdfHG(cosTheta: number, g: number): number {
  return evaluateHG(cosTheta, g);
}
