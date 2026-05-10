/**
 * hgPhase.ts — Henyey-Greenstein phase function.
 *
 * Used by the Sprint 7 volume scattering path AND single-scatter SSS
 * (opalescent, glueChip, ringMottled glass types).
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
 *   The HG distribution has a closed-form inversion:
 *     cosθ = (1 + g² - ((1 - g²) / (1 - g + 2g·u))²) / (2g)   if g ≠ 0
 *     cosθ = 1 - 2u                                              if g = 0
 *
 * References:
 *   Henyey, Greenstein 1941, "Diffuse radiation in the galaxy",
 *   Astrophysical Journal 93:70–83.
 *
 *   Pharr, Jakob, Humphreys "Physically Based Rendering" 4th ed., §11.4
 *   "Henyey-Greenstein Phase Function".
 *
 *   Phase 6 Sprint 7 spec: plan/phase-6-roadmap.md §Sprint 7.
 *   Fork GLSL mirror: plan/sprint-7-pt-fork-patch.md §bsdf.glsl.js.
 */

// ────────────────────────────────────────────────────────────────────────────
// Public API
// ────────────────────────────────────────────────────────────────────────────

const INV_4PI = 1 / (4 * Math.PI);

/**
 * Evaluate the Henyey-Greenstein phase function p(cosθ, g).
 *
 * @param cosTheta - dot(wi, wo): cosine of scattering angle.  In [-1, 1].
 * @param g        - anisotropy parameter in (-1, 1).  Clamped to (-0.9999, 0.9999).
 * @returns        Differential probability density [sr⁻¹].
 *                 Integrates to 1 over the full sphere.
 */
export function evaluateHG(cosTheta: number, g: number): number {
  // Clamp g away from exact ±1 to avoid division by zero in the denominator.
  g = Math.max(-0.9999, Math.min(0.9999, g));
  const g2 = g * g;
  const denom = 1 + g2 - 2 * g * cosTheta;
  // denom^(3/2)
  return INV_4PI * (1 - g2) / (denom * Math.sqrt(denom));
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
  g = Math.max(-0.9999, Math.min(0.9999, g));

  let cosTheta: number;
  if (Math.abs(g) < 1e-4) {
    // Isotropic: cosTheta uniformly distributed in [-1, 1]
    cosTheta = 1 - 2 * u2;
  } else {
    // HG inversion (Pharr et al. §11.4)
    const sqrtTerm = (1 - g * g) / (1 - g + 2 * g * u2);
    cosTheta = (1 + g * g - sqrtTerm * sqrtTerm) / (2 * g);
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
