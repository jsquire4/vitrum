/**
 * equiAngular.ts — Equi-angular sampling for volume single-scatter NEE.
 *
 * Equi-angular sampling (also known as the "Szécsi trick" or Kulla-Fajardo
 * distance sampling) samples a scatter distance along a ray weighted toward
 * the point on the ray closest to a point light source.  This dramatically
 * reduces variance when the light is close to the ray compared to uniform
 * distance sampling, which concentrates samples near t=0 regardless of
 * light position.
 *
 * Algorithm (Kulla & Fajardo 2012, §3):
 *   Given a ray with origin o and unit direction d, and a point light at p:
 *   1. Project p onto the ray: t_closest = dot(p - o, d)
 *      (the parameter where the ray passes closest to p)
 *   2. Compute the perpendicular distance from p to the ray:
 *      D = ||(p - o) - t_closest * d||
 *   3. The equi-angular distribution parameterizes t by angle θ ∈ (-π/2, π/2):
 *      t(θ) = D · tan(θ) + t_closest
 *   4. Sample θ uniformly in (θ_min, θ_max) where θ_min = atan2(-t_closest, D),
 *      θ_max = atan2(t_max - t_closest, D):
 *      θ_sampled = θ_min + u × (θ_max - θ_min)
 *      t_sampled  = D · tan(θ_sampled) + t_closest
 *   5. PDF: p(t) = D / ((1 + ((t - t_closest)/D)²) · (θ_max - θ_min) · D²)
 *            = 1 / (D · (θ_max - θ_min) · (1 + ((t - t_closest)/D)²))
 *
 * Volume scope: uniform homogeneous media. This exported CPU sampler is a
 * standalone host/oracle building block; active render backends keep their
 * shader-language transport implementations at the backend boundary.
 *
 * References:
 *   Kulla & Fajardo 2012, "Importance Sampling Techniques for Path Tracing
 *   in Participating Media", EGSR. Earlier revisions of this file cited
 *   "Kulla & Conty" — that's a different (later) Sony Imageworks author
 *   pairing on unrelated papers; Marcos Fajardo is the actual co-author
 *   here. Szécsi popularised the same idea independently in real-time
 *   contexts.
 *
 *   Pharr, Jakob, Humphreys "Physically Based Rendering" 4th ed., §14.1.2.
 *
 *   Historical implementation record:
 *   plan/archive/phase-6-roadmap.md §Sprint 7.
 */

import { requireFiniteVec3, requirePositive, requireUnitRandom } from './numericGuards.js';

// ────────────────────────────────────────────────────────────────────────────
// Public API
// ────────────────────────────────────────────────────────────────────────────

/**
 * Result of equi-angular distance sampling.
 */
export interface EquiAngularSample {
  /** Sampled distance along the ray from rayOrigin.
   *  Always in [0, ∞).  Callers should clamp to scene max-t before use. */
  readonly t: number;
  /** Probability density of the sampled distance [m⁻¹] (in terms of arc-length
   *  along the ray, NOT solid angle).  Use this for MIS balance with
   *  exponential-distance sampling. */
  readonly pdf: number;
}

/**
 * Optional tuning for {@link sampleEquiAngular}.
 */
export interface EquiAngularOptions {
  /**
   * Upper ray parameter for angular extent (maps to θ_max via atan2).
   * Default: 1e6 (caller should still clip to scene bounds).
   */
  readonly sceneTMax?: number;
  /**
   * When the light lies on the ray (D≈0), sample t uniformly on [0, L].
   * Default: 100 scene units.
   */
  readonly degenerateFallbackLength?: number;
}

/**
 * Sample a scatter distance along the ray using equi-angular sampling toward
 * a point light.
 *
 * The sample is drawn from the full unbounded ray (t ∈ [0, ∞)).  The PDF
 * integrates to 1 over [0, ∞) in the limit t_max → ∞ but is computed for
 * the finite interval [0, t_max] with t_max clamped to a large scene bound
 * internally.  In practice the caller should use the returned t without
 * further clamping unless the ray has hit a surface at t < returned t.
 *
 * @param u         - Uniform random in [0, 1).
 * @param rayOrigin - World-space ray origin.
 * @param rayDir    - World-space ray direction (unit vector).
 * @param lightPos  - World-space position of the point light.
 * @param opts      - Optional `sceneTMax` / `degenerateFallbackLength`.
 * @returns         Sampled (t, pdf) pair.
 *
 * **Degenerate guards:**
 * - `D < 1e-6` (light on the ray): uniform fallback on `[0, degenerateFallbackLength]`.
 * - `thetaRange < 1e-8` (light direction nearly perpendicular to the ray, e.g.
 *   `tClosest >> sceneTMax`): returns `{ t: 0, pdf: 0 }` to avoid division by
 *   zero downstream.  Callers should treat `pdf === 0` as "do not use this
 *   sample" and fall back to another sampling strategy (e.g. exponential
 *   distance).
 */
export function sampleEquiAngular(
  u: number,
  rayOrigin: readonly [number, number, number],
  rayDir: readonly [number, number, number],
  lightPos: readonly [number, number, number],
  opts?: EquiAngularOptions,
): EquiAngularSample {
  requireUnitRandom(u, 'sampleEquiAngular.u');
  requireFiniteVec3(rayOrigin, 'sampleEquiAngular.rayOrigin');
  requireFiniteVec3(rayDir, 'sampleEquiAngular.rayDir');
  requireFiniteVec3(lightPos, 'sampleEquiAngular.lightPos');
  const sceneTMax = requirePositive(opts?.sceneTMax ?? 1e6, 'sampleEquiAngular.sceneTMax');
  const degenerateFallbackLength = requirePositive(
    opts?.degenerateFallbackLength ?? 100,
    'sampleEquiAngular.degenerateFallbackLength',
  );
  const rayDirLength = Math.hypot(rayDir[0], rayDir[1], rayDir[2]);
  if (rayDirLength < 1e-12) throw new RangeError('sampleEquiAngular.rayDir must be non-zero');
  // Step 1: project light onto ray
  const deltaX = lightPos[0] - rayOrigin[0];
  const deltaY = lightPos[1] - rayOrigin[1];
  const deltaZ = lightPos[2] - rayOrigin[2];

  // t at closest approach: t_c = dot(lightPos - rayOrigin, rayDir)
  const tClosest =
    (deltaX * rayDir[0] + deltaY * rayDir[1] + deltaZ * rayDir[2]) / rayDirLength;

  // Step 2: perpendicular distance D = ||(lightPos - rayOrigin) - t_c * rayDir||
  const perpX = deltaX - tClosest * (rayDir[0] / rayDirLength);
  const perpY = deltaY - tClosest * (rayDir[1] / rayDirLength);
  const perpZ = deltaZ - tClosest * (rayDir[2] / rayDirLength);
  const D = Math.hypot(perpX, perpY, perpZ);

  // Degenerate: light is on the ray.  Fall back to uniform sampling on [0, L].
  if (D < 1e-6) {
    const L = degenerateFallbackLength;
    const tFallback = u * L;
    return { t: tFallback, pdf: 1 / L };
  }

  // Step 3: angular extents. Sample from t=0 to sceneTMax along the ray.
  const thetaMin = Math.atan2(-tClosest, D);
  const thetaMax = Math.atan2(sceneTMax - tClosest, D);
  const thetaRange = thetaMax - thetaMin;

  // Guard: near-zero angular range (light nearly perpendicular to the ray at
  // extreme distance).  Division by thetaRange would produce Inf/NaN PDF.
  if (thetaRange < 1e-8) {
    return { t: 0, pdf: 0 };
  }

  // Step 4: uniform sample in angular space
  const theta = thetaMin + u * thetaRange;
  const t = D * Math.tan(theta) + tClosest;

  // Step 5: PDF in t-space, evaluated at the CLAMPED t so the returned
  // (t, pdf) pair is self-consistent (Kulla & Fajardo 2012, §3).
  // p(t) = 1 / (D · thetaRange · (1 + ((t - t_closest)/D)²))
  //
  // History: the 2026-05-11 in-flight sweep (Tier D) flagged a clamp/PDF
  // mismatch here — the sampler previously returned `max(0, t)` while
  // computing the PDF on the unclamped `t`. Closed by commit d97e806
  // (M4 sweep, Item 10) on 2026-05-11; the clamp now applies to both the
  // returned sample and the ratio fed into the PDF, so `pdf == p(returned_t)`.
  // Verified-closed during the 2026-05-17 session.
  const tClamped = Math.max(0, t);
  const ratio = (tClamped - tClosest) / D;
  const pdf = 1 / (D * thetaRange * (1 + ratio * ratio));

  return { t: tClamped, pdf };
}
