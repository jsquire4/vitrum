/**
 * bdptMIS.ts — BDPT MIS helpers: full Veach §10.3 (T2.H4).
 *
 * Complete strategy-PDF enumeration for a BDPT path of arbitrary length.
 * Uses the recursive ratio sweep from PBR4e Eq. 16.16 to compute all k+1
 * strategy PDFs in O(k) time. Handles specular vertices (zero-weight),
 * geometric term G(x↔y), and the camera/light endpoint corner cases.
 *
 * Removed 2026-05-18: the Sprint-10c `_partial` helpers were a single-
 * strategy MIS aid retained for fork-side dispatch (2-strategy case only).
 * No production consumers remained; the `_full` helpers below are the
 * canonical path.
 *
 * References:
 *   - Veach 1997, PhD thesis §9.2 (power heuristic), §10.3 (BDPT MIS weights),
 *     Algorithm 10.4 (strategy PDF enumeration).
 *   - Pharr et al. 2023, PBR 4th ed. §16.3.5, Eq. 16.16 (recursive ratio).
 *
 * @module bdptMIS
 */

import type { BDPTVertex } from './bdptVertex.js';

// ────────────────────────────────────────────────────────────────────────────
// Full Veach §10.3 implementation (T2.H4)
// ────────────────────────────────────────────────────────────────────────────

/**
 * Compute the geometry term G(xᵢ ↔ xⱼ) between two path vertices.
 *
 * ```
 * G(xᵢ ↔ xⱼ) = |cos θᵢ · cos θⱼ| / ‖xᵢ − xⱼ‖²
 * ```
 *
 * where θᵢ is the angle at xᵢ between the connection direction and the
 * shading normal, and likewise for θⱼ.
 *
 * Returns 0 when the two positions coincide (degenerate connection) or when
 * either cosine is 0 (backface / tangent incidence).
 *
 * Reference: Veach 1997 §8.3.2, Eq. 8.10.
 *
 * @param posI      - world-space position of vertex i [x, y, z]
 * @param normalI   - unit shading normal at vertex i [x, y, z]
 * @param posJ      - world-space position of vertex j [x, y, z]
 * @param normalJ   - unit shading normal at vertex j [x, y, z]
 * @returns geometry term ≥ 0
 */
export function geometricTermG(
  posI: readonly [number, number, number],
  normalI: readonly [number, number, number],
  posJ: readonly [number, number, number],
  normalJ: readonly [number, number, number],
): number {
  // Direction from i to j (unnormalized)
  const dx = posJ[0] - posI[0];
  const dy = posJ[1] - posI[1];
  const dz = posJ[2] - posI[2];
  const dist2 = dx * dx + dy * dy + dz * dz;
  if (dist2 <= 0) return 0;

  const invDist = 1 / Math.sqrt(dist2);

  // Unit direction i→j
  const wx = dx * invDist;
  const wy = dy * invDist;
  const wz = dz * invDist;

  // |cos θᵢ| = |normalI · w|, |cos θⱼ| = |normalJ · (−w)|
  const cosI = Math.abs(normalI[0] * wx + normalI[1] * wy + normalI[2] * wz);
  const cosJ = Math.abs(normalJ[0] * wx + normalJ[1] * wy + normalJ[2] * wz);

  return (cosI * cosJ) / dist2;
}

// ── BDPTFullVertex ────────────────────────────────────────────────────────────

/**
 * A single vertex for the full Veach §10.3 strategy enumeration.
 *
 * The full path is represented as a merged array `vertices[0..k]` where:
 *   - `vertices[0]` = light endpoint (emitter surface)
 *   - `vertices[k]` = camera endpoint (camera position or primary hit)
 *   - interior vertices carry BSDF hits from either subpath
 *
 * Index ordering follows Veach §10.3 Figure 10.4:
 *   light endpoint → scene bounces → camera endpoint
 *
 * All PDF fields are in **solid-angle measure** as required by the ratio sweep.
 * If your caller has area-measure PDFs at surface vertices, convert them with
 *   `pdfSA = pdfArea / G(xᵢ, xᵢ₊₁)`
 * before building this array.
 */
export interface BDPTFullVertex {
  /** World-space position [x, y, z]. */
  readonly position: readonly [number, number, number];
  /** Unit shading normal [x, y, z]. Required for G(x↔y) evaluation. */
  readonly normal: readonly [number, number, number];
  /**
   * Forward PDF (solid-angle) for sampling this vertex along the path's
   * forward direction (light→camera).
   */
  readonly pdfFwd: number;
  /**
   * Reverse PDF (solid-angle) for sampling this vertex in the reverse
   * direction (camera→light). Required by the recursive ratio sweep
   * (PBR4e Eq. 16.16).
   */
  readonly pdfRev: number;
  /**
   * True when this vertex lies on a specular (delta-function BSDF) surface.
   * Any strategy whose sweep passes through a specular vertex has weight 0 per
   * Veach §10.3.5 — the delta PDF cannot be evaluated by an explicit
   * connection from the other subpath.
   */
  readonly isSpecular: boolean;
}

// ── buildBDPTStrategyPDFs_full ────────────────────────────────────────────────

/**
 * Enumerate all Veach §10.3 strategy PDFs for a full BDPT path.
 *
 * For a path with `n = vertices.length` vertices (k = n−1 segments), there are
 * `n` strategies indexed by `s ∈ {0, …, n−1}` where `s` is the number of light
 * subpath vertices (s=0 = pure camera path, s=n−1 = pure light path).
 *
 * The reference strategy `selectedS` is the one actually used to construct the
 * path. Its probability is `pRef`. All other strategy probabilities are obtained
 * by the recursive ratio sweep (PBR4e Eq. 16.16 / Veach Algorithm 10.4):
 *
 * Left sweep (decrementing s): at each step, vertex v_s is "claimed" by the
 * camera subpath. Ratio:
 * ```
 *   p_{s−1} / p_s = pdfRev[s] / (pdfFwd[s] · G(v_{s−1} ↔ v_s))
 * ```
 *
 * Right sweep (incrementing s): vertex v_{s+1} is claimed by the light subpath.
 * Ratio:
 * ```
 *   p_{s+1} / p_s = pdfFwd[s+1] · G(v_s ↔ v_{s+1}) / pdfRev[s+1]
 * ```
 *
 * **Specular zero-weight rule (Veach §10.3.5):** if the vertex at the sweep
 * boundary is specular, the sweep breaks and all further strategies in that
 * direction remain zero.
 *
 * @param vertices    - merged path [v_0=light endpoint, …, v_{n-1}=camera endpoint]
 * @param selectedS   - index of the chosen strategy (0-based light vertex count)
 * @param pRef        - path probability of the selected strategy (must be > 0)
 * @returns Float64Array of per-strategy path PDFs, length `vertices.length`
 */
export function buildBDPTStrategyPDFs_full(
  vertices: ReadonlyArray<BDPTFullVertex>,
  selectedS: number,
  pRef: number,
): Float64Array {
  const n = vertices.length;
  if (n === 0) return new Float64Array(0);

  const pdfs = new Float64Array(n);
  pdfs[selectedS] = pRef;

  // ── Left sweep: p_{s−1} = p_s · pdfRev[s] / (pdfFwd[s] · G(v_{s−1}, v_s)) ──
  {
    let p = pRef;
    for (let s = selectedS; s > 0; s--) {
      const v     = vertices[s]!;
      const vPrev = vertices[s - 1]!;

      // Specular rule: the connection point between the two subpaths for
      // strategy s−1 is the edge (v_{s−1}, v_s). If either endpoint of that
      // edge is specular the strategy has zero probability — break the sweep.
      if (v.isSpecular || vPrev.isSpecular) break;

      const g = geometricTermG(vPrev.position, vPrev.normal, v.position, v.normal);
      if (g <= 0 || v.pdfFwd <= 0) break;

      p = p * (v.pdfRev / (v.pdfFwd * g));
      pdfs[s - 1] = p;
    }
  }

  // ── Right sweep: p_{s+1} = p_s · pdfFwd[s+1] · G(v_s, v_{s+1}) / pdfRev[s+1] ──
  {
    let p = pRef;
    for (let s = selectedS; s < n - 1; s++) {
      const v     = vertices[s]!;
      const vNext = vertices[s + 1]!;

      if (vNext.isSpecular || v.isSpecular) break;

      const g = geometricTermG(v.position, v.normal, vNext.position, vNext.normal);
      if (g <= 0 || vNext.pdfRev <= 0) break;

      p = p * ((vNext.pdfFwd * g) / vNext.pdfRev);
      pdfs[s + 1] = p;
    }
  }

  return pdfs;
}

// ── bdptConnectionMIS_full ────────────────────────────────────────────────────

/**
 * Compute the full Veach §10.3 power-heuristic MIS weight for one BDPT strategy.
 *
 * Given the per-strategy path-PDF vector from {@link buildBDPTStrategyPDFs_full},
 * returns the power-heuristic weight for `selectedS`:
 *
 * ```
 * w_s = p_s^β / Σᵢ p_i^β
 * ```
 *
 * with β=2 (Veach §9.2 recommended exponent). The standard BDPT renderer uses
 * one sample per strategy (nᵢ = 1 for all i), so the `nᵢ` factor is omitted.
 *
 * **Graceful degradation:**
 *   - All-zero PDFs: returns 0.
 *   - `selectedS` out of range: returns 0.
 *   - Selected strategy PDF is 0: returns 0.
 *
 * @param pdfsByStrategy - Float64Array from `buildBDPTStrategyPDFs_full`, length n
 * @param selectedS      - index of the strategy whose weight to compute
 * @param beta           - power heuristic exponent (default 2; β=1 = balance)
 * @returns MIS weight ∈ [0, 1]
 */
export function bdptConnectionMIS_full(
  pdfsByStrategy: ReadonlyArray<number> | Float64Array,
  selectedS: number,
  beta: number = 2,
): number {
  const len = pdfsByStrategy.length;
  if (selectedS < 0 || selectedS >= len) return 0;

  let denominator = 0;
  for (let i = 0; i < len; i++) {
    const p = pdfsByStrategy[i] ?? 0;
    if (p > 0) denominator += Math.pow(p, beta);
  }

  if (denominator <= 0) return 0;

  const p_s = pdfsByStrategy[selectedS] ?? 0;
  if (p_s <= 0) return 0;

  return Math.pow(p_s, beta) / denominator;
}
