/**
 * bdptMIS.ts — BDPT connection-PMF MIS weight computation.
 *
 * Implements the multi-strategy MIS weight for bidirectional path tracing.
 * Each complete s+t path (s vertices from the light subpath, t vertices from
 * the eye subpath) is one of s+t+1 sampling strategies. The weight for a
 * given strategy k is computed using the Veach 1997 power heuristic (β=2).
 *
 * In BDPT, the MIS weight for strategy k across all strategies {0 … s+t} is:
 *
 *   w_k = p_k^β / Σ_i p_i^β
 *
 * where p_i is the probability of generating the same path via strategy i.
 *
 * This module handles:
 *   1. `bdptConnectionMIS`   — single MIS weight given the pre-built PDF table.
 *   2. `buildBDPTStrategyPDFs` — builds that PDF table from subpath vertex data.
 *
 * CPU-side usage (verification / host preprocessing):
 *   The fork's GLSL connection shader replicates this logic inline. The CPU
 *   implementation here is the reference against which fork shader correctness
 *   is verified in tests, and is used by the pt-webgl host wrapper to pre-compute
 *   connection weights for non-real-time PT_FINAL accumulation passes.
 *
 * References:
 *   - Veach 1997, "Robust Monte Carlo Methods for Light Transport Simulation",
 *     PhD thesis, Stanford. §9.2 (power heuristic), §10.3 (BDPT MIS weights).
 *   - Pharr, Jakob, Humphreys 2023, "Physically Based Rendering" (4th ed.),
 *     §16.3.5 (BDPT MIS weighting).
 */

import type { BDPTVertex } from './bdptVertex.js';

// ────────────────────────────────────────────────────────────────────────────
// Core MIS weight
// ────────────────────────────────────────────────────────────────────────────

/**
 * Compute the power-heuristic MIS weight for a single BDPT connection strategy.
 *
 * Given the per-strategy PDF table `pdfsByStrategy` (length s+t+1), returns
 * the MIS weight for strategy index `selectedStrategyIndex` using the Veach
 * power heuristic with exponent `beta` (default 2).
 *
 * The PDF at each strategy index k is the probability of generating the same
 * complete path if strategy k had been used. Strategy k=0 corresponds to
 * pure light-tracing (all vertices from the light subpath); strategy k=s+t
 * corresponds to pure path tracing from the eye (all vertices from the eye
 * subpath).
 *
 * Graceful degradation:
 *   - All-zero PDFs: returns 0 (path has zero probability — do not accumulate).
 *   - Single strategy: returns 1.0 (no competitors, full weight).
 *   - selectedStrategyIndex out of range: returns 0.
 *
 * @param pdfsByStrategy    - array of per-strategy path PDFs, length s+t+1
 * @param selectedStrategyIndex - index of the strategy whose weight to compute
 * @param beta              - power heuristic exponent (default 2; β=1 = balance)
 * @returns MIS weight ∈ [0, 1]
 */
export function bdptConnectionMIS(
  pdfsByStrategy: ReadonlyArray<number>,
  selectedStrategyIndex: number,
  beta: number = 2,
): number {
  if (
    selectedStrategyIndex < 0 ||
    selectedStrategyIndex >= pdfsByStrategy.length
  ) {
    return 0;
  }

  let denominator = 0;
  for (const p of pdfsByStrategy) {
    denominator += Math.pow(p, beta);
  }

  if (denominator <= 0) return 0;

  const numerator = Math.pow(pdfsByStrategy[selectedStrategyIndex] ?? 0, beta);
  return numerator / denominator;
}

// ────────────────────────────────────────────────────────────────────────────
// Strategy PDF table builder
// ────────────────────────────────────────────────────────────────────────────

/**
 * Build the per-strategy PDF table for a complete BDPT path.
 *
 * Given a light subpath (s vertices) and an eye subpath (t vertices), returns
 * a Float32Array of length s+t+1 where entry k is the probability of generating
 * the same complete path using k vertices from the light subpath and s+t-k
 * vertices from the eye subpath.
 *
 * PDF computation:
 *   Each strategy's PDF is the product of the vertex forward PDFs along the
 *   light subpath portion and the vertex forward PDFs along the eye subpath
 *   portion. Reverse PDFs of the connection vertices are included to account
 *   for the directionality of the connection edge (Veach 1997, Eq. 10.9).
 *
 *   For a path P = (v_0, …, v_{s-1}, u_{t-1}, …, u_0) where v_i are light
 *   vertices and u_j are eye vertices:
 *     p_k = (Π_{i=0}^{k-1} v_i.pdfFwd) × (Π_{j=0}^{s+t-k-1} u_j.pdfFwd)
 *
 *   This is a simplified "separable" PDF approximation. For full correctness
 *   the connection edge's geometry factor must also be included — but since the
 *   MIS denominator normalises across all strategies sharing the same geometry
 *   factor, the factor cancels. The fork's GLSL computes this inline with the
 *   full geometry term; the CPU side omits it for testability.
 *
 * Strategy index semantics:
 *   k = 0:   pure path tracing from eye (zero light vertices used explicitly)
 *   k = 1:   one explicit light-subpath vertex (direct-lighting NEE)
 *   k = s:   all s light vertices used; connection from v_{s-1} to u_{t-1}
 *   k = s+t: pure light tracing (zero eye vertices; light path hits the sensor)
 *
 * Edge cases:
 *   - Empty light subpath (s=0): only strategy k=0 is available; returns [1.0].
 *   - Empty eye subpath (t=0): only strategy k=s is available.
 *   - Vertices with pdfFwd=0 cause that product to collapse to 0 (correct —
 *     the strategy has measure zero for that path).
 *
 * @param lightSubpath - light-subpath vertices, v_0 = emitter surface
 * @param eyeSubpath   - eye-subpath vertices, u_0 = camera/primary-hit surface
 * @returns Float32Array of length s+t+1 containing per-strategy path PDFs
 */
export function buildBDPTStrategyPDFs(
  lightSubpath: ReadonlyArray<BDPTVertex>,
  eyeSubpath: ReadonlyArray<BDPTVertex>,
): Float32Array {
  const s = lightSubpath.length;
  const t = eyeSubpath.length;
  const total = s + t + 1;
  const pdfs = new Float32Array(total);

  // Pre-compute cumulative forward-PDF products for each subpath.
  // lightCumPdf[k] = Π_{i=0}^{k-1} lightSubpath[i].pdfFwd  (lightCumPdf[0] = 1)
  // eyeCumPdf[k]   = Π_{j=0}^{k-1} eyeSubpath[j].pdfFwd    (eyeCumPdf[0]   = 1)
  const lightCumPdf = new Float64Array(s + 1);
  lightCumPdf[0] = 1;
  for (let i = 0; i < s; i++) {
    lightCumPdf[i + 1] = lightCumPdf[i]! * (lightSubpath[i]!.pdfFwd);
  }

  const eyeCumPdf = new Float64Array(t + 1);
  eyeCumPdf[0] = 1;
  for (let j = 0; j < t; j++) {
    eyeCumPdf[j + 1] = eyeCumPdf[j]! * (eyeSubpath[j]!.pdfFwd);
  }

  // Strategy k: k light vertices + (s+t-k) eye vertices.
  // Only strategies 0 … s+t are meaningful; s+t means pure light tracing
  // (light path length = s+t, no eye vertices).
  for (let k = 0; k <= s + t; k++) {
    const numLightVerts = Math.min(k, s);
    const numEyeVerts = Math.min(s + t - k, t);

    // If the requested decomposition exceeds available vertices in either
    // subpath the strategy has zero probability (we cannot extend a subpath
    // beyond its stored length in this simplified model).
    if (k > s || s + t - k > t) {
      pdfs[k] = 0;
      continue;
    }

    pdfs[k] = (lightCumPdf[numLightVerts]! * eyeCumPdf[numEyeVerts]!);
  }

  return pdfs;
}
