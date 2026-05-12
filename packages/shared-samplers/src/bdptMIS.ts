/**
 * bdptMIS.ts — Single-strategy MIS aid for fork-side dispatch (Sprint 10c scaffold).
 *
 * `bdptConnectionMIS_partial` applies the Veach power heuristic (β=2 by default)
 * to a per-strategy PDF vector. `buildBDPTStrategyPDFs_partial` fills that vector
 * using a **simplified separable product** of forward PDFs along light/eye subpath
 * prefixes — see its JSDoc for exactly which indices are non-zero.
 *
 * These are partial helpers: they handle only the single-strategy case needed for
 * current fork-side dispatch wiring. They are **not** a full BDPT strategy PDF
 * enumeration for arbitrary connection topologies (Veach §10.3). See
 * `plan/sprint-bdpt-veach-full-future.md` for the full §10.3 enumeration spec.
 *
 * References:
 *   - Veach 1997, PhD thesis §9.2 (power heuristic), §10.3 (BDPT MIS weights).
 *   - Pharr et al. 2023, PBR 4th ed. §16.3.5.
 */

import type { BDPTVertex } from './bdptVertex.js';

// ────────────────────────────────────────────────────────────────────────────
// Core MIS weight
// ────────────────────────────────────────────────────────────────────────────

/**
 * Single-strategy MIS aid: compute the power-heuristic MIS weight for one BDPT
 * connection strategy.
 *
 * Given the per-strategy PDF table `pdfsByStrategy`, returns the MIS weight for
 * `selectedStrategyIndex`.  Typical use: the table from
 * {@link buildBDPTStrategyPDFs_partial} (mostly zero entries except boundary
 * cases `s=0` / `t=0` and the interior index `k=s` when both `s,t>0`).
 *
 * This is a partial helper — it does not enumerate all Veach §10.3 strategies.
 * See `plan/sprint-bdpt-veach-full-future.md` for the full strategy enumeration.
 *
 * Graceful degradation:
 *   - All-zero PDFs: returns 0 (path has zero probability — do not accumulate).
 *   - Single non-zero strategy: competitor weight concentrates correctly.
 *   - selectedStrategyIndex out of range: returns 0.
 *
 * @param pdfsByStrategy    - array of per-strategy path PDFs, length s+t+1
 * @param selectedStrategyIndex - index of the strategy whose weight to compute
 * @param beta              - power heuristic exponent (default 2; β=1 = balance)
 * @returns MIS weight ∈ [0, 1]
 */
export function bdptConnectionMIS_partial(
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
 * Build the per-strategy PDF table used by {@link bdptConnectionMIS_partial}.
 *
 * **Simplified / partial model (intentional):** for each k ∈ [0, s+t] we form
 * `p_k = lightCumPdf[min(k,s)] · eyeCumPdf[min(s+t−k, t)]` where each cumulative
 * array holds forward-PDF products along the corresponding subpath prefix.
 *
 * **Feasibility guard:** entries are set to zero when `k > s` **or** `k < s`
 * (equivalently `s + t − k > t`).  Therefore:
 *   - `s = 0`, `t > 0` ⇒ only `k = 0` may be non-zero (pure eye prefix).
 *   - `t = 0`, `s > 0` ⇒ only `k = s` may be non-zero (pure light prefix).
 *   - `s > 0` **and** `t > 0` ⇒ only `k = s` may be non-zero (full light prefix
 *     × full eye prefix).  Other indices are zero — this is a single-strategy
 *     MIS aid, not a full Veach BDPT strategy enumeration; extend here when
 *     fork-side connection PDFs cover all `k`.
 *
 * Reverse PDFs and geometry terms at the connection edge are omitted on the CPU
 * (see fork GLSL for the full factor). For the full §10.3 enumeration spec, see
 * `plan/sprint-bdpt-veach-full-future.md`.
 *
 * @param lightSubpath - light-subpath vertices, v_0 = emitter surface
 * @param eyeSubpath   - eye-subpath vertices, u_0 = camera/primary-hit surface
 * @returns Float32Array of length s+t+1 (mostly zeros except the cases above)
 */
export function buildBDPTStrategyPDFs_partial(
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
