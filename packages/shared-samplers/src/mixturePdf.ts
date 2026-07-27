import { normalizedPairFirst, requireFinite, requireNonNegative, requirePositive } from './numericGuards.js';

/**
 * mixturePdf.ts — Multiple Importance Sampling (MIS) combiners.
 *
 * Implements the balance and power heuristics from Veach & Guibas 1995
 * ("Optimally Combining Sampling Techniques for Monte Carlo Rendering",
 * SIGGRAPH 95), plus a general mixture PDF evaluator for N strategies.
 *
 * These public CPU-side implementations support host algorithms and numerical
 * reference tests. Render backends keep equivalent shader-language math at
 * their actual transport sites; they do not import this JavaScript helper.
 *
 * References:
 *   - Veach & Guibas 1995, "Optimally Combining Sampling Techniques for
 *     Monte Carlo Rendering", SIGGRAPH.
 *   - Pharr, Jakob, Humphreys 2023, "Physically Based Rendering: From
 *     Theory to Implementation" (4th ed.), §13.10 (MIS).
 */

/**
 * Multiple Importance Sampling (MIS) balance heuristic.
 *
 * Given two PDFs evaluated for the same direction, returns the MIS weight
 * for sampling strategy 1 (pdf1). The weight for strategy 2 is 1 - result.
 *
 *   w_balance(pdf1, pdf2) = pdf1 / (pdf1 + pdf2)
 *
 * Numerically stable: returns 0.5 when both PDFs are 0.
 *
 * @param pdf1 - PDF of strategy 1 at the sampled direction (≥ 0)
 * @param pdf2 - PDF of strategy 2 at the sampled direction (≥ 0)
 * @returns weight ∈ [0, 1] for strategy 1
 */
export function balanceHeuristic(pdf1: number, pdf2: number): number {
  requireNonNegative(pdf1, 'balanceHeuristic.pdf1');
  requireNonNegative(pdf2, 'balanceHeuristic.pdf2');
  return normalizedPairFirst(pdf1, pdf2);
}

/**
 * Power heuristic (Veach 1997, exponent β).
 *
 * Raises each PDF to the power β before computing the balance heuristic.
 * β = 2 is the standard choice and typically outperforms the balance
 * heuristic for direct-lighting NEE + BSDF MIS in path tracing.
 *
 *   w_power(pdf1, pdf2, β) = pdf1^β / (pdf1^β + pdf2^β)
 *
 * Numerically stable: returns 0.5 when both terms are 0.
 *
 * @param pdf1 - PDF of strategy 1 at the sampled direction (≥ 0)
 * @param pdf2 - PDF of strategy 2 at the sampled direction (≥ 0)
 * @param beta - exponent (default 2; β=1 reduces to balanceHeuristic)
 * @returns weight ∈ [0, 1] for strategy 1
 */
export function powerHeuristic(pdf1: number, pdf2: number, beta: number = 2): number {
  requireNonNegative(pdf1, 'powerHeuristic.pdf1');
  requireNonNegative(pdf2, 'powerHeuristic.pdf2');
  requirePositive(beta, 'powerHeuristic.beta');
  if (pdf1 === 0 && pdf2 === 0) return 0.5;
  const scale = Math.max(pdf1, pdf2);
  const p1b = Math.pow(pdf1 / scale, beta);
  const p2b = Math.pow(pdf2 / scale, beta);
  return normalizedPairFirst(p1b, p2b);
}

/**
 * Mixture PDF combiner for N sampling strategies.
 *
 * Given N strategies with per-strategy selection probabilities and PDF
 * evaluations at the sampled direction, returns the combined (mixture) PDF:
 *
 *   p_mixture = Σ_i  probabilities[i] × pdfs[i]
 *
 * This is the denominator of the MIS weight in the one-sample model when
 * any of the N strategies could have generated the sample. The selection
 * probabilities should sum to 1. The function intentionally does not renormalize
 * them, so callers with unnormalized weights receive the weighted sum they passed.
 *
 * Typical usage in PT direct lighting:
 *   strategies: [BSDF sampling, env-map sampling, light-tree sampling]
 *   probabilities: [pBSDF, pEnv, pLight]  (sum = 1)
 *   pdfs: [bsdfPdf, envPdf, lightPdf]  at the chosen direction
 *
 * @param probabilities - selection probability for each strategy (length N, sum ≈ 1)
 * @param pdfs          - PDF of each strategy at the sampled direction (length N)
 * @returns combined mixture PDF value (≥ 0)
 * @throws if arrays have different lengths or are empty
 * @throws if all probabilities are zero — a MIS weight denominator of 0 indicates
 *   no active sampling strategy, which is a caller logic error. At least one strategy
 *   must have a non-zero selection probability.
 */
export function mixturePdf(
  probabilities: readonly number[],
  pdfs: readonly number[],
): number {
  if (probabilities.length === 0 || pdfs.length === 0) {
    throw new Error('mixturePdf: arrays must not be empty');
  }
  if (probabilities.length !== pdfs.length) {
    throw new Error('mixturePdf: probabilities and pdfs must have the same length');
  }
  let probSum = 0;
  for (let i = 0; i < probabilities.length; i++) {
    const probability = probabilities[i] ?? 0;
    requireNonNegative(probability, `mixturePdf.probabilities[${i}]`);
    probSum += probability;
  }
  if (probSum === 0) {
    throw new Error(
      'mixturePdf: MIS strategy weights cannot all be zero — at least one strategy must have non-zero probability. ' +
      'A zero-denominator mixture PDF would cause division by zero in MIS weight computation.',
    );
  }
  let result = 0;
  for (let i = 0; i < probabilities.length; i++) {
    const probability = probabilities[i] ?? 0;
    const pdf = pdfs[i] ?? 0;
    requireNonNegative(pdf, `mixturePdf.pdfs[${i}]`);
    result += probability * pdf;
    requireFinite(result, 'mixturePdf.result');
  }
  return result;
}
