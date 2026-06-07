/**
 * denoiserFidelityOracle.test.ts — objective denoiser-fidelity oracle (Phase 0-F).
 *
 * The maintainer cannot rely on visual review, so every CPU-runnable denoiser in
 * @vitrum/shared-denoisers must be validated by an OBJECTIVE oracle, never an
 * eyeball. This file is that oracle. It asserts two properties for each denoiser
 * under test, plus a self-discrimination guard that proves the oracle actually
 * REJECTS known-bad denoisers (a passthrough and a biased filter). If the oracle
 * ever passes a passthrough or a constant-bias filter it is broken — the
 * discrimination block below pins exactly that.
 *
 * ── The two oracle properties ────────────────────────────────────────────────
 *
 *  P1. IDENTITY ON CONVERGED INPUT.  denoise(clean) ≈ clean.
 *      A good edge-aware denoiser must NOT blur an already-noise-free image.
 *      We feed a smooth, noise-free reference that lies in the denoiser's
 *      representable space (with matching aux buffers — normals / depth /
 *      albedo / position features) and assert the output is within a tight
 *      tolerance of the input.
 *
 *  P2. VARIANCE REDUCTION WITHOUT BIAS.  Take the same clean reference, add
 *      seeded zero-mean noise to make a noisy input, denoise it, and assert:
 *        (a) MSE-vs-clean DROPS substantially vs the noisy input (it denoises),
 *        (b) the MEAN over a flat region is preserved (no energy bias):
 *            mean(denoise(noisy)) ≈ mean(clean)  over a flat patch.
 *
 * ── What is / isn't covered on CPU ───────────────────────────────────────────
 *
 *  • BMFR (Koskela 2019, `bmfrFitBlock`) — COVERED. It is a genuine pure-CPU
 *    denoise filter: noisy color + per-pixel features → reconstructed color.
 *  • SVGF-real (`svgfRealCpu.ts`) — the CPU emulation only covers reprojection,
 *    variance-from-moments, and the 7×7 spatial *variance* fallback. The actual
 *    edge-aware à-trous COLOR filter is WGSL-only (no CPU port exists), so a true
 *    `denoise(noisy)→clean` color filter for SVGF cannot run on CPU. We instead
 *    cover the SVGF temporal-accumulation path's variance-reduction-without-bias
 *    property directly (it is the temporal half of SVGF and IS CPU-runnable),
 *    via the same oracle predicates. See the SVGF block at the bottom.
 *  • OIDN (`oidnBridge.ts`) — NOT COVERED on CPU. It requires `onnxruntime-web`
 *    (an optional peerDependency that is NOT installed in this repo) plus an
 *    ONNX UNet model file with trained weights. Without weights there is no
 *    denoiser to run. Its CPU-runnable layout transforms (_hwcToNchw round-trip)
 *    are already pinned in oidnBridge.test.ts; the inference itself is GPU/WASM-
 *    runtime-dependent and out of scope for a weightless CPU oracle.
 */

import { describe, it, expect } from 'vitest';
import {
  BMFR_FEATURE_COUNT,
  bmfrFeatureRow,
  bmfrFitBlock,
} from '../src/bmfrRegression.js';
import { svgfReprojCPU } from '../src/svgfRealCpu.js';

// ── Seeded PRNG ──────────────────────────────────────────────────────────────
// Math.random is banned in radiometric code; the oracle uses a small seeded LCG
// so the noisy inputs are byte-reproducible across runs.
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

/** Zero-mean uniform noise in [-amp, +amp] from a seeded generator. */
function zeroMeanNoise(rand: () => number, amp: number): number {
  return (rand() - 0.5) * 2 * amp;
}

// ── A denoiser, abstracted ───────────────────────────────────────────────────
// The oracle is parameterised by a DenoiseFn so the SAME predicates run against
// the real denoiser AND the deliberately-broken variants. A DenoiseFn maps a
// flat HxWx3 RGB color buffer (+ the frame's aux buffers) to a denoised buffer.
interface DenoiseFrame {
  readonly width: number;
  readonly height: number;
  /** Noisy / input RGB, flat row-major interleaved, length W*H*3. */
  readonly color: Float32Array;
  /** Per-pixel BMFR feature rows, length W*H*BMFR_FEATURE_COUNT. */
  readonly features: Float32Array;
}
type DenoiseFn = (frame: DenoiseFrame) => Float32Array;

// ── Synthetic frame builder ──────────────────────────────────────────────────
//
// We build a single-plane frame: one normal, one depth, world position varying
// smoothly across the screen. The CLEAN reference color is a LOW-ORDER
// POLYNOMIAL of the block-local position — i.e. it lies inside BMFR's feature
// span [1, p.xyz, n.xyz, p².xyz]. This is the principled "converged" input for
// a regression denoiser: a signal the basis CAN represent must be recovered to
// (near) machine precision. (A signal OUTSIDE the basis would fail identity not
// because the denoiser is broken but because the basis is finite — that would
// test the basis, not the denoiser. The flat-region mean check in P2 is taken
// over a sub-patch so the energy/bias claim is basis-independent.)
const PLANE_NORMAL: [number, number, number] = [0, 0, 1];

interface SyntheticFrame extends DenoiseFrame {
  /** Noise-free ground-truth color (== `color` for the clean frame). */
  readonly clean: Float32Array;
  /** Indices of pixels inside the mean/bias measurement sub-region. */
  readonly flatRegion: readonly number[];
  /** Ground-truth mean of the CLEAN signal over the sub-region, per channel. */
  readonly flatMean: readonly [number, number, number];
}

/**
 * Build a clean reference frame. `color` == `clean` here (no noise added).
 *
 * The clean signal is a SINGLE continuous low-order polynomial of block-local
 * position over the WHOLE frame — no discontinuities — so it lies entirely
 * inside BMFR's [1, p.xyz, n.xyz, p².xyz] span and identity can hold tightly.
 * (Injecting a hard-edged constant patch would put a step discontinuity in the
 * signal, which a single global least-squares fit CANNOT represent — that would
 * fail identity because of the basis, not the denoiser. We avoid that trap.)
 *
 * The no-bias measurement region is a centered sub-patch; its ground-truth mean
 * is just the mean of the smooth clean signal there. Energy-preservation means
 * mean(denoise(noisy)) over the region ≈ mean(clean) over the region — that is
 * basis-independent and is what "no bias" actually asserts.
 */
function buildCleanFrame(W: number, H: number): SyntheticFrame {
  const px = W * H;
  const features = new Float32Array(px * BMFR_FEATURE_COUNT);
  const clean = new Float32Array(px * 3);

  // Block-local position spans roughly [-1, 1] in x/y so the squared feature
  // terms are well-conditioned (the caller-normalisation BMFR documents).
  const halfW = (W - 1) / 2;
  const halfH = (H - 1) / 2;

  // Centered sub-region over which the no-bias mean is measured.
  const flatX0 = Math.floor(W * 0.35);
  const flatX1 = Math.floor(W * 0.65);
  const flatY0 = Math.floor(H * 0.35);
  const flatY1 = Math.floor(H * 0.65);
  const flatRegion: number[] = [];

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const pi = y * W + x;
      const lx = (x - halfW) / halfW; // ~[-1, 1]
      const ly = (y - halfH) / halfH; // ~[-1, 1]
      const row = new Float32Array(BMFR_FEATURE_COUNT);
      bmfrFeatureRow([lx, ly, 0], PLANE_NORMAL, row);
      features.set(row, pi * BMFR_FEATURE_COUNT);

      // One smooth, continuous polynomial everywhere (in BMFR's span):
      //   const + linear + quadratic, different per channel.
      clean[pi * 3]     = 0.50 + 0.15 * lx - 0.08 * ly + 0.05 * lx * lx;
      clean[pi * 3 + 1] = 0.40 + 0.10 * ly + 0.04 * ly * ly;
      clean[pi * 3 + 2] = 0.30 - 0.12 * lx + 0.06 * lx * lx;

      if (x >= flatX0 && x <= flatX1 && y >= flatY0 && y <= flatY1) {
        flatRegion.push(pi);
      }
    }
  }

  // Ground-truth mean of the clean signal over the measurement region.
  let mr = 0, mg = 0, mb = 0;
  for (const pi of flatRegion) {
    mr += clean[pi * 3] ?? 0;
    mg += clean[pi * 3 + 1] ?? 0;
    mb += clean[pi * 3 + 2] ?? 0;
  }
  const fn = flatRegion.length;
  const flatMean: [number, number, number] = [mr / fn, mg / fn, mb / fn];

  return {
    width: W,
    height: H,
    color: clean.slice(), // clean input == clean reference
    features,
    clean,
    flatRegion,
    flatMean,
  };
}

/** Add seeded zero-mean noise to a clean frame's color, returning a new frame. */
function addNoise(frame: SyntheticFrame, seed: number, amp: number): SyntheticFrame {
  const rand = lcg(seed);
  const noisy = frame.clean.slice();
  for (let i = 0; i < noisy.length; i++) {
    noisy[i] = (noisy[i] ?? 0) + zeroMeanNoise(rand, amp);
  }
  return { ...frame, color: noisy };
}

// ── Metrics ──────────────────────────────────────────────────────────────────

/** Mean-squared error of `a` vs `b` over the full RGB buffer. */
function mse(a: Float32Array, b: Float32Array): number {
  let acc = 0;
  for (let i = 0; i < a.length; i++) {
    const d = (a[i] ?? 0) - (b[i] ?? 0);
    acc += d * d;
  }
  return acc / a.length;
}

/** Max absolute per-element deviation of `a` vs `b`. */
function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs((a[i] ?? 0) - (b[i] ?? 0));
    if (d > m) m = d;
  }
  return m;
}

/** Per-channel mean over a set of pixel indices. */
function regionMean(buf: Float32Array, region: readonly number[]): [number, number, number] {
  let r = 0, g = 0, b = 0;
  for (const pi of region) {
    r += buf[pi * 3] ?? 0;
    g += buf[pi * 3 + 1] ?? 0;
    b += buf[pi * 3 + 2] ?? 0;
  }
  const n = region.length;
  return [r / n, g / n, b / n];
}

// ── The oracle predicates ────────────────────────────────────────────────────
//
// Returned as booleans (not bare expect()s) so the discrimination block can
// assert that the SAME predicates REJECT the known-bad denoisers. This is what
// makes the oracle real: it must FAIL on a passthrough and on a biased filter.

interface OracleTolerances {
  /** P1: max-abs identity tolerance (clean must come back clean). */
  readonly identityMaxAbs: number;
  /** P2a: required MSE-reduction factor (denoised MSE < noisy MSE * factor). */
  readonly varianceReductionFactor: number;
  /** P2b: max-abs region-mean BIAS tolerance, averaged over realizations. */
  readonly meanBiasTol: number;
}

interface OracleResult {
  readonly identityPass: boolean;
  readonly identityMaxAbs: number;
  readonly varianceReductionPass: boolean;
  readonly noisyMse: number;
  readonly denoisedMse: number;
  readonly noBiasPass: boolean;
  readonly meanBiasMaxAbs: number;
}

/**
 * Run BOTH oracle properties against a denoiser and return per-property verdicts.
 * Pure / side-effect-free so callers can assert pass (real denoiser) or assert
 * !pass (known-bad denoiser) on the exact same logic.
 *
 * `noisyFrames` is a list of K INDEPENDENT seeded noise realizations of the same
 * clean reference. Why K and not one:
 *   • Variance reduction (P2a) is a per-realization property — evaluated on the
 *     first realization (and it holds on every one).
 *   • BIAS (P2b) is a property of the EXPECTATION over noise: a single noisy
 *     frame's region-mean carries O(amp/√N) sampling noise (~2e-2 here) that is
 *     NOT bias — averaging the denoised region-mean over K realizations cancels
 *     that zero-mean sampling noise and exposes any SYSTEMATIC energy shift. An
 *     unbiased fit → bias → 0 as K grows; a constant-offset filter stays pinned
 *     at the offset. (Empirically: unbiased ≈ 4.5e-3, +0.05-biased ≈ 5.45e-2 at
 *     K=24 — the meanBiasTol of 1.5e-2 cleanly separates them.)
 */
function evaluateDenoiser(
  denoise: DenoiseFn,
  clean: SyntheticFrame,
  noisyFrames: readonly SyntheticFrame[],
  tol: OracleTolerances,
): OracleResult {
  // P1 — identity on converged input.
  const onClean = denoise(clean);
  const identityMaxAbs = maxAbsDiff(onClean, clean.clean);
  const identityPass = identityMaxAbs <= tol.identityMaxAbs;

  // P2a — variance reduction (per-realization; use the first realization).
  const first = noisyFrames[0]!;
  const onNoisy = denoise(first);
  const noisyMse = mse(first.color, clean.clean);
  const denoisedMse = mse(onNoisy, clean.clean);
  const varianceReductionPass = denoisedMse < noisyMse * tol.varianceReductionFactor;

  // P2b — no bias: average the denoised region-mean over all K realizations,
  // then compare to the clean region-mean. The zero-mean sampling noise cancels;
  // only a systematic shift survives.
  const acc: [number, number, number] = [0, 0, 0];
  for (const nf of noisyFrames) {
    const out = denoise(nf);
    const m = regionMean(out, nf.flatRegion);
    acc[0] += m[0]; acc[1] += m[1]; acc[2] += m[2];
  }
  const k = noisyFrames.length;
  const denMean: [number, number, number] = [acc[0] / k, acc[1] / k, acc[2] / k];
  const meanBiasMaxAbs = Math.max(
    Math.abs(denMean[0] - clean.flatMean[0]),
    Math.abs(denMean[1] - clean.flatMean[1]),
    Math.abs(denMean[2] - clean.flatMean[2]),
  );
  const noBiasPass = meanBiasMaxAbs <= tol.meanBiasTol;

  return {
    identityPass, identityMaxAbs,
    varianceReductionPass, noisyMse, denoisedMse,
    noBiasPass, meanBiasMaxAbs,
  };
}

/** Build K independent seeded noise realizations of a clean frame. */
function makeRealizations(
  clean: SyntheticFrame, baseSeed: number, amp: number, k: number,
): SyntheticFrame[] {
  const out: SyntheticFrame[] = [];
  for (let i = 0; i < k; i++) out.push(addNoise(clean, baseSeed + i * 7919, amp));
  return out;
}

// ── Real denoiser under test: BMFR (Koskela 2019) ────────────────────────────
//
// BMFR is block-local; the synthetic frames are small enough to treat the whole
// frame as ONE block (mirrors bmfrRegression.test.ts's "block" construction).
// Albedo is unity here (clean color is already the lighting signal), so no
// demodulate/remodulate is needed — the demodulation helpers are exercised
// separately in albedoModulation usage and svgfReal tests.
function bmfrDenoise(lambda: number): DenoiseFn {
  return (frame) => {
    const pixelCount = frame.width * frame.height;
    return bmfrFitBlock(frame.features, frame.color.slice(), pixelCount, lambda);
  };
}

// ── Known-bad denoisers (for the discrimination proof) ───────────────────────

/** Passthrough: returns the input unchanged → must FAIL variance-reduction. */
const passthroughDenoise: DenoiseFn = (frame) => frame.color.slice();

/** Biased: adds a constant offset → must FAIL the no-bias (mean) check. */
function biasedDenoise(offset: number): DenoiseFn {
  // Built on top of a REAL denoiser (BMFR) so it DOES reduce variance and DOES
  // preserve identity-shape — it fails ONLY the bias check. This is the sharp
  // test: a filter that denoises well but silently shifts energy.
  const base = bmfrDenoise(1e-4);
  return (frame) => {
    const out = base(frame);
    for (let i = 0; i < out.length; i++) out[i] = (out[i] ?? 0) + offset;
    return out;
  };
}

// =============================================================================
// BMFR — the real CPU denoiser
// =============================================================================

// Shared oracle config (BMFR + discrimination share the same frame + tolerances).
const ORACLE_W = 16, ORACLE_H = 16;
const ORACLE_NOISE_AMP = 0.30;
const ORACLE_K = 24; // noise realizations for the bias estimate
const ORACLE_BASE_SEED = 0xBEEF;

// Tolerances + justification (empirics measured on this exact frame/seed set):
//   identityMaxAbs = 5e-3 — BMFR fits a low-order polynomial exactly in f64
//     accumulation; the only residual is the λ-regularisation pull (λ=1e-4) plus
//     f32 round-off in the output. Empirically the clean→clean max-abs is < 2e-3;
//     5e-3 is a snug ceiling that still rejects any real blur (the +0.05 bias
//     variant deviates by 0.05 ≫ tol; a box blur of this gradient deviates >2e-2).
//   varianceReductionFactor = 0.30 — the regression must cut MSE-vs-clean to
//     under 30% of the noisy input's. A passthrough (factor 1.0) fails this hard.
//   meanBiasTol = 1.5e-2 — the no-bias check averages the denoised region-mean
//     over K=24 seeded realizations, cancelling the O(amp/√N)≈2e-2 per-frame
//     sampling noise. Measured: unbiased BMFR bias ≈ 4.5e-3, +0.05-biased ≈
//     5.45e-2 — 1.5e-2 sits ~3.3× above the unbiased residual and ~3.6× below
//     the biased shift, cleanly separating them.
const ORACLE_TOL: OracleTolerances = {
  identityMaxAbs: 5e-3,
  varianceReductionFactor: 0.30,
  meanBiasTol: 1.5e-2,
};

describe('Denoiser fidelity oracle — BMFR (Koskela 2019, bmfrFitBlock)', () => {
  // 16×16 frame, treated as a single BMFR block. The polynomial clean signal is
  // exactly representable by BMFR's [1, p, n, p²] basis.
  const clean = buildCleanFrame(ORACLE_W, ORACLE_H);
  const noisyFrames = makeRealizations(clean, ORACLE_BASE_SEED, ORACLE_NOISE_AMP, ORACLE_K);
  const tol = ORACLE_TOL;

  const result = evaluateDenoiser(bmfrDenoise(1e-4), clean, noisyFrames, tol);

  it('P1 — identity on converged input: denoise(clean) ≈ clean', () => {
    expect(result.identityMaxAbs).toBeLessThanOrEqual(tol.identityMaxAbs);
    expect(result.identityPass).toBe(true);
  });

  it('P2a — variance reduction: denoise(noisy) MSE drops substantially vs noisy', () => {
    // Sanity: the noisy input genuinely has high error (the test is meaningful).
    expect(result.noisyMse).toBeGreaterThan(1e-2);
    expect(result.denoisedMse).toBeLessThan(result.noisyMse * tol.varianceReductionFactor);
    expect(result.varianceReductionPass).toBe(true);
  });

  it('P2b — no bias: flat-region mean is preserved (no energy shift)', () => {
    expect(result.meanBiasMaxAbs).toBeLessThanOrEqual(tol.meanBiasTol);
    expect(result.noBiasPass).toBe(true);
  });
});

// =============================================================================
// DISCRIMINATION PROOF — the oracle MUST reject known-bad denoisers.
// =============================================================================
//
// This is the acceptance gate for the oracle ITSELF. If these tests ever go
// green-by-passing (i.e. the oracle accepts a passthrough or a biased filter),
// the oracle is decoration, not validation. We assert the oracle's predicates
// REJECT each bad variant on the property it violates.

describe('Denoiser fidelity oracle — discrimination (rejects known-bad denoisers)', () => {
  const clean = buildCleanFrame(ORACLE_W, ORACLE_H);
  const noisyFrames = makeRealizations(clean, ORACLE_BASE_SEED, ORACLE_NOISE_AMP, ORACLE_K);
  const tol = ORACLE_TOL;

  it('REJECTS a passthrough denoiser (fails variance reduction)', () => {
    const r = evaluateDenoiser(passthroughDenoise, clean, noisyFrames, tol);
    // A passthrough returns the input untouched: identity trivially holds, but
    // denoised MSE == noisy MSE, so variance-reduction MUST fail.
    expect(r.identityPass).toBe(true); // it does pass identity (no-op on clean)
    expect(r.varianceReductionPass).toBe(false); // ← the oracle catches it here
    expect(r.denoisedMse).toBeCloseTo(r.noisyMse, 5);
  });

  it('REJECTS a biased denoiser (fails the no-bias mean check)', () => {
    const r = evaluateDenoiser(biasedDenoise(0.05), clean, noisyFrames, tol);
    // The biased variant is a REAL (BMFR) denoiser + a constant +0.05 offset:
    // it reduces variance fine, but shifts the region mean by ~0.05 ≫ tol.
    expect(r.varianceReductionPass).toBe(true); // it does reduce variance...
    expect(r.identityPass).toBe(false);         // ...but breaks identity (+0.05)
    expect(r.noBiasPass).toBe(false);           // ← the oracle catches the bias
    expect(r.meanBiasMaxAbs).toBeGreaterThan(0.03);
  });

  it('a biased-but-converged filter is caught even though it "denoises"', () => {
    // Sharpest case: the bias variant passes BOTH "looks denoised" eyeball
    // proxies (low variance, smooth) yet the oracle still rejects it. This is
    // the whole point — energy bias is invisible to the eye, visible to math.
    const r = evaluateDenoiser(biasedDenoise(0.05), clean, noisyFrames, tol);
    const overallAccept = r.identityPass && r.varianceReductionPass && r.noBiasPass;
    expect(overallAccept).toBe(false);
  });
});

// =============================================================================
// SVGF-real — temporal-accumulation half (CPU-runnable), same oracle predicates.
// =============================================================================
//
// The SVGF edge-aware à-trous COLOR filter is WGSL-only; only the temporal
// reprojection + EMA accumulation (svgfReprojCPU) runs on CPU. Temporal
// accumulation IS a denoiser: across frames of a STATIC scene with zero motion,
// the per-pixel EMA averages out zero-mean per-frame noise (variance reduction)
// without shifting the converged mean (no bias). We assert exactly P2a + P2b on
// that path. P1 (identity) for SVGF-temporal is already pinned in svgfReal.test
// ("converge blended color toward input when input is constant").
describe('Denoiser fidelity oracle — SVGF temporal accumulation (svgfReprojCPU)', () => {
  const W = 8, H = 8;
  const px = W * H;

  // Static-scene geometry: one depth, one normal, one object id, zero motion.
  const depth = new Float32Array(px).fill(2.0);
  const normal = new Float32Array(px * 3);
  for (let i = 0; i < px; i++) {
    normal[i * 3]     = PLANE_NORMAL[0] / 2 + 0.5;
    normal[i * 3 + 1] = PLANE_NORMAL[1] / 2 + 0.5;
    normal[i * 3 + 2] = PLANE_NORMAL[2] / 2 + 0.5;
  }
  const objId = new Uint32Array(px).fill(1);
  const motion = new Float32Array(px * 2).fill(0);

  // Converged (clean) per-pixel color: a smooth gradient.
  const clean = new Float32Array(px * 3);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const pi = y * W + x;
      clean[pi * 3]     = 0.5 + 0.2 * (x / (W - 1));
      clean[pi * 3 + 1] = 0.4;
      clean[pi * 3 + 2] = 0.3 - 0.1 * (y / (H - 1));
    }
  }

  // Accumulate N noisy frames (each = clean + fresh seeded zero-mean noise).
  const N_FRAMES = 64;
  const noiseAmp = 0.25;
  const rand = lcg(0xC0FFEE);

  // Explicit (unparameterised) typed-array annotations so reassigning the
  // svgfReprojCPU outputs (Float32Array<ArrayBufferLike>) typechecks under TS5.5.
  let prevColor: Float32Array = new Float32Array(px * 3);
  let historyIn: Uint32Array = new Uint32Array(px).fill(0);
  let momentsIn: Float32Array = new Float32Array(px * 2).fill(0);

  // Track the FIRST noisy frame's MSE-vs-clean as the "noisy input" baseline.
  let firstNoisyMse = 0;
  let lastAccum: Float32Array = new Float32Array(px * 3);

  for (let f = 0; f < N_FRAMES; f++) {
    const currColor = clean.slice();
    for (let i = 0; i < currColor.length; i++) {
      currColor[i] = (currColor[i] ?? 0) + zeroMeanNoise(rand, noiseAmp);
    }
    if (f === 0) firstNoisyMse = mse(currColor, clean);

    const res = svgfReprojCPU({
      currColor,
      prevColor,
      motionVec: motion,
      currDepth: depth,
      currNormal: normal,
      currObjId: objId,
      prevDepth: depth,
      prevNormal: normal,
      prevObjId: objId,
      historyLengthIn: historyIn,
      momentsIn,
      width: W,
      height: H,
    });

    prevColor = res.colorOut;
    historyIn = res.historyLengthOut;
    momentsIn = res.momentsOut;
    lastAccum = res.colorOut;
  }

  // Flat-region mean check uses the whole frame's gradient: instead of a flat
  // patch, the no-bias claim here is "the accumulated mean equals the clean
  // mean per pixel" — assert it as a global mean over all pixels (the gradient
  // mean is preserved because EMA of zero-mean noise is unbiased).
  const cleanMean = regionMean(clean, Array.from({ length: px }, (_, i) => i));
  const accumMean = regionMean(lastAccum, Array.from({ length: px }, (_, i) => i));

  it('P2a — variance reduction: accumulated MSE drops far below a single noisy frame', () => {
    const accumMse = mse(lastAccum, clean);
    // After 64 EMA frames (alphaMin floor ~0.05 → effective window ~20 frames),
    // residual MSE should be a small fraction of one frame's. Tolerance 0.30
    // mirrors the BMFR factor; empirically the ratio is < 0.10.
    expect(firstNoisyMse).toBeGreaterThan(1e-2);
    expect(accumMse).toBeLessThan(firstNoisyMse * 0.30);
  });

  it('P2b — no bias: accumulated global mean preserves the clean mean', () => {
    // EMA of a constant-plus-zero-mean-noise signal converges to the constant;
    // the per-pixel mean (and hence the global mean) must match clean within a
    // small residual. 1.5e-2 ceiling (same as BMFR's mean tol).
    expect(Math.abs(accumMean[0] - cleanMean[0])).toBeLessThanOrEqual(1.5e-2);
    expect(Math.abs(accumMean[1] - cleanMean[1])).toBeLessThanOrEqual(1.5e-2);
    expect(Math.abs(accumMean[2] - cleanMean[2])).toBeLessThanOrEqual(1.5e-2);
  });
});
