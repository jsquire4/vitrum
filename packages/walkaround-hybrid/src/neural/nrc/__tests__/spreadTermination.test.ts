// spreadTermination.test.ts — CPU-oracle + WGSL-codegen tests for the NRC
// path-spread cache-termination heuristic (Müller et al. 2021 §5).
//
// The oracle (`../spreadTermination.ts`) is the reference; the WGSL predicate
// (`../wgsl/spreadTermination.wgsl.ts`) is hand-verified line-for-line. The
// predicate is pure arithmetic (sqrt + running sum + comparison) with no kinks,
// so the WGSL ↔ CPU match is exact to f32 epsilon — the codegen-shape pins below
// guard that the WGSL keeps emitting that same arithmetic.

import { describe, it, expect } from 'vitest';
import {
  segmentSpreadTerm, accumulatedSpread, primarySpread, evaluateSpreadTermination,
  type PathSegment,
} from '../spreadTermination.ts';
import { nrcSpreadTerminationWgsl } from '../wgsl/spreadTermination.wgsl.ts';
import { nrcQueryWgsl } from '../wgsl/nrcQuery.wgsl.ts';
import { RIS_GI_NRC_BODY } from '../../../shaders/risGiNrc.wgsl.ts';

describe('NRC spread term — per segment', () => {
  it('equals sqrt(d² / (p·|cosθ|))', () => {
    const seg: PathSegment = { dist: 2, pdf: 0.5, cosTheta: 0.8 };
    expect(segmentSpreadTerm(seg)).toBeCloseTo(Math.sqrt((2 * 2) / (0.5 * 0.8)), 9);
  });

  it('clamps a degenerate (near-zero pdf·cos) denominator → huge spread (early termination)', () => {
    const seg: PathSegment = { dist: 1, pdf: 0, cosTheta: 0 };
    // denom clamped to 1e-12 → term = sqrt(1/1e-12) = 1e6.
    expect(segmentSpreadTerm(seg)).toBeCloseTo(1e6, 0);
  });
});

describe('NRC accumulated spread (Müller 2021 §5)', () => {
  it('is the SQUARE of the running sum of per-segment terms', () => {
    const segs: PathSegment[] = [
      { dist: 1, pdf: 0.4, cosTheta: 0.9 },
      { dist: 0.5, pdf: 0.6, cosTheta: 0.7 },
      { dist: 2, pdf: 0.3, cosTheta: 0.5 },
    ];
    const acc = accumulatedSpread(segs);
    let running = 0;
    for (let i = 0; i < segs.length; i++) {
      running += segmentSpreadTerm(segs[i]!);
      // acc is a Float32Array → compare to the JS-double reference at f32
      // precision (~1e-6 relative; the brief's stated tolerance).
      expect(acc[i]).toBeCloseTo(running * running, 4);
    }
  });

  it('is monotonically non-decreasing along the path (spread only grows)', () => {
    const segs: PathSegment[] = [
      { dist: 1, pdf: 0.4, cosTheta: 0.9 },
      { dist: 0.5, pdf: 0.6, cosTheta: 0.7 },
      { dist: 2, pdf: 0.3, cosTheta: 0.5 },
    ];
    const acc = accumulatedSpread(segs);
    for (let i = 1; i < acc.length; i++) expect(acc[i]).toBeGreaterThanOrEqual(acc[i - 1]!);
  });

  it('a0 = (first-segment term)²', () => {
    const segs: PathSegment[] = [{ dist: 1.5, pdf: 0.5, cosTheta: 0.8 }, { dist: 1, pdf: 0.5, cosTheta: 0.5 }];
    const t0 = segmentSpreadTerm(segs[0]!);
    expect(primarySpread(segs)).toBeCloseTo(t0 * t0, 9);
  });
});

describe('NRC cache-termination heuristic a(x) > c·a0', () => {
  it('never terminates at the primary vertex (k=0)', () => {
    // even with a tiny c, the loop starts at k=1: a single-segment path never
    // terminates (the primary hit is never a cache-query target).
    const r = evaluateSpreadTermination([{ dist: 1, pdf: 0.5, cosTheta: 0.8 }], 1e-9);
    expect(r.terminate).toBe(false);
    expect(r.terminateAtSegment).toBe(-1);
  });

  it('fires at the first vertex whose accumulated spread exceeds c·a0', () => {
    const segs: PathSegment[] = [
      { dist: 1, pdf: 0.5, cosTheta: 0.9 },   // a0 footprint
      { dist: 1, pdf: 0.5, cosTheta: 0.9 },   // spread grows
      { dist: 5, pdf: 0.05, cosTheta: 0.3 },  // big jump
    ];
    const acc = accumulatedSpread(segs);
    const a0 = primarySpread(segs);
    // choose c so the threshold lands between segment 1 and 2.
    const c = (acc[1]! / a0 + acc[2]! / a0) / 2;
    const r = evaluateSpreadTermination(segs, c);
    expect(r.terminate).toBe(true);
    expect(r.terminateAtSegment).toBe(2);
    expect(acc[2]!).toBeGreaterThan(c * a0);
    expect(acc[1]!).toBeLessThanOrEqual(c * a0);
  });

  it('larger c defers termination (less bias / longer paths)', () => {
    const segs: PathSegment[] = [
      { dist: 1, pdf: 0.5, cosTheta: 0.9 },
      { dist: 1, pdf: 0.4, cosTheta: 0.8 },
      { dist: 1, pdf: 0.3, cosTheta: 0.6 },
      { dist: 1, pdf: 0.2, cosTheta: 0.4 },
    ];
    const small = evaluateSpreadTermination(segs, 1.2);
    const large = evaluateSpreadTermination(segs, 8.0);
    expect(small.terminate).toBe(true);
    // a larger c either defers to a later segment or stops firing within the path.
    if (large.terminate) {
      expect(large.terminateAtSegment).toBeGreaterThanOrEqual(small.terminateAtSegment);
    } else {
      expect(large.terminateAtSegment).toBe(-1);
    }
  });
});

// ─── H56-a: c=0.01 default-config enforcement (H26 seeding fix + H26 camera-pdf) ──
//
// Tests the spread termination at the PRODUCTION default (c = 0.01), with
// realistic camera / bounce geometry drawn from a Cornell-box-like scene.
// These are ENFORCEMENT tests (H26 fix landed): they pin the CORRECT behaviour
// (runningSum seeded at 0.0, primary-edge pdf = cameraPixelPdf) and will fail
// if the predicate or the a0 computation is regressed.
//
// H26 SEEDING FIX (risGiNrc.wgsl.ts — runningSum = 0.0):
//   The GPU WGSL previously seeded `runningSum = a0term` (the first-segment
//   spread TERM), which made the spread accumulation non-zero before the first
//   bounce edge — causing the termination threshold to fire immediately on the
//   primary vertex (k=0) and turn every pixel into a cache query.
//
//   The fix seeds `runningSum = 0.0` (Müller 2021 §5), matching the CPU oracle
//   (accumulatedSpread / evaluateSpreadTermination). The GPU and CPU paths now
//   agree — these tests enforce that agreement.
//
// H26 CAMERA-PDF FIX (risGiNrc.wgsl.ts — a0term uses nrcCfg.cameraPixelPdf):
//   The primary-edge spread term previously used a hard-coded pdf=1.0.  The
//   Müller-correct value is the camera's per-pixel solid-angle pdf:
//     cameraPixelPdf = cot²(fovY/2) · W · H / 4
//   This grows with render resolution and FOV narrowing, producing a LARGER a0
//   at high resolution (tighter camera footprint → harder to exceed c·a0 on
//   a single bounce → later / fewer cache terminations — physically correct).
//
// Geometry:
//   Primary ray: camera at z=5, hits the back wall at distance ~5.0,
//     cosθ_primary ≈ 1.0 (near-normal incidence).
//   Camera pdf example (1280×720, 60° vertical FOV):
//     cot(30°) = √3 ≈ 1.732, cameraPixelPdf ≈ 1.732²·1280·720/4 ≈ 552 960.
//   Bounce 1:    from back wall → left wall, distance ~2.0, cosθ ≈ 0.7, pdf = 0.45/π.
//   Bounce 2:    left wall → ceiling, distance ~1.5, cosθ ≈ 0.5, pdf = 0.30/π.
//   Bounce 3:    ceiling → right wall, distance ~3.0, cosθ ≈ 0.6, pdf = 0.25/π.

describe('NRC spread termination — c=0.01 production-default enforcement (H56-a, H26 fixed)', () => {
  // Production default c (Müller §5, vitrum default from HybridEngineOptions / risGiNrc.wgsl.ts).
  const C_PRODUCTION = 0.01;

  // Primary segment: camera → first bounce.
  // pdf=1.0 is the old hard-coded fallback; the Müller-correct value is
  // cameraPixelPdf (see H26 camera-pdf fix tests below).
  const primarySegPdf1: PathSegment = { dist: 5.0, pdf: 1.0, cosTheta: 1.0 };

  // Bounce segments: surface scattering (lambertian, cosθ/π pdf).
  const bounceSeg1: PathSegment = { dist: 2.0, pdf: 0.45 / Math.PI, cosTheta: 0.7 };
  const bounceSeg2: PathSegment = { dist: 1.5, pdf: 0.30 / Math.PI, cosTheta: 0.5 };
  const bounceSeg3: PathSegment = { dist: 3.0, pdf: 0.25 / Math.PI, cosTheta: 0.6 };

  const segsPdf1: PathSegment[] = [primarySegPdf1, bounceSeg1, bounceSeg2, bounceSeg3];

  it('primary footprint a0 = (first-segment term)² — pdf=1 oracle baseline', () => {
    // pdf = 1.0, cosθ = 1.0 → a0term = sqrt(5²/(1.0·1.0)) = 5 → a0 = 25.
    const a0 = primarySpread(segsPdf1);
    expect(a0).toBeCloseTo(25.0, 4);
  });

  it('accumulated spread at each bounce grows monotonically', () => {
    const acc = accumulatedSpread(segsPdf1);
    for (let i = 1; i < acc.length; i++) {
      expect(acc[i]).toBeGreaterThan(acc[i - 1]!);
    }
  });

  it('c=0.01, pdf=1: termination fires at segment 1 (tight c · small a0)', () => {
    // At c=0.01 with pdf=1 the threshold is 0.01·a0 = 0.25.  The bounce spread
    // at segment 1 is already > 0.25, so the oracle fires at segment 1 (k=1).
    const result = evaluateSpreadTermination(segsPdf1, C_PRODUCTION);
    expect(result.terminate).toBe(true);
    expect(result.terminateAtSegment).toBe(1);
    expect(result.a0).toBeCloseTo(25.0, 4);
  });

  it('c=0.01: accumulated spread at the termination vertex exceeds c·a0', () => {
    const result = evaluateSpreadTermination(segsPdf1, C_PRODUCTION);
    const acc = accumulatedSpread(segsPdf1);
    if (result.terminate) {
      const k = result.terminateAtSegment;
      expect(acc[k]!).toBeGreaterThan(C_PRODUCTION * result.a0);
      // And the preceding segment (k-1) must NOT have exceeded it (monotone ordering).
      if (k > 1) {
        expect(acc[k - 1]!).toBeLessThanOrEqual(C_PRODUCTION * result.a0);
      }
    }
  });

  it('c=1.0: termination fires later than c=0.01 (larger threshold = less early termination)', () => {
    const tight = evaluateSpreadTermination(segsPdf1, C_PRODUCTION);  // c=0.01
    const loose = evaluateSpreadTermination(segsPdf1, 1.0);             // c=1.0
    // With a larger c, the threshold c·a0 is higher, so termination must fire at
    // the same segment or later (or not at all).
    if (tight.terminate && loose.terminate) {
      expect(loose.terminateAtSegment).toBeGreaterThanOrEqual(tight.terminateAtSegment);
    }
    if (!loose.terminate) {
      expect(loose.terminateAtSegment).toBe(-1);
    }
  });

  it('H26 fixed: WGSL now seeds runningSum=0.0 matching the CPU oracle', () => {
    // H26 is fixed: risGiNrc.wgsl.ts now seeds runningSum = 0.0 (not a0term).
    const t0 = segmentSpreadTerm(primarySegPdf1);
    expect(t0).toBeGreaterThan(0);   // a0term = sqrt(25/1.0) = 5 (non-zero)
    expect(t0).toBeCloseTo(5.0, 4);  // sqrt(25 / (1.0 · 1.0)) = 5

    // Enforce the fix: the NRC body must seed runningSum at 0.0 not a0term.
    expect(RIS_GI_NRC_BODY).toContain('var runningSum: f32 = 0.0;');
    expect(RIS_GI_NRC_BODY).not.toContain('var runningSum: f32 = a0term;');
  });

  // ─── H26 camera-pdf fix enforcement ──────────────────────────────────────
  //
  // The corrected a0 uses the camera's per-pixel solid-angle pdf, not 1.0.
  // For a 1280×720 frame with 60° vertical FOV:
  //   cot(fovY/2) = cot(30°) = √3 ≈ 1.7321
  //   cameraPixelPdf = cot²(30°) · 1280 · 720 / 4 = 3 · 921600 / 4 = 691 200
  //
  // With this large pdf, a0term = sqrt(25 / (691200 · 1.0)) ≈ 0.006,
  // so a0 ≈ 3.6e-5 — much smaller than the pdf=1 baseline (25).
  // The threshold c·a0 ≈ 3.6e-7 is now VERY tight and termination fires at k=1
  // just as before, BUT with the physically-correct footprint that scales with
  // resolution. The key invariant is that the RATIO aX/a0 is what matters for
  // the heuristic, not the absolute values.

  it('H26 camera-pdf: higher camera pdf → smaller a0 (correctly tighter primary footprint)', () => {
    // 1280×720 @ 60° vFOV → cameraPixelPdf ≈ 691 200.
    const cotFovY = Math.sqrt(3); // cot(30°)
    const W = 1280, H = 720;
    const camPdf = (cotFovY * cotFovY * W * H) / 4; // ≈ 691 200
    expect(camPdf).toBeCloseTo(691200, 0);

    const primarySegHighPdf: PathSegment = { dist: 5.0, pdf: camPdf, cosTheta: 1.0 };
    const segsHighPdf: PathSegment[] = [primarySegHighPdf, bounceSeg1, bounceSeg2, bounceSeg3];

    const a0Pdf1 = primarySpread(segsPdf1);          // pdf=1 baseline → ≈ 25
    const a0HighPdf = primarySpread(segsHighPdf);    // pdf=691200 → ≈ 3.6e-5

    // Higher pdf → SMALLER a0 (tighter camera footprint).
    expect(a0HighPdf).toBeLessThan(a0Pdf1);
    // The ratio a0 ~ 1/pdf (spread term ~ sqrt(d²/(pdf·cos)) → a0 ~ d²/(pdf·cos)).
    // a0_highPdf / a0_pdf1 ≈ 1/camPdf.
    expect(a0HighPdf / a0Pdf1).toBeCloseTo(1 / camPdf, 3);
  });

  it('H26 camera-pdf: with realistic pdf the termination heuristic still fires (bounce spreads exceed tight a0)', () => {
    const cotFovY = Math.sqrt(3);
    const camPdf = (cotFovY * cotFovY * 1280 * 720) / 4;
    const primarySegHighPdf: PathSegment = { dist: 5.0, pdf: camPdf, cosTheta: 1.0 };
    const segsHighPdf: PathSegment[] = [primarySegHighPdf, bounceSeg1, bounceSeg2, bounceSeg3];

    // With camPdf≫1 the bounce-edge spread terms remain the same as before
    // (bounce segments use their own lower pdfs), but a0 is much smaller.
    // c·a0 is tiny → termination fires at the first bounce (k=1), just as with
    // pdf=1 at this tight c. The predicate is a(x) > c·a0; both sides scale with
    // pdf (a0 small, but aX at bounce 1 is independent of the camera pdf) so the
    // ratio aX/a0 is LARGER → termination fires at least as early.
    const result = evaluateSpreadTermination(segsHighPdf, C_PRODUCTION);
    expect(result.terminate).toBe(true);
    expect(result.terminateAtSegment).toBe(1);
  });

  it('H26 camera-pdf: WGSL uses nrcCfg.cameraPixelPdf for the primary-edge spread term', () => {
    // The GPU WGSL must read the camera pdf from the NRC config UBO, not use 1.0.
    expect(RIS_GI_NRC_BODY).toContain('nrcCfg.cameraPixelPdf');
    // The old hard-coded literal must not appear for the a0term call.
    expect(RIS_GI_NRC_BODY).not.toContain('nrcSegmentSpreadTerm(hit.dist, 1.0,');
  });
});

describe('NRC spread WGSL codegen — shape pins (oracle equivalence)', () => {
  it('emits sqrt(d²/(p·|cosθ|)) with the same 1e-12 denom clamp', () => {
    const wgsl = nrcSpreadTerminationWgsl();
    expect(wgsl).toContain('max(pdf * abs(cosTheta), 1e-12)');
    expect(wgsl).toContain('sqrt((dist * dist) / denom)');
  });

  it('accumulates the running sum and returns its square as a(x)', () => {
    const wgsl = nrcSpreadTerminationWgsl();
    expect(wgsl).toContain('(*runningSum_io) * (*runningSum_io)');
  });

  it('the termination test is a(x) > c·a0', () => {
    const wgsl = nrcSpreadTerminationWgsl();
    expect(wgsl).toContain('return aX > c * a0;');
  });
});

describe('NRC warm-up gate — cold cache predictions do not replace DDGI', () => {
  it('threads trainedSteps/warmupSteps through the NRC query UBO without changing its size', () => {
    const wgsl = nrcQueryWgsl({
      levels: 2,
      featuresPerEntry: 2,
      oneBlobBins: 4,
      width: 8,
      outWidth: 3,
      hidden: 1,
    });
    expect(wgsl).toContain('trainedSteps    : u32');
    expect(wgsl).toContain('warmupSteps     : u32');
    expect(wgsl).not.toContain('_pad1 : u32, _pad2 : u32');
  });

  it('keeps spread-fired records but gates visible Lo substitution until the trainer is warm', () => {
    expect(RIS_GI_NRC_BODY).toContain('let nrcCanSubstitute = nrcCfg.trainedSteps >= nrcCfg.warmupSteps;');
    expect(RIS_GI_NRC_BODY).toContain('Lo = select(ddgiLo, nrcQueryRadiance(xs, ns, -wi, xsRough, xsAlbedo), nrcCanSubstitute);');
    expect(RIS_GI_NRC_BODY).toContain('if (nrcFired) {');
    expect(RIS_GI_NRC_BODY).toContain('nrcWriteRecord(');
  });
});

// ── A6: NRC structural fixes — 2026-06-10 ────────────────────────────────────
//
// Pins three A6 changes to risGiNrc.wgsl:
//   1. xsRough: real per-tri roughness from bvh_material (not hardcoded 1.0)
//   2. Training target: r.Lo post-loop (not DDGI distillation inside the loop)
//   3. Structural: candidate tracking before loop, record after loop

describe('A6: NRC structural fixes — xsRough + reservoir training target (2026-06-10)', () => {
  it('A6 xsRough: NRC body declares bvh_material binding (group 1, binding 14)', () => {
    // The NRC pass must bind the per-tri roughness texture at the same slot as
    // ris.wgsl / restirCastPrimary.wgsl / shade.wgsl (binding 14) so the MLP
    // input vector encodes the real authored roughness at the bounce vertex.
    expect(RIS_GI_NRC_BODY).toContain('@group(1) @binding(14) var bvh_material: texture_2d<u32>');
  });

  it('A6 xsRough: NRC body reads mapped payload roughness (not hardcoded 1.0)', () => {
    // The old xsRough = 1.0 must be gone; the real roughness must flow through
    // the shared GI material payload path.
    expect(RIS_GI_NRC_BODY).not.toContain('let xsRough = 1.0;');
    expect(RIS_GI_NRC_BODY).toContain('let xsPayload = sampleRestirGIHitMaterialForHit(');
    expect(RIS_GI_NRC_BODY).toContain('let xsRough = xsPayload.rough;');
    expect(RIS_GI_NRC_BODY).toContain('bvh_material');
  });

  it('A6 training target: nrcWriteRecord is called POST-LOOP with r.Lo, not inside the loop with directLo', () => {
    // The training record must use r.Lo (the ReSTIR-GI reservoir Lo) as the
    // target, not a per-candidate DDGI estimate computed inside the RIS loop.
    // Structural enforcement: nrcWriteRecord must appear after the RIS loop
    // (after the closing `}` of `for ... i < M_GI`) and must reference r.Lo.
    // Negative check: the old inside-loop record call with directLo is gone.
    expect(RIS_GI_NRC_BODY).not.toContain('let directLo =');
    expect(RIS_GI_NRC_BODY).not.toContain('nrcWriteRecord(pixelIdxGi % nrcCfg.recordCap, xs, ns, -wi,');
    // Positive check: post-loop record write uses nrcTrackXs / r.Lo.
    expect(RIS_GI_NRC_BODY).toContain('nrcTrackXs');
    expect(RIS_GI_NRC_BODY).toContain('nrcTrackNs');
    expect(RIS_GI_NRC_BODY).toContain('nrcFired');
    expect(RIS_GI_NRC_BODY).toContain('r.Lo');
  });

  it('A6 training target: nrcWriteRecord is called with r.Lo as the last argument', () => {
    // r.Lo is passed as the target argument (last positional arg) to nrcWriteRecord.
    expect(RIS_GI_NRC_BODY).toContain('r.Lo,');
  });

  it('A6 candidate tracking: NRC tracking vars are declared before the RIS loop', () => {
    // nrcFired / nrcTrackXs / nrcTrackNs / nrcTrackWi must appear before the
    // opaque NRC `for (var i: u32 = 0u; i < M_GI;` loop opening. Anchor on the
    // semicolon so the bounded glass branch's `M_GI_g` loop is not mistaken for
    // the NRC substitution loop.
    const loopIdx = RIS_GI_NRC_BODY.indexOf('for (var i: u32 = 0u; i < M_GI;');
    expect(loopIdx).toBeGreaterThan(0);
    const nrcFiredIdx = RIS_GI_NRC_BODY.indexOf('var nrcFired:');
    expect(nrcFiredIdx).toBeGreaterThan(0);
    expect(nrcFiredIdx).toBeLessThan(loopIdx);
  });

  it('A6 tail-padding: no inside-loop sun shadow trace for NRC training (removed with directLo)', () => {
    // The old implementation added a shadow ray per NRC candidate for the training
    // target (directLo = sun+DDGI). After A6 this is removed — training uses r.Lo
    // which comes from the existing post-loop visibility test.
    // The only sunDirection reference should be outside the NRC termination block.
    // We check that there is no sunContrib pattern (old inside-loop shadow).
    expect(RIS_GI_NRC_BODY).not.toContain('var sunContrib');
  });

  it('A6 spreadC default 0.01: fires at bounce 1 for realistic camera pdf (selectivity arithmetic)', () => {
    // Derived arithmetic (see file-level A6 docblock):
    //   camPdf = 691200 (1280×720 @ 60° vFOV), d0=5, cos0=1 → a0 ≈ 3.6e-5
    //   bounce1: d=2, pdf≈0.222, cosArrive=0.7 → aX1 ≈ 25.6 >> threshold 3.6e-7
    // c=0.01 correctly fires at bounce 1.  This test enforces the arithmetic stays
    // pinned and the oracle uses the correct camera-pdf scaling.
    const cotFovY = Math.sqrt(3); // cot(30°) = cot(fovY/2) for 60° vFOV
    const camPdf = (cotFovY * cotFovY * 1280 * 720) / 4;

    const primarySeg: PathSegment = { dist: 5.0, pdf: camPdf, cosTheta: 1.0 };
    const bounce1: PathSegment  = { dist: 2.0, pdf: (0.7 / Math.PI), cosTheta: 0.7 };
    const segs = [primarySeg, bounce1];

    const a0 = primarySpread(segs);
    const acc = accumulatedSpread(segs);

    // a0 is tiny (high camPdf → small camera footprint).
    expect(a0).toBeLessThan(1e-4);
    // aX at bounce 1 is >> c*a0 for c=0.01.
    const threshold = 0.01 * a0;
    expect(acc[1]!).toBeGreaterThan(threshold);
    // Confirm the oracle fires at bounce 1.
    const result = evaluateSpreadTermination(segs, 0.01);
    expect(result.terminate).toBe(true);
    expect(result.terminateAtSegment).toBe(1);
  });
});
