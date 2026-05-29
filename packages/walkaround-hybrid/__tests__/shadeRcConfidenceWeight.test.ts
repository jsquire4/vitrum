/**
 * shade.wgsl lo_indirect — per-pixel confidence-ratio (balance-heuristic)
 * composition of ReSTIR-GI and Radiance-Cascades indirect estimates.
 *
 * Replaces the old fixed host scalar `(1 - rcWeight)·Lo_restir + rcWeight·Lo_rc`
 * with a per-pixel convex blend driven by each estimator's reliability:
 *
 *   m        = clamp(Meff / restirGiMClamp, 0, 1)   ReSTIR-GI confidence
 *   c_restir = m
 *   c_rc     = clamp(rcWeight, 0, 1) · (1 - m)      RC prior gated by ReSTIR unreliability
 *   w_rc     = c_rc / (c_rc + c_restir)             (0 when c_rc + c_restir ≈ 0)
 *   w_restir = 1 - w_rc
 *   Lo       = w_restir·Lo_restir + w_rc·Lo_rc
 *
 * Both estimators integrate the same diffuse-indirect radiance, so any blend
 * summing to 1 is unbiased. This file pins the math contract with a CPU port
 * of the weight function plus structural pins on the WGSL.
 *
 * The properties pinned:
 *   (a) RC disabled (rcWeight = 0, Lo_rc = 0) → pure ReSTIR-GI, BIT-IDENTICAL.
 *   (b) the two weights always sum to exactly 1.
 *   (c) high-confidence-RC / low-confidence-ReSTIR pixel weights toward RC,
 *       and the reverse weights toward ReSTIR.
 *   (d) degenerate pixel (no valid reservoir AND rcWeight 0) stays 0.
 */

import { describe, expect, it } from 'vitest';
import { SHADE_WGSL } from '../src/shaders/shade.wgsl.js';

// ── CPU port of the WGSL weight function (shade.wgsl.ts lo_indirect tail) ──
// Mirrors the WGSL line-for-line so the test fails if the shader math drifts.
interface BlendResult {
  wRestir: number;
  wRc: number;
  /** Final per-channel composed radiance. */
  lo: [number, number, number];
}

function composeIndirect(
  loRestir: [number, number, number],
  loRc: [number, number, number],
  Meff: number,
  restirGiMClamp: number,
  rcWeight: number,
): BlendResult {
  const m = Math.min(Math.max(Meff / Math.max(restirGiMClamp, 1), 0), 1);
  const cRestir = m;
  const cRc = Math.min(Math.max(rcWeight, 0), 1) * (1 - m);
  const cSum = cRestir + cRc;
  const wRc = cSum > 1e-6 ? cRc / Math.max(cSum, 1e-6) : 0.0;
  const wRestir = 1.0 - wRc;
  return {
    wRestir,
    wRc,
    lo: [
      wRestir * loRestir[0] + wRc * loRc[0],
      wRestir * loRestir[1] + wRc * loRc[1],
      wRestir * loRestir[2] + wRc * loRc[2],
    ],
  };
}

describe('shade lo_indirect — confidence-ratio RC/ReSTIR composition', () => {
  // (a) RC disabled: the host binds an all-zero rcParams placeholder, so both
  // rcWeight == 0 AND sampleCascadeC0 returns vec3f(0). The blend must collapse
  // to a bit-identical Lo_indirect passthrough for ANY ReSTIR confidence.
  it('(a) rc-disabled (rcWeight=0, Lo_rc=0) is a bit-identical ReSTIR passthrough', () => {
    const loRestir: [number, number, number] = [0.1234, 0.5678, 0.9012];
    const loRc: [number, number, number] = [0, 0, 0]; // sampleCascadeC0 → 0 when enabled==0
    for (const Meff of [0, 1, 8, 25, 50, 123]) {
      const r = composeIndirect(loRestir, loRc, Meff, 50, /* rcWeight */ 0);
      // w_restir is exactly 1.0, w_rc exactly 0.0 — no float drift.
      expect(r.wRc).toBe(0);
      expect(r.wRestir).toBe(1);
      // Result is byte-identical to Lo_indirect.
      expect(r.lo[0]).toBe(loRestir[0]);
      expect(r.lo[1]).toBe(loRestir[1]);
      expect(r.lo[2]).toBe(loRestir[2]);
    }
  });

  // (b) Unbiasedness invariant: the weights are a convex partition of unity for
  // every (Meff, rcWeight) combination, including the degenerate corners.
  it('(b) w_restir + w_rc == 1 across the full parameter grid', () => {
    for (const Meff of [0, 1, 5, 25, 50, 100, 500]) {
      for (const rcWeight of [0, 0.25, 0.5, 0.75, 1.0]) {
        const r = composeIndirect([0.3, 0.3, 0.3], [0.7, 0.7, 0.7], Meff, 50, rcWeight);
        expect(r.wRestir + r.wRc).toBeCloseTo(1, 12);
        // Both weights are valid probabilities.
        expect(r.wRc).toBeGreaterThanOrEqual(0);
        expect(r.wRc).toBeLessThanOrEqual(1);
      }
    }
  });

  // (c) The whole point: reliability steers the blend per-pixel.
  it('(c) high-RC-confidence / low-ReSTIR-confidence pixel weights TOWARD RC', () => {
    // Fresh disocclusion: Meff = 1 (just RIS-initialised), host trusts RC fully.
    const fresh = composeIndirect([1, 1, 1], [2, 2, 2], /* Meff */ 1, 50, /* rcWeight */ 1.0);
    // m = 1/50 = 0.02 ⇒ c_rc = 0.98, c_restir = 0.02 ⇒ w_rc ≫ w_restir.
    expect(fresh.wRc).toBeGreaterThan(fresh.wRestir);
    expect(fresh.wRc).toBeGreaterThan(0.9);

    // Converged pixel: Meff at/above the M-clamp ⇒ m = 1 ⇒ c_rc = 0 ⇒ pure ReSTIR
    // EVEN with rcWeight = 1. RC fades out exactly where ReSTIR is trustworthy.
    const converged = composeIndirect([1, 1, 1], [2, 2, 2], /* Meff */ 50, 50, /* rcWeight */ 1.0);
    expect(converged.wRestir).toBe(1);
    expect(converged.wRc).toBe(0);
  });

  it('(c2) the converse: low-RC-prior / high-ReSTIR-confidence weights TOWARD ReSTIR', () => {
    // Mid-converged pixel (m = 0.5) but the host barely trusts RC (rcWeight 0.1).
    const r = composeIndirect([1, 1, 1], [2, 2, 2], /* Meff */ 25, 50, /* rcWeight */ 0.1);
    // c_restir = 0.5, c_rc = 0.1·0.5 = 0.05 ⇒ w_restir = 0.5/0.55 ≈ 0.909.
    expect(r.wRestir).toBeGreaterThan(r.wRc);
    expect(r.wRestir).toBeCloseTo(0.5 / 0.55, 10);
  });

  it('(c3) rcWeight scales RC influence monotonically at fixed ReSTIR confidence', () => {
    // Hold m = 0.5; raising rcWeight must raise w_rc.
    const lo: [number, number, number] = [0.5, 0.5, 0.5];
    const w0 = composeIndirect(lo, lo, 25, 50, 0.2).wRc;
    const w1 = composeIndirect(lo, lo, 25, 50, 0.6).wRc;
    const w2 = composeIndirect(lo, lo, 25, 50, 1.0).wRc;
    expect(w1).toBeGreaterThan(w0);
    expect(w2).toBeGreaterThan(w1);
  });

  // (d) No ReSTIR estimate at all (Meff = 0) but RC active: RC takes the pixel.
  // The convex-blend stays unbiased and the disabled-corner stays 0.
  it('(d1) Meff=0 with rcWeight>0 hands the full weight to RC', () => {
    const r = composeIndirect([0, 0, 0], [0.4, 0.5, 0.6], /* Meff */ 0, 50, /* rcWeight */ 0.8);
    // m = 0 ⇒ c_restir = 0, c_rc = 0.8 ⇒ w_rc = 1.
    expect(r.wRc).toBe(1);
    expect(r.wRestir).toBe(0);
    expect(r.lo).toEqual([0.4, 0.5, 0.6]);
  });

  it('(d2) degenerate corner (Meff=0, rcWeight=0) returns 0 with no NaN', () => {
    const r = composeIndirect([0, 0, 0], [0, 0, 0], /* Meff */ 0, 50, /* rcWeight */ 0);
    // c_rc + c_restir == 0 ⇒ guard forces w_rc = 0, w_restir = 1.
    expect(r.wRc).toBe(0);
    expect(r.wRestir).toBe(1);
    expect(Number.isNaN(r.lo[0])).toBe(false);
    expect(r.lo).toEqual([0, 0, 0]);
  });
});

describe('shade lo_indirect — WGSL structural pins', () => {
  it('accumulates a bilinear-weighted ReSTIR M alongside Lo_indirect', () => {
    // Confidence proxy is the reservoir sample count M, blended by the same
    // bilinear weight `bw` as the radiance.
    expect(SHADE_WGSL).toContain('Maccum = Maccum + f32(g.M) * bw;');
    expect(SHADE_WGSL).toMatch(/Meff\s*=\s*Maccum\s*\/\s*totalW;/);
  });

  it('derives the confidence-ratio weights from M and the host rcWeight prior', () => {
    expect(SHADE_WGSL).toMatch(
      /let m\s*=\s*clamp\(Meff\s*\/\s*f32\(max\(ubo\.restirGiMClamp,\s*1u\)\),\s*0\.0,\s*1\.0\);/,
    );
    expect(SHADE_WGSL).toContain('let cRestir = m;');
    expect(SHADE_WGSL).toContain('let cRc = clamp(rcParams.rcWeight, 0.0, 1.0) * (1.0 - m);');
    expect(SHADE_WGSL).toMatch(/let wRc\s*=\s*select\(0\.0,\s*cRc\s*\/\s*max\(cSum,\s*1e-6\),\s*cSum\s*>\s*1e-6\);/);
    expect(SHADE_WGSL).toContain('let wRestirGi = 1.0 - wRc;');
  });

  it('still composes a convex blend (weights sum to 1 by construction)', () => {
    expect(SHADE_WGSL).toContain('return wRestirGi * Lo_indirect + wRc * Lo_rc;');
  });

  it('no longer uses the old fixed host-scalar lerp', () => {
    // The misleading fixed-scalar form `let wRc = clamp(rcParams.rcWeight,…);`
    // is gone — wRc is now derived from the per-pixel confidence ratio.
    expect(SHADE_WGSL).not.toMatch(/let wRc\s*=\s*clamp\(rcParams\.rcWeight,\s*0\.0,\s*1\.0\);/);
  });
});
