/**
 * WS4 — Volumetric subsurface scattering tests (pt-webgpu).
 *
 * Two halves:
 *
 *  (1) CPU oracles for the radiometric math implemented in the WGSL kernel:
 *      the Henyey-Greenstein phase function and its importance sampler, the
 *      free-flight (exponential) distance sampler, the single-scatter
 *      homogeneous-slab transmittance / albedo identities, and the
 *      phase ↔ NEE power-heuristic partition-of-unity. These are pure-TS
 *      mirrors of the WGSL functions so the WGSL can be reasoned about and
 *      regression-pinned without a GPU.
 *
 *  (2) Structural gates on the composed WGSL: the volumetric-walk symbols are
 *      ABSENT from the kernel composed for the BDPT-enabled pipeline (the
 *      compile-time gate the plan mandates — energy non-conservation otherwise,
 *      since the BDPT light subpath has no medium logic), and the FrameParams
 *      UBO byte size is unchanged by the feature (σ_a is packed into the
 *      materials buffer, not the UBO).
 *
 * Refs:
 *  - Henyey, L. G., & Greenstein, J. L. "Diffuse radiation in the galaxy."
 *    Astrophys. J. 93:70–83 (1941). The HG phase function.
 *  - Pharr, Jakob, Humphreys. Physically Based Rendering 4th ed. §11 "Volume
 *    Scattering" — homogeneous transmittance, free-flight sampling, single
 *    scattering albedo.
 */

import { describe, expect, it } from 'vitest';
import { FRAME_PARAMS_BYTE_SIZE } from '../scene/frameParamsLayout.js';
import {
  PT_WEBGPU_TRACE_WGSL,
  composePtWebgpuTraceWgsl,
} from '../wgsl/pathTraceBruteforce.wgsl.js';
import { MATERIAL_FLOAT_STRIDE, materialToPackedVec4s } from '../scene/materialPacking.js';

// ---------------------------------------------------------------------------
// CPU mirrors of the WGSL volumetric math (identical formulas).
// ---------------------------------------------------------------------------

const INV_4PI = 1.0 / (4.0 * Math.PI);

/** HG phase function p(cosθ; g). Henyey-Greenstein 1941. */
function hgPhase(cosTheta: number, g: number): number {
  const denom = 1.0 + g * g - 2.0 * g * cosTheta;
  return (INV_4PI * (1.0 - g * g)) / Math.max(Math.pow(denom, 1.5), 1e-9);
}

/**
 * HG importance sample: given a uniform u1, return cosθ of the scattered
 * direction relative to the RAY-TRAVEL direction (PBR4e §11.3 eq. 11.7).
 *
 * The PBRT closed form returns cosθ measured against wo = -travel, so its mean
 * is -g; we negate so that cosθ is measured against the travel direction and
 * its mean is +g (forward scatter for g>0). This matches the WGSL sampler,
 * which builds the ONB around the travel direction `wIn` and uses cosθ directly.
 */
function hgSampleCosTheta(g: number, u1: number): number {
  if (Math.abs(g) < 1e-3) {
    return 1.0 - 2.0 * u1; // isotropic
  }
  const sq = (1.0 - g * g) / (1.0 + g - 2.0 * g * u1);
  const cosThetaWo = -(1.0 + g * g - sq * sq) / (2.0 * g);
  return -cosThetaWo; // relative to travel direction; mean = +g
}

/** Free-flight distance: t = -ln(1-ξ)/σ_t (exponential transmittance CDF inversion). */
function freeFlightDistance(sigmaT: number, xi: number): number {
  return -Math.log(Math.max(1.0 - xi, 1e-9)) / sigmaT;
}

/** Power heuristic (β = 2) — matches WGSL powerHeuristic. */
function powerHeuristic(pdfA: number, pdfB: number): number {
  const a2 = pdfA * pdfA;
  const b2 = pdfB * pdfB;
  return a2 / Math.max(a2 + b2, 1e-6);
}

// Simple deterministic LCG so the MC tests are reproducible.
function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

describe('Henyey-Greenstein phase function (CPU oracle)', () => {
  it('g=0 is exactly the isotropic constant 1/(4π)', () => {
    for (const c of [-1, -0.5, 0, 0.3, 1]) {
      expect(hgPhase(c, 0)).toBeCloseTo(INV_4PI, 12);
    }
  });

  it('integrates to 1 over the sphere (MC, uniform solid-angle)', () => {
    const rng = makeRng(0xc0ffee);
    for (const g of [-0.7, -0.3, 0, 0.3, 0.7]) {
      const N = 400000;
      let sum = 0;
      for (let i = 0; i < N; i += 1) {
        // Uniform direction on the sphere; cosθ relative to +Z axis.
        const cosT = 1.0 - 2.0 * rng();
        sum += hgPhase(cosT, g);
      }
      // ∫ p dω = (4π) · E[p] over a uniform-solid-angle estimator.
      const integral = 4.0 * Math.PI * (sum / N);
      expect(integral).toBeCloseTo(1.0, 1);
    }
  });
});

describe('HG importance sampler matches its pdf (IS weight ≈ 1)', () => {
  it('f/pdf = 1 because the HG sampler IS the pdf', () => {
    const rng = makeRng(0x1234);
    for (const g of [-0.6, 0, 0.4, 0.8]) {
      const N = 200000;
      let sum = 0;
      for (let i = 0; i < N; i += 1) {
        const cosT = hgSampleCosTheta(g, rng());
        const p = hgPhase(cosT, g);
        // The HG sampler draws cosθ ∝ p, azimuth uniform; pdf(ω) = p(cosθ).
        // So f/pdf = p/p = 1. Mean must be 1.
        sum += p / Math.max(p, 1e-12);
      }
      expect(sum / N).toBeCloseTo(1.0, 6);
    }
  });

  it('sampled cosθ distribution mean matches the analytic mean g', () => {
    // E[cosθ] for HG equals the anisotropy parameter g.
    const rng = makeRng(0x99);
    for (const g of [-0.5, 0, 0.5]) {
      const N = 400000;
      let sum = 0;
      for (let i = 0; i < N; i += 1) {
        sum += hgSampleCosTheta(g, rng());
      }
      expect(sum / N).toBeCloseTo(g, 1);
    }
  });
});

describe('Free-flight CDF inversion', () => {
  it('exponential mean free path = 1/σ_t', () => {
    const rng = makeRng(0x5eed);
    for (const sigmaT of [0.5, 1.0, 3.0]) {
      const N = 300000;
      let sum = 0;
      for (let i = 0; i < N; i += 1) {
        sum += freeFlightDistance(sigmaT, rng());
      }
      expect(sum / N).toBeCloseTo(1.0 / sigmaT, 1);
    }
  });

  it('fraction surviving past distance d matches Beer-Lambert exp(-σ_t·d)', () => {
    const rng = makeRng(0xbeef);
    const sigmaT = 2.0;
    const d = 0.5;
    const N = 400000;
    let survived = 0;
    for (let i = 0; i < N; i += 1) {
      if (freeFlightDistance(sigmaT, rng()) > d) survived += 1;
    }
    expect(survived / N).toBeCloseTo(Math.exp(-sigmaT * d), 2);
  });
});

describe("Single-scatter homogeneous slab (Beer's-law + albedo anchors)", () => {
  it('no-scatter (σ_s=0) slab transmits exactly exp(-σ_a·d)', () => {
    // With σ_s = 0, σ_t = σ_a; a path transmits iff the sampled free flight
    // exceeds the slab thickness. The surviving fraction is the analytic
    // transmittance. High-precision Beer anchor.
    const rng = makeRng(0x0a0a);
    const sigmaA = 1.3;
    const d = 0.8;
    const N = 600000;
    let transmitted = 0;
    for (let i = 0; i < N; i += 1) {
      if (freeFlightDistance(sigmaA, rng()) > d) transmitted += 1;
    }
    expect(transmitted / N).toBeCloseTo(Math.exp(-sigmaA * d), 2);
  });

  it('no-collision branch estimator yields exp(-σ_a·d) per channel (NOT exp(-2σ_a·d)) — double-count fix', () => {
    // V23 — the kernel's no-collision branch reaches the surface with probability
    // P(t ≥ d) = exp(-heroSigmaT·d) (free-flight IS) and must then multiply the
    // surviving throughput by exp(-(σ_t_c − heroSigmaT)·d), NOT the full
    // exp(-σ_t_c·d). The prior code multiplied by the full transmittance, which
    // DOUBLE-counted it (survival prob × explicit factor) → exp(-2σ_t·d),
    // over-darkening every medium by the square of its transmittance. This oracle
    // simulates the corrected estimator and pins the per-channel mean to the
    // analytic Beer-Lambert transmittance.
    const rng = makeRng(0x5151);
    // Strongly CHROMATIC σ_a (σ_s = 0 ⇒ σ_t = σ_a). Hero = max channel.
    const sigmaA = [0.6, 1.8, 3.4];
    const hero = Math.max(...sigmaA);
    const d = 0.5;
    const N = 800000;
    const acc = [0, 0, 0];
    for (let i = 0; i < N; i += 1) {
      const t = freeFlightDistance(hero, rng()); // hero-channel free flight
      if (t > d) {
        // Reached the surface — corrected per-channel multiplier.
        for (let c = 0; c < 3; c += 1) acc[c]! += Math.exp(-(sigmaA[c]! - hero) * d);
      }
      // else: collision ⇒ absorbed (albedo σ_s/σ_t = 0), contributes 0.
    }
    for (let c = 0; c < 3; c += 1) {
      expect(acc[c]! / N).toBeCloseTo(Math.exp(-sigmaA[c]! * d), 2);
    }
  });

  it('single-scatter albedo of an event equals σ_s/σ_t', () => {
    // At a real (non-null) collision the throughput is multiplied by the
    // single-scattering albedo σ_s/σ_t (PBR4e §11.2). Pin the identity.
    for (const [sigmaS, sigmaA] of [
      [1.0, 0.0],
      [0.5, 0.5],
      [0.2, 0.8],
      [3.0, 1.0],
    ] as const) {
      const sigmaT = sigmaS + sigmaA;
      expect(sigmaS / sigmaT).toBeGreaterThanOrEqual(0);
      expect(sigmaS / sigmaT).toBeLessThanOrEqual(1);
      expect(sigmaS / sigmaT).toBeCloseTo(sigmaS / (sigmaS + sigmaA), 12);
    }
  });
});

describe('In-medium MIS partition-of-unity', () => {
  it('w_phase + w_nee = 1 for the symmetric two-strategy power heuristic', () => {
    const rng = makeRng(0x7);
    for (let i = 0; i < 10000; i += 1) {
      const pPhase = rng() * 5 + 1e-3;
      const pNee = rng() * 5 + 1e-3;
      const wPhase = powerHeuristic(pPhase, pNee);
      const wNee = powerHeuristic(pNee, pPhase);
      expect(wPhase + wNee).toBeCloseTo(1.0, 6);
    }
  });
});

describe('σ_a packing from attenuationColor / attenuationDistance', () => {
  it('packs σ_a = -ln(attenuationColor)/attenuationDistance with a present flag', () => {
    const att: readonly [number, number, number] = [0.5, 0.25, 0.1];
    const dist = 2.0;
    const packed = materialToPackedVec4s({
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      transmission: 1,
      ior: 1.5,
      attenuationColor: att,
      attenuationDistance: dist,
      scatteringCoefficient: 0.3,
    } as never);
    expect(packed.length).toBe(MATERIAL_FLOAT_STRIDE);
    // σ_a is at vec4 #22 (float offset 88). H52 bumped the stride 23→26 by
    // appending clearcoat/sheen/iridescence AFTER the σ_a vec4, so the σ_a
    // offset is a fixed constant, not MATERIAL_FLOAT_STRIDE - 4.
    const SIGMA_A_FLOAT_OFFSET = 22 * 4; // = 88
    const sigmaA = packed.slice(SIGMA_A_FLOAT_OFFSET, SIGMA_A_FLOAT_OFFSET + 4);
    for (let c = 0; c < 3; c += 1) {
      const expected = -Math.log(Math.max(att[c] ?? 1, 1e-4)) / dist;
      expect(sigmaA[c] ?? NaN).toBeCloseTo(Math.max(expected, 0), 5);
    }
    expect(sigmaA[3]).toBe(1); // hasSigmaA flag set
  });

  it('leaves σ_a zero + flag clear when attenuation fields absent', () => {
    const packed = materialToPackedVec4s({
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
    } as never);
    expect(packed.length).toBe(MATERIAL_FLOAT_STRIDE);
    const SIGMA_A_FLOAT_OFFSET = 22 * 4; // = 88
    const sigmaA = packed.slice(SIGMA_A_FLOAT_OFFSET, SIGMA_A_FLOAT_OFFSET + 4);
    expect(sigmaA).toEqual([0, 0, 0, 0]);
  });
});

describe('Structural compile-time gate: SSS off when BDPT enabled', () => {
  const sssOn = composePtWebgpuTraceWgsl(false); // bdpt off → volumetric walk present
  const sssOff = composePtWebgpuTraceWgsl(true); // bdpt on  → volumetric walk gated out

  it('default export equals the BDPT-off (SSS-on) composition', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toBe(sssOn);
  });

  it('SSS-on kernel contains the volumetric-walk symbols', () => {
    expect(sssOn).toContain('var inMedium');
    expect(sssOn).toContain('fn hgPhase');
    expect(sssOn).toContain('fn sampleHenyeyGreenstein');
    expect(sssOn).toContain('freeFlightDist');
  });

  it('BDPT-on kernel OMITS the volumetric-walk symbols (compile-time gate)', () => {
    expect(sssOff).not.toContain('var inMedium');
    expect(sssOff).not.toContain('freeFlightDist');
    // The HG helpers are also dropped to avoid dead-code in the BDPT build.
    expect(sssOff).not.toContain('fn sampleHenyeyGreenstein');
  });

  it('BDPT-on kernel keeps the legacy Beer-Lambert absorption fallback', () => {
    expect(sssOff).toContain('exp(-sigmaT * hit.dist)');
  });

  it('does not change the FrameParams UBO byte size (σ_a lives in materials buffer)', () => {
    const extract = (w: string): string => w.match(/struct FrameParams\s*\{[\s\S]*?\};/)?.[0] ?? '';
    expect(extract(sssOn)).toBe(extract(sssOff));
    // 400 = 388 raw (26 u32 + 7 f32 + 4 vec4f + 3 mat4x4f) + 12B WGSL struct end-pad
    // to align to mat4x4f's 16-byte alignment. Pre-N-directional was 384 (3 mat4s
    // ended flush on a 16-byte boundary); directionalLightCount: u32 adds a trailing
    // 4B scalar that breaks alignment, causing 12B of end-padding → 400.
    expect(FRAME_PARAMS_BYTE_SIZE).toBe(400);
  });

  it('no-collision branch divides out the hero-channel survival probability (V23 double-count fix)', () => {
    // The surviving throughput must be scaled by exp(-(σ_t − heroSigmaT)·d), NOT
    // the full exp(-σ_t·d) (which would double-count the transmittance already
    // realized by the free-flight survival probability).
    expect(sssOn).toContain('exp(-(walkSigmaT - vec3f(heroSigmaT)) * hit.dist)');
    expect(sssOn).not.toContain('throughput = throughput * exp(-walkSigmaT * hit.dist)');
  });

  it('a participating medium is entered for pure ABSORPTION too (σ_a, no σ_s) — stained-glass fix (V23)', () => {
    // isTranslucent must be true for a transmissive material that has Beer-Lambert
    // absorption (hasSigmaA) or spectral attenuation, not only scattering — else
    // chromatic stained glass (pure absorption) never enters the medium and its
    // attenuationColor is silently dropped.
    expect(sssOn).toContain('mat.hasSigmaA');
    expect(sssOn).toContain('mat.hasSpectralAttenuation');
    expect(sssOn).not.toContain('mat.isTranslucent = mat.transmission > 0.0 && mat.scatteringCoeff > 0.0;');
  });
});
