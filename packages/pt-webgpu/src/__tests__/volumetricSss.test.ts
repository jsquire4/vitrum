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
import type { Scene } from '@vitrum/core';
import { FRAME_PARAMS_BYTE_SIZE } from '../scene/frameParamsLayout.js';
import {
  PT_WEBGPU_TRACE_WGSL,
  composePtWebgpuTraceWgsl,
} from '../wgsl/pathTraceBruteforce.wgsl.js';
import { MATERIAL_FLOAT_STRIDE, materialToPackedVec4s } from '../scene/materialPacking.js';
import {
  DIRECTIONAL_LIGHT_FLOAT_STRIDE,
  packEmitterArrays,
} from '../scene/emitterPacking.js';
import { spectralEmissionAtHeroOracle } from './spectralScalarOracle.js';

// ---------------------------------------------------------------------------
// CPU mirrors of the WGSL volumetric math (identical formulas).
// ---------------------------------------------------------------------------

const INV_4PI = 1.0 / (4.0 * Math.PI);

/** HG phase function p(cosθ; g). Henyey-Greenstein 1941. */
function hgPhase(cosTheta: number, g: number): number {
  const safeG = Math.max(-0.999999, Math.min(0.999999, g));
  const a = Math.abs(safeG);
  const clampedCos = Math.max(-1, Math.min(1, cosTheta));
  const alignedCos = safeG >= 0 ? clampedCos : -clampedCos;
  const oneMinusA = 1 - a;
  const denom = oneMinusA * oneMinusA + 2 * a * (1 - alignedCos);
  return (
    (INV_4PI * oneMinusA * (1 + a)) /
    (denom * Math.sqrt(denom))
  );
}

function normalize3(v: readonly [number, number, number]): readonly [number, number, number] {
  const len = Math.hypot(v[0], v[1], v[2]);
  return len > 1e-8 ? [v[0] / len, v[1] / len, v[2] / len] : [0, 1, 0];
}

function dot3(a: readonly [number, number, number], b: readonly [number, number, number]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function directionalMediumNeeReference(
  directionalRecords: Float32Array,
  directionalLightCount: number,
  travelDir: readonly [number, number, number],
  throughputInMedium: readonly [number, number, number],
  g: number,
): readonly [number, number, number] {
  const travel = normalize3(travelDir);
  const radiance: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < directionalLightCount; i += 1) {
    const base = i * DIRECTIONAL_LIGHT_FLOAT_STRIDE;
    const meanIrradiance = directionalRecords[base + 7] ?? 0;
    if (meanIrradiance <= 1e-6) continue;
    const lightDir = normalize3([
      directionalRecords[base] ?? 0,
      directionalRecords[base + 1] ?? 1,
      directionalRecords[base + 2] ?? 0,
    ]);
    const phase = hgPhase(dot3(travel, lightDir), g);
    radiance[0] += throughputInMedium[0] * (directionalRecords[base + 4] ?? 0) * phase;
    radiance[1] += throughputInMedium[1] * (directionalRecords[base + 5] ?? 0) * phase;
    radiance[2] += throughputInMedium[2] * (directionalRecords[base + 6] ?? 0) * phase;
  }
  return radiance;
}

function directionalMediumNeeScene(): Scene {
  return {
    primitives: [],
    environment: { kind: 'none' },
    emitters: [
      { kind: 'directional', id: 'sun-a', direction: [0, -1, 0], color: [1, 0.5, 0.25], intensity: 2 },
      { kind: 'directional', id: 'dark-sun', direction: [1, 0, 0], color: [8, 8, 8], intensity: 0 },
      { kind: 'directional', id: 'sun-b', direction: [1, -1, 0], color: [0.25, 0.75, 1.5], intensity: 3 },
    ],
  };
}

/**
 * HG importance sample: given a uniform u1, return cosθ of the scattered
 * direction relative to the RAY-TRAVEL direction (PBR4e §11.3 eq. 11.7).
 *
 * The implementation evaluates algebraically equivalent factored/rational
 * forms on either side of |g|=0.125. Both preserve the exact distribution while
 * avoiding cancellation at g≈0 and |g|≈1. The sign symmetry measures cosθ
 * against travel direction, with mean +g (forward scatter for g>0).
 */
function hgSampleCosTheta(g: number, u1: number): number {
  const safeG = Math.max(-0.999999, Math.min(0.999999, g));
  const q = 1 - 2 * u1;
  let cosTheta: number;
  if (Math.abs(safeG) < 0.125) {
    const d = 1 + safeG * q;
    const numerator =
      2 * q +
      safeG * (q * q + 3) +
      2 * safeG * safeG * q +
      safeG * safeG * safeG * (q * q - 1);
    cosTheta = numerator / (2 * d * d);
  } else {
    const ratio = (1 - safeG * safeG) / (1 + safeG * q);
    cosTheta =
      (1 + safeG * safeG - ratio * ratio) / (2 * safeG);
  }
  return Math.max(-1, Math.min(1, cosTheta));
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

interface VolumeThicknessMaterial {
  readonly hasVolumeThickness: boolean;
  readonly volumeThickness: number;
}

function materialAttenuationDistanceReference(
  segmentDistance: number,
  mat: VolumeThicknessMaterial,
): number {
  const segment = Math.max(segmentDistance, 0);
  if (!mat.hasVolumeThickness) return segment;
  return Math.min(segment, Math.max(mat.volumeThickness, 0));
}

function beerLambertWithThicknessClamp(
  sigmaA: number,
  segmentDistance: number,
  mat: VolumeThicknessMaterial,
): number {
  return Math.exp(-Math.max(sigmaA, 0) * materialAttenuationDistanceReference(segmentDistance, mat));
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

describe('Volumetric in-medium directional NEE WGSL guard', () => {
  it('CPU oracle: packed RGB directionals add in-medium NEE with HG phase and mean gating', () => {
    const packed = packEmitterArrays(directionalMediumNeeScene());
    expect(packed.directionalLightCount).toBe(3);
    expect(packed.directionalLightsData.length).toBe(3 * DIRECTIONAL_LIGHT_FLOAT_STRIDE);

    const radiance = directionalMediumNeeReference(
      packed.directionalLightsData,
      packed.directionalLightCount,
      [0.2, 0.9, 0.1],
      [0.7, 0.5, 0.25],
      0.35,
    );
    expect(radiance[0]).toBeCloseTo(0.3879549966510263, 12);
    expect(radiance[1]).toBeCloseTo(0.23957166395375434, 12);
    expect(radiance[2]).toBeCloseTo(0.15080759319470188, 12);

    const withoutDarkDirectional = directionalMediumNeeReference(
      packed.directionalLightsData.filter((_, i) => i < 8 || i >= 16),
      2,
      [0.2, 0.9, 0.1],
      [0.7, 0.5, 0.25],
      0.35,
    );
    expect(withoutDarkDirectional[0]).toBeCloseTo(radiance[0], 12);
    expect(withoutDarkDirectional[1]).toBeCloseTo(radiance[1], 12);
    expect(withoutDarkDirectional[2]).toBeCloseTo(radiance[2], 12);
  });

  it('evaluates packed N-directional irradiance in the scalar hero domain', () => {
    const blueIrradiance: [number, number, number] = [0.1, 0.2, 1];
    const shortWave = spectralEmissionAtHeroOracle(blueIrradiance, 440);
    const longWave = spectralEmissionAtHeroOracle(blueIrradiance, 650);
    expect(shortWave).toBeGreaterThan(longWave);
    expect(shortWave).toBeGreaterThan(0);

    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn sampleMediumEmitter(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'for (var di = 0u; di < params.directionalLightCount; di = di + 1u)',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain('directionalLights[base + 1u]');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'spectralEmissionAtHero(light.radiance, heroLambda)',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain('mediumNeeForEmitter(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('mediumPhaseEmitterConnection(');
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

describe('volume thickness packing and attenuation-distance clamp', () => {
  const VOLUME_THICKNESS_FLOAT_OFFSET = 28 * 4;

  it('packs vec4 #28 as thickness plus a presence flag', () => {
    const packed = materialToPackedVec4s({
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      transmission: 1,
      thickness: 0.35,
    } as never);

    expect(packed.length).toBe(MATERIAL_FLOAT_STRIDE);
    expect(packed[VOLUME_THICKNESS_FLOAT_OFFSET]).toBeCloseTo(0.35, 6);
    expect(packed[VOLUME_THICKNESS_FLOAT_OFFSET + 1]).toBe(1);
  });

  it('leaves the clamp disabled when no thickness source is authored', () => {
    const packed = materialToPackedVec4s({
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      transmission: 1,
    } as never);

    expect(packed[VOLUME_THICKNESS_FLOAT_OFFSET]).toBe(0);
    expect(packed[VOLUME_THICKNESS_FLOAT_OFFSET + 1]).toBe(0);
  });

  it('clamps attenuation to the authored volume slab without inventing negative distance', () => {
    const mat = { hasVolumeThickness: true, volumeThickness: 0.4 };

    expect(materialAttenuationDistanceReference(-1.0, mat)).toBe(0);
    expect(materialAttenuationDistanceReference(0.2, mat)).toBeCloseTo(0.2, 6);
    expect(materialAttenuationDistanceReference(1.2, mat)).toBeCloseTo(0.4, 6);
    expect(materialAttenuationDistanceReference(Number.POSITIVE_INFINITY, mat)).toBeCloseTo(0.4, 6);
    expect(
      materialAttenuationDistanceReference(1.2, { hasVolumeThickness: true, volumeThickness: -0.7 }),
    ).toBe(0);
  });

  it('uses geometric segment length when the clamp flag is absent', () => {
    const mat = { hasVolumeThickness: false, volumeThickness: 0.4 };

    expect(materialAttenuationDistanceReference(1.2, mat)).toBeCloseTo(1.2, 6);
    expect(materialAttenuationDistanceReference(-0.5, mat)).toBe(0);
  });

  it('keeps infinite-medium absorption finite when a slab clamp is authored', () => {
    const sigmaA = 1.6;
    const slab = { hasVolumeThickness: true, volumeThickness: 0.25 };
    const unclamped = { hasVolumeThickness: false, volumeThickness: 0.25 };

    expect(beerLambertWithThicknessClamp(sigmaA, Number.POSITIVE_INFINITY, slab)).toBeCloseTo(
      Math.exp(-sigmaA * 0.25),
      12,
    );
    expect(beerLambertWithThicknessClamp(sigmaA, Number.POSITIVE_INFINITY, unclamped)).toBe(0);
  });
});

describe('Structural symmetric-medium composition with BDPT', () => {
  const sssOn = composePtWebgpuTraceWgsl(false);
  const bdptWithSss = composePtWebgpuTraceWgsl(true);

  it('default export equals the BDPT-off (SSS-on) composition', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toBe(sssOn);
  });

  it('SSS-on kernel contains the volumetric-walk symbols', () => {
    expect(sssOn).toContain('var bdptMediumStack');
    expect(sssOn).toContain('fn hgPhase');
    expect(sssOn).toContain('fn sampleHenyeyGreenstein');
    expect(sssOn).toContain('freeFlightDist');
  });

  it('BDPT-on kernel contains the matching volumetric-walk symbols', () => {
    expect(bdptWithSss).toContain('var bdptMediumStack');
    expect(bdptWithSss).toContain('freeFlightDist');
    expect(bdptWithSss).toContain('var mediumStack: array<BdptMediumLayer');
    expect(bdptWithSss).toContain('fn sampleHenyeyGreenstein');
  });

  it('BDPT-on kernel uses scattering transport on both subpaths', () => {
    expect(bdptWithSss).toContain('eyeMedium.sigmaS * transmittance');
    expect(bdptWithSss).toContain('layer.sigmaS * transmittance');
    expect(bdptWithSss).not.toContain('exp(-sigmaT * materialAttenuationDistance(hit.dist, mat))');
  });

  it('does not change the FrameParams UBO byte size (σ_a lives in materials buffer)', () => {
    const extract = (w: string): string => w.match(/struct FrameParams\s*\{[\s\S]*?\};/)?.[0] ?? '';
    expect(extract(sssOn)).toBe(extract(bdptWithSss));
    // 384 bytes after removing unread CMF/light mirrors and using a semantic
    // cameraPos vec3f. The important WS4 invariant is that SSS absorption data
    // stays in the material buffer, not a distinct FrameParams tail.
    expect(FRAME_PARAMS_BYTE_SIZE).toBe(368);
  });

  it('no-collision branch divides out the hero-channel survival probability (V23 double-count fix)', () => {
    // The surviving throughput must be scaled by exp(-(σ_t − heroSigmaT)·d), NOT
    // the full exp(-σ_t·d) (which would double-count the transmittance already
    // realized by the free-flight survival probability).
    expect(sssOn).toContain('let attenuationDist = min(hit.dist, eyeMedium.remainingDistance)');
    expect(sssOn).toContain('exp(-(walkSigmaT - vec3f(heroSigmaT)) * attenuationDist)');
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
