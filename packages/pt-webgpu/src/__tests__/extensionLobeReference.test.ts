/**
 * pt-webgpu extension-lobe reference oracles.
 *
 * These CPU mirrors pin the math behind the clearcoat / sheen / iridescence
 * rows that are otherwise easy to overstate with string-contract tests alone.
 */
import { describe, expect, it } from 'vitest';

import { PT_WEBGPU_PATH_TRACE_BSDF_WGSL } from '../wgsl/pathTrace/bsdf.wgsl.js';

const PI = Math.PI;
const INV_PI = 1 / PI;

type Vec3 = readonly [number, number, number];

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function mul(a: Vec3, b: Vec3): Vec3 {
  return [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
}

function scale(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}

function normalize(v: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

function expectVecClose(actual: Vec3, expected: Vec3, precision = 8): void {
  expect(actual[0]).toBeCloseTo(expected[0], precision);
  expect(actual[1]).toBeCloseTo(expected[1], precision);
  expect(actual[2]).toBeCloseTo(expected[2], precision);
}

function fresnelSchlick(cosTheta: number, f0: Vec3): Vec3 {
  const m = Math.min(1, Math.max(0, 1 - cosTheta));
  const m5 = m * m * m * m * m;
  return [
    f0[0] + (1 - f0[0]) * m5,
    f0[1] + (1 - f0[1]) * m5,
    f0[2] + (1 - f0[2]) * m5,
  ];
}

function ggxD(nDotH: number, alpha: number): number {
  const a2 = alpha * alpha;
  const d = nDotH * nDotH * (a2 - 1) + 1;
  return a2 / Math.max(PI * d * d, 1e-6);
}

function smithG1(nDotV: number, roughness: number): number {
  const r = roughness + 1;
  const k = r * r * 0.125;
  return nDotV / Math.max(nDotV * (1 - k) + k, 1e-6);
}

function evalClearcoatLobe(
  clearcoat: number,
  clearcoatRoughness: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
): Vec3 {
  if (clearcoat < 1e-4) return [0, 0, 0];
  const nDotL = Math.max(dot(normal, wi), 0);
  const nDotV = Math.max(dot(normal, wo), 0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return [0, 0, 0];
  const h = normalize(add(wi, wo));
  const nDotH = Math.max(dot(normal, h), 0);
  const layerWeight = clearcoat *
    fresnelSchlick(Math.abs(dot(normal, wo)), [0.04, 0.04, 0.04])[0];
  const alpha = Math.max(clearcoatRoughness * clearcoatRoughness, 1e-3);
  const d = ggxD(nDotH, alpha);
  const g = smithG1(nDotV, clearcoatRoughness) * smithG1(nDotL, clearcoatRoughness);
  const denom = Math.max(4 * nDotV * nDotL, 1e-6);
  return [layerWeight * d * g / denom, layerWeight * d * g / denom, layerWeight * d * g / denom];
}

function clearcoatPdf(
  clearcoat: number,
  clearcoatRoughness: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
): number {
  if (clearcoat < 1e-4) return 0;
  const nDotV = Math.max(dot(normal, wo), 0);
  const nDotL = Math.max(dot(normal, wi), 0);
  if (nDotV <= 1e-5 || nDotL <= 1e-5) return 0;
  const h = normalize(add(wi, wo));
  const nDotH = Math.max(dot(normal, h), 0);
  const alpha = Math.max(clearcoatRoughness * clearcoatRoughness, 1e-3);
  const d = ggxD(nDotH, alpha);
  const g1Wo = smithG1(nDotV, clearcoatRoughness);
  return (d * g1Wo) / Math.max(4 * nDotV, 1e-6);
}

function charlieD(nDotH: number, alpha: number): number {
  const invAlpha = 1 / Math.max(alpha, 1e-4);
  const sinThetaH = Math.sqrt(Math.max(0, 1 - nDotH * nDotH));
  return (2 + invAlpha) * Math.pow(sinThetaH, invAlpha) * INV_PI * 0.5;
}

function sheenVisibility(nDotL: number, nDotV: number): number {
  return 1 / Math.max(4 * (nDotL + nDotV - nDotL * nDotV), 1e-6);
}

function evalSheenLobe(
  sheen: number,
  sheenRoughness: number,
  sheenColor: Vec3,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
): Vec3 {
  if (sheen < 1e-4) return [0, 0, 0];
  const nDotL = Math.max(dot(normal, wi), 0);
  const nDotV = Math.max(dot(normal, wo), 0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return [0, 0, 0];
  const h = normalize(add(wi, wo));
  const nDotH = Math.max(dot(normal, h), 0);
  const alpha = Math.max(sheenRoughness * sheenRoughness, 1e-3);
  return scale(sheenColor, sheen * charlieD(nDotH, alpha) * sheenVisibility(nDotL, nDotV));
}

function charlieSheenPdf(
  sheen: number,
  sheenRoughness: number,
  normal: Vec3,
  wo: Vec3,
  wi: Vec3,
): number {
  if (sheen < 1e-4) return 0;
  const nDotL = Math.max(dot(normal, wi), 0);
  const nDotV = Math.max(dot(normal, wo), 0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return 0;
  const h = normalize(add(wi, wo));
  const nDotH = Math.max(dot(normal, h), 0);
  const vDotH = Math.max(dot(wo, h), 1e-6);
  const alpha = Math.max(sheenRoughness * sheenRoughness, 1e-3);
  return (charlieD(nDotH, alpha) * nDotH) / Math.max(4 * vDotH, 1e-6);
}

function charlieSampleNdotH(u: number, sheenRoughness: number): number {
  const alpha = Math.max(sheenRoughness * sheenRoughness, 1e-3);
  const invAlpha = 1 / Math.max(alpha, 1e-4);
  const sinThetaH = Math.pow(u, 1 / (invAlpha + 2));
  return Math.sqrt(Math.max(0, 1 - sinThetaH * sinThetaH));
}

function sampledFullPdf(
  basePdf: number,
  clearcoat: number,
  clearcoatDensity: number,
  sheen: number,
  sheenDensity: number,
): number {
  const raw = basePdf + clearcoat * clearcoatDensity + sheen * sheenDensity;
  return raw / lobeWeightSum(clearcoat, sheen);
}

function lobeWeightSum(clearcoat: number, sheen: number): number {
  return Math.max(1 + Math.max(clearcoat, 0) + Math.max(sheen, 0), 1);
}

function sourceLobeProbabilities(clearcoat: number, sheen: number): {
  base: number;
  clearcoat: number;
  sheen: number;
} {
  const denom = lobeWeightSum(clearcoat, sheen);
  return {
    base: 1 / denom,
    clearcoat: Math.max(clearcoat, 0) / denom,
    sheen: Math.max(sheen, 0) / denom,
  };
}

function mix3(a: Vec3, b: Vec3, t: number): Vec3 {
  return [
    a[0] * (1 - t) + b[0] * t,
    a[1] * (1 - t) + b[1] * t,
    a[2] * (1 - t) + b[2] * t,
  ];
}

describe('pt-webgpu extension lobe CPU references', () => {
  const normal: Vec3 = [0, 0, 1];
  const wo = normalize([0.42, -0.11, 0.9]);
  const wi = normalize([-0.27, 0.38, 0.82]);

  it('clearcoat is zero-default and scales linearly with its scalar', () => {
    expectVecClose(evalClearcoatLobe(0, 0.35, normal, wo, wi), [0, 0, 0]);

    const half = evalClearcoatLobe(0.5, 0.35, normal, wo, wi);
    const full = evalClearcoatLobe(1, 0.35, normal, wo, wi);
    expect(half[0]).toBeGreaterThan(0);
    expectVecClose(half, scale(full, 0.5), 10);
  });

  it('sheen is zero-default, color-tinted, and scalar-linear', () => {
    const color: Vec3 = [0.8, 0.35, 0.12];
    expectVecClose(evalSheenLobe(0, 0.75, color, normal, wo, wi), [0, 0, 0]);

    const one = evalSheenLobe(1, 0.75, color, normal, wo, wi);
    const quarter = evalSheenLobe(0.25, 0.75, color, normal, wo, wi);
    expect(one[0]).toBeGreaterThan(one[1]);
    expect(one[1]).toBeGreaterThan(one[2]);
    expectVecClose(quarter, scale(one, 0.25), 10);
  });

  it('normalizes the sampled full PDF by the same base/clearcoat/sheen lobe weights as the source sampler', () => {
    const basePdf = 0.19;
    const cc = 0.6;
    const sheen = 0.35;
    const ccDensity = clearcoatPdf(cc, 0.42, normal, wo, wi);
    const sheenDensity = charlieSheenPdf(1, 0.7, normal, wo, wi);
    const raw = basePdf + cc * ccDensity + sheen * sheenDensity;
    const sampled = sampledFullPdf(basePdf, cc, ccDensity, sheen, sheenDensity);

    expect(sampled).toBeCloseTo(raw / (1 + cc + sheen), 12);
    expect(sampled).toBeLessThan(raw);
    expect(sampledFullPdf(basePdf, 0, ccDensity, 0, sheenDensity)).toBeCloseTo(basePdf, 12);
  });

  it('oracles source-lobe selection probabilities independently of WGSL string pins', () => {
    const weights = sourceLobeProbabilities(0.6, 0.25);
    expect(weights.base).toBeCloseTo(1 / 1.85, 12);
    expect(weights.clearcoat).toBeCloseTo(0.6 / 1.85, 12);
    expect(weights.sheen).toBeCloseTo(0.25 / 1.85, 12);
    expect(weights.base + weights.clearcoat + weights.sheen).toBeCloseTo(1, 12);

    const clamped = sourceLobeProbabilities(-0.4, 0.5);
    expect(clamped.base).toBeCloseTo(1 / 1.5, 12);
    expect(clamped.clearcoat).toBe(0);
    expect(clamped.sheen).toBeCloseTo(0.5 / 1.5, 12);
    expect(lobeWeightSum(-0.4, -0.25)).toBe(1);
  });

  it('oracles source-sampler throughput as the inverse of the selected mixture probability', () => {
    const clearcoat = 0.4;
    const sheen = 0.7;
    const denom = lobeWeightSum(clearcoat, sheen);
    const specProb = 0.31;
    const diffProb = 0.69;
    const fresnelR = 0.18;

    const opaqueSpecSelect = specProb / denom;
    const opaqueDiffSelect = diffProb / denom;
    const dielectricReflectSelect = fresnelR / denom;
    const dielectricRefractSelect = (1 - fresnelR) / denom;

    expect(opaqueSpecSelect * (denom / specProb)).toBeCloseTo(1, 12);
    expect(opaqueDiffSelect * (denom / diffProb)).toBeCloseTo(1, 12);
    expect(dielectricReflectSelect * (denom / fresnelR)).toBeCloseTo(1, 12);
    expect(dielectricRefractSelect * (denom / (1 - fresnelR))).toBeCloseTo(1, 12);

    const withoutDenominator = specProb * (denom / specProb);
    expect(withoutDenominator).toBeCloseTo(denom, 12);
    expect(withoutDenominator).toBeGreaterThan(1);
  });

  it('keeps iridescence as an F0 modifier with an exact zero-default path', () => {
    const baseF0: Vec3 = [0.04, 0.08, 0.16];
    const iridescentF0: Vec3 = [0.21, 0.36, 0.72];
    expectVecClose(mix3(baseF0, iridescentF0, 0), baseF0);
    expectVecClose(mix3(baseF0, iridescentF0, 0.4), [0.108, 0.192, 0.384]);

    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain('if (iridescence <= 0.0) {');
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain('return baseF0; // zero-default: numerically identical to pre-H52 path.');
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain('return mix(baseF0, iridF, iridescence);');
  });

  it('uses a Charlie half-vector sampler and matching sheen PDF', () => {
    const u = 0.37;
    const roughness = 0.7;
    const alpha = Math.max(roughness * roughness, 1e-3);
    const invAlpha = 1 / Math.max(alpha, 1e-4);
    const nDotH = charlieSampleNdotH(u, roughness);
    const sinThetaH = Math.sqrt(Math.max(0, 1 - nDotH * nDotH));
    expect(Math.pow(sinThetaH, invAlpha + 2)).toBeCloseTo(u, 12);

    const h = normalize(add(wi, wo));
    const expectedPdf = charlieD(Math.max(dot(normal, h), 0), alpha) *
      Math.max(dot(normal, h), 0) /
      Math.max(4 * Math.max(dot(wo, h), 1e-6), 1e-6);
    expect(charlieSheenPdf(1, roughness, normal, wo, wi)).toBeCloseTo(expectedPdf, 12);

    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain('fn charlieSheenPdf(');
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain('fn charlieSheenSample(');
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain('let sheenPdf = sheen * charlieSheenPdf(');
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain('let bs = charlieSheenSample(rng, -incomingDir, normal, tanT, tanB, sheenRoughness);');
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).not.toContain('let sheenPdf = sheen * nDotL * INV_PI;');
  });

  it('uses component-wise lobe math rather than collapsing colored lobes to luminance', () => {
    const cc = evalClearcoatLobe(0.7, 0.45, normal, wo, wi);
    const sheen = evalSheenLobe(0.55, 0.7, [0.2, 0.7, 1.0], normal, wo, wi);
    const combined = add(cc, sheen);
    expectVecClose(combined, add(cc, sheen), 12);
    expect(combined[2]).toBeGreaterThan(combined[0]);
    expect(mul(combined, [1, 0, 0])[0]).toBeCloseTo(combined[0], 12);
  });
});
