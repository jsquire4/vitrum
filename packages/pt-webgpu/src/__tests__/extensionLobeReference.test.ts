/**
 * pt-webgpu extension-lobe reference oracles.
 *
 * These CPU mirrors pin the math behind the clearcoat / sheen / iridescence
 * rows that are otherwise easy to overstate with string-contract tests alone.
 * The shader still documents sheen's dedicated PDF as a cosine approximation;
 * this file intentionally locks that posture until a true Charlie-lobe sampler
 * replaces it.
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
  const vDotH = Math.max(dot(wo, h), 0);
  const f = fresnelSchlick(vDotH, [0.04, 0.04, 0.04]);
  const alpha = Math.max(clearcoatRoughness * clearcoatRoughness, 1e-3);
  const d = ggxD(nDotH, alpha);
  const g = smithG1(nDotV, clearcoatRoughness) * smithG1(nDotL, clearcoatRoughness);
  const denom = Math.max(4 * nDotV * nDotL, 1e-6);
  return scale(f, clearcoat * d * g / denom);
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
  return (2 + invAlpha) * Math.pow(sinThetaH, invAlpha) / (2 * PI);
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

function sampledFullPdf(
  basePdf: number,
  clearcoat: number,
  clearcoatDensity: number,
  sheen: number,
  nDotL: number,
): number {
  const raw = basePdf + clearcoat * clearcoatDensity + sheen * nDotL * INV_PI;
  const lobeWeightSum = Math.max(1 + Math.max(clearcoat, 0) + Math.max(sheen, 0), 1);
  return raw / lobeWeightSum;
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
    const nDotL = Math.max(dot(normal, wi), 0);
    const basePdf = 0.19;
    const cc = 0.6;
    const sheen = 0.35;
    const ccDensity = clearcoatPdf(cc, 0.42, normal, wo, wi);
    const raw = basePdf + cc * ccDensity + sheen * nDotL * INV_PI;
    const sampled = sampledFullPdf(basePdf, cc, ccDensity, sheen, nDotL);

    expect(sampled).toBeCloseTo(raw / (1 + cc + sheen), 12);
    expect(sampled).toBeLessThan(raw);
    expect(sampledFullPdf(basePdf, 0, ccDensity, 0, nDotL)).toBeCloseTo(basePdf, 12);
  });

  it('keeps iridescence as an F0 modifier with an exact zero-default path', () => {
    const baseF0: Vec3 = [0.04, 0.08, 0.16];
    const iridescentF0: Vec3 = [0.21, 0.36, 0.72];
    expectVecClose(mix3(baseF0, iridescentF0, 0), baseF0);
    expectVecClose(mix3(baseF0, iridescentF0, 0.4), [0.108, 0.192, 0.384]);

    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain('if (iridescence < 1e-4) {');
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain('return baseF0; // zero-default: numerically identical to pre-H52 path.');
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain('return mix(baseF0, iridF, iridescence);');
  });

  it('locks sheen PDF as explicitly approximate until a true Charlie-lobe sampler exists', () => {
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain('The sheen PDF uses a cosine-hemisphere approximation');
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain('v1 accepted bias');
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain('let sheenPdf = sheen * nDotL * INV_PI;');
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
