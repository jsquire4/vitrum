import { describe, expect, it } from 'vitest';

import {
  PT_WEBGPU_MICROFACET_ALPHA_FLOOR,
  ROUGH_DIELECTRIC_SMOOTH_THRESHOLD,
  ptWebgpuMicrofacetAlpha,
  roughDielectricGgxD,
  roughDielectricTransmissionEval,
  roughDielectricTransmissionPdf,
  sampleRoughDielectric,
} from '../math/roughDielectric.js';
import { PT_WEBGPU_PATH_TRACE_BSDF_WGSL } from '../wgsl/pathTrace/bsdf.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CONNECT_WGSL } from '../wgsl/pathTrace/connect.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CONNECT_LITE_WGSL } from '../wgsl/pathTrace/connectLite.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_LITE_WGSL } from '../wgsl/pathTrace/kernelLite.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL } from '../wgsl/pathTrace/material.wgsl.js';
import { RESTIR_PT_PRODUCER_WGSL } from '../wgsl/pathTrace/restirPtProducer.wgsl.js';

describe('positive authored feature semantics', () => {
  it('classifies only exact-zero roughness as a delta interface', () => {
    expect(ROUGH_DIELECTRIC_SMOOTH_THRESHOLD).toBe(0);
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      'return roughness <= 0;',
    );

    const normal = [0, 0, 1] as const;
    const tinyFinite = { roughness: 1e-7, etaTOverI: 1.5 };
    const sampled = sampleRoughDielectric(
      tinyFinite,
      normal,
      normal,
      1,
      0.37,
      0.61,
    );
    expect(sampled.wi).not.toBeNull();
    const wi = sampled.wi!;
    const pdf = roughDielectricTransmissionPdf(
      tinyFinite,
      normal,
      normal,
      wi,
    );
    const value = roughDielectricTransmissionEval(
      tinyFinite,
      normal,
      normal,
      wi,
      false,
    );
    expect(Number.isFinite(pdf)).toBe(true);
    expect(Number.isFinite(value)).toBe(true);
    expect(pdf).toBeGreaterThan(0);
    expect(value).toBeGreaterThan(0);

    const delta = { roughness: 0, etaTOverI: 1.5 };
    expect(roughDielectricTransmissionPdf(delta, normal, normal, wi)).toBe(0);
    expect(
      roughDielectricTransmissionEval(delta, normal, normal, wi, false),
    ).toBe(0);
  });

  it('shares one finite-alpha contract and retains the narrow GGX peak', () => {
    expect(PT_WEBGPU_MICROFACET_ALPHA_FLOOR).toBe(1e-3);
    expect(ptWebgpuMicrofacetAlpha(0)).toBe(
      PT_WEBGPU_MICROFACET_ALPHA_FLOOR,
    );
    expect(ptWebgpuMicrofacetAlpha(1e-7)).toBe(
      PT_WEBGPU_MICROFACET_ALPHA_FLOOR,
    );
    expect(ptWebgpuMicrofacetAlpha(0.5)).toBeCloseTo(0.25, 15);

    const peak = roughDielectricGgxD(
      1,
      PT_WEBGPU_MICROFACET_ALPHA_FLOOR,
    );
    expect(Number.isFinite(peak)).toBe(true);
    expect(peak).toBeGreaterThan(1e5);
    expect(peak).toBeCloseTo(
      1 / (Math.PI * PT_WEBGPU_MICROFACET_ALPHA_FLOOR ** 2),
      8,
    );

    for (const source of [PT_WEBGPU_PATH_TRACE_BSDF_WGSL]) {
      expect(source).toContain('max(roughness * roughness, 0.001)');
    }
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL).toContain(
      'let d = (1.0 - n2) + n2 * a2;',
    );
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL).not.toContain(
      'return a2 / max(PI * d * d, 1e-6);',
    );
  });

  it('does not erase tiny clearcoat, sheen, iridescence, or anisotropy weights', () => {
    for (const source of [PT_WEBGPU_PATH_TRACE_BSDF_WGSL]) {
      expect(source).not.toMatch(/(?:clearcoat|sheen|iridescence) < 1e-/);
      expect(source).not.toMatch(/if \(anisotropy (?:>|<=) 1e-/);
    }
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      'if (clearcoat <= 0.0)',
    );
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain('if (sheen <= 0.0)');
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      'if (iridescence <= 0.0)',
    );
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      'if (anisotropy > 0.0)',
    );
  });

  it('keeps every positive transmission out of opaque-only classifications', () => {
    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      'if (transmission > 0.0) { return false; }',
    );
    expect(RESTIR_PT_PRODUCER_WGSL).not.toContain(
      'if (transmission > 1e-5)',
    );
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL).toContain(
      'mat.doubleSided || mat.transmission > 0.0',
    );
  });

  it('routes procedural skies through their baked environment map only', () => {
    for (const source of [
      PT_WEBGPU_PATH_TRACE_CONNECT_WGSL,
      PT_WEBGPU_PATH_TRACE_CONNECT_LITE_WGSL,
      PT_WEBGPU_PATH_TRACE_KERNEL_LITE_WGSL,
      RESTIR_PT_PRODUCER_WGSL,
    ]) {
      expect(source).not.toContain('environmentSun.w');
      expect(source).not.toContain('sampleSky');
    }
  });

  it('does not erase tiny positive directional irradiance in lite or ReSTIR suffix paths', () => {
    for (const source of [PT_WEBGPU_PATH_TRACE_KERNEL_LITE_WGSL, RESTIR_PT_PRODUCER_WGSL]) {
      expect(source).toContain('dIrrMean.w > 0.0');
      expect(source).not.toContain('dIrrMean.w > 1e-6');
    }
  });
});
