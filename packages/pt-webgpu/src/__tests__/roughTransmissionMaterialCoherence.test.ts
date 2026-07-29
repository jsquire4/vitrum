import { describe, expect, it } from 'vitest';

import {
  roughDielectricFresnel,
  roughDielectricMaterialEventProbabilities,
  roughDielectricMaterialFresnel,
} from '../math/roughDielectric.js';
import { PT_WEBGPU_BDPT_CONNECTION_WGSL } from '../wgsl/bdpt/bdptConnection.wgsl.js';
import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from '../wgsl/bdpt/bdptLightSubpath.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_BSDF_WGSL } from '../wgsl/pathTrace/bsdf.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CONNECT_WGSL } from '../wgsl/pathTrace/connect.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CONNECT_LITE_WGSL } from '../wgsl/pathTrace/connectLite.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_WGSL } from '../wgsl/pathTrace/kernel.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_LITE_WGSL } from '../wgsl/pathTrace/kernelLite.wgsl.js';
import { SPPM_PHOTON_PASS_WGSL } from '../wgsl/pathTrace/sppmBindings.wgsl.js';

describe('rough-transmission material BSDF coherence', () => {
  it('preserves exact IOR Fresnel for the no-op material controls', () => {
    for (const eta of [1 / 1.5, 1.1, 1.5, 2]) {
      for (const cosTheta of [1, 0.8, 0.35, 0.05]) {
        const expected = roughDielectricFresnel(cosTheta, eta);
        const actual = roughDielectricMaterialFresnel(
          cosTheta,
          eta,
          [0.04, 0.04, 0.04],
        );
        expect(actual[0]).toBeCloseTo(expected, 14);
        expect(actual[1]).toBeCloseTo(expected, 14);
        expect(actual[2]).toBeCloseTo(expected, 14);
      }
    }
  });

  it('applies coloured authored F0 and keeps the R/T event mixture normalized', () => {
    const eta = 1.5;
    const macro = roughDielectricMaterialFresnel(
      0.72,
      eta,
      [0.01, 0.04, 0.12],
    );
    const micro = roughDielectricMaterialFresnel(
      0.51,
      eta,
      [0.01, 0.04, 0.12],
    );
    const baseline = roughDielectricMaterialFresnel(
      0.51,
      eta,
      [0.04, 0.04, 0.04],
    );

    expect(micro[0]).toBeLessThan(baseline[0]);
    expect(micro[1]).toBeCloseTo(baseline[1], 14);
    expect(micro[2]).toBeGreaterThan(baseline[2]);
    for (const channel of [...macro, ...micro]) {
      expect(Number.isFinite(channel)).toBe(true);
      expect(channel).toBeGreaterThanOrEqual(0);
      expect(channel).toBeLessThanOrEqual(1);
    }

    const probabilities = roughDielectricMaterialEventProbabilities(
      0.67,
      macro,
      micro,
    );
    expect(
      probabilities.reflection +
        probabilities.diffuse +
        probabilities.transmission,
    ).toBeCloseTo(1, 14);
    expect(probabilities.reflection).toBeGreaterThan(0);
    expect(probabilities.diffuse).toBeGreaterThan(0);
    expect(probabilities.transmission).toBeGreaterThan(0);

    // The sampled-event estimators reproduce the same per-channel integrand
    // after their scalar event probabilities cancel.
    const transmission = 0.67;
    for (const reflectance of micro) {
      const sampledReflectionWeight =
        reflectance / probabilities.reflection;
      const sampledTransmissionWeight =
        transmission * (1 - reflectance) / probabilities.transmission;
      expect(
        probabilities.reflection * sampledReflectionWeight,
      ).toBeCloseTo(reflectance, 14);
      expect(
        probabilities.transmission * sampledTransmissionWeight,
      ).toBeCloseTo(transmission * (1 - reflectance), 14);
    }
    for (const reflectance of macro) {
      const sampledDiffuseWeight =
        (1 - reflectance) * (1 - transmission) / probabilities.diffuse;
      expect(
        probabilities.diffuse * sampledDiffuseWeight,
      ).toBeCloseTo((1 - reflectance) * (1 - transmission), 14);
    }
  });

  it('uses one material-Fresnel model in finite eval, PDF, and source sampling', () => {
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      'fn materialDielectricFresnel(',
    );
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      'fn materialDielectricLayeredInterface(',
    );
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      'let interfaceResponse = materialDielectricLayeredInterface(',
    );
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      'let microfacetInterface = materialDielectricLayeredInterface(',
    );
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      'clamp(luminance(microfacetInterface.reflectance), 0.0, 1.0);',
    );
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      'clamp(luminance(microfacetInterface.baseTransmittance), 0.0, 1.0);',
    );
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).not.toContain(
      'let microfacetF = frDielectric(abs(dot(wo, wm)), etaTOverI);',
    );

    for (const source of [
      PT_WEBGPU_PATH_TRACE_KERNEL_WGSL,
      PT_WEBGPU_PATH_TRACE_KERNEL_LITE_WGSL,
      PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL,
      SPPM_PHOTON_PASS_WGSL,
    ]) {
      const sampleCall = source.lastIndexOf('sampleNextBounceDirection');
      expect(sampleCall).toBeGreaterThan(-1);
      const call = source.slice(sampleCall, sampleCall + 1_500);
      expect(call).toContain('.iridescence');
      expect(call).toContain('.iridescenceIor');
      expect(call).toContain('.iridescenceThicknessMin');
      expect(call).toContain('.iridescenceThicknessMax');
      expect(call).toContain('.specularColor');
      expect(call).toContain('.specularIntensity');
    }
  });

  it('keeps full/lite connections and both BDPT endpoints on the same evaluator', () => {
    for (const source of [
      PT_WEBGPU_PATH_TRACE_CONNECT_WGSL,
      PT_WEBGPU_PATH_TRACE_CONNECT_LITE_WGSL,
      PT_WEBGPU_BDPT_CONNECTION_WGSL,
    ]) {
      expect(source).toContain('evaluateFiniteBsdfFullWithClearcoatNormal(');
      expect(source).toContain('iridescence');
      expect(source).toContain('iridescenceIor');
      expect(source).toContain('iridescenceThicknessMin');
      expect(source).toContain('iridescenceThicknessMax');
      expect(source).toContain('specularColor');
      expect(source).toContain('specularIntensity');
    }
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'let eventProbabilities = bsdfDielectricFiniteEventProbabilities(',
    );
  });
});
