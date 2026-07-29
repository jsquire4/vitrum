import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_PATH_TRACE_BSDF_WGSL } from '../wgsl/pathTrace/bsdf.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_WGSL } from '../wgsl/pathTrace/kernel.wgsl.js';

function count(source: string, token: string): number {
  return source.split(token).length - 1;
}

function sliceFunction(name: string, nextName: string): string {
  const start = PT_WEBGPU_PATH_TRACE_BSDF_WGSL.indexOf(`fn ${name}(`);
  const end = PT_WEBGPU_PATH_TRACE_BSDF_WGSL.indexOf(
    `fn ${nextName}(`,
    start,
  );
  expect(start).toBeGreaterThanOrEqual(0);
  expect(end).toBeGreaterThan(start);
  return PT_WEBGPU_PATH_TRACE_BSDF_WGSL.slice(start, end);
}

function schlick04(cosTheta: number): number {
  return 0.04 + 0.96 * (1 - cosTheta) ** 5;
}

function clearcoatAttenuation(
  clearcoat: number,
  viewNormalCosine: number,
): number {
  return 1 - Math.min(1, Math.max(0, clearcoat)) *
    schlick04(Math.min(1, Math.max(0, Math.abs(viewNormalCosine))));
}

function sheenDirectionalAlbedo(cosTheta: number, alpha: number): number {
  const c = 1 - Math.min(1, Math.max(0, cosTheta));
  return 0.65584461 * c ** 3 +
    1 / (4.16526551 + Math.exp(-7.97291361 * Math.sqrt(alpha) + 6.33516894));
}

function sheenAttenuation(
  sheen: number,
  roughness: number,
  maxSheenColor: number,
  nDotV: number,
  nDotL: number,
): number {
  const alpha = Math.max(roughness, 0.07) ** 2;
  const fullScale = Math.min(
    1 - maxSheenColor * sheenDirectionalAlbedo(nDotV, alpha),
    1 - maxSheenColor * sheenDirectionalAlbedo(nDotL, alpha),
  );
  return 1 + (fullScale - 1) * Math.min(1, Math.max(0, sheen));
}

describe('pt-webgpu layered clearcoat and sheen energy', () => {
  it('keeps the zero-default identity and attenuates the lower layers once', () => {
    const base = 0.62;
    const sheenLobe = 0.18;
    const clearcoatLobe = 0.07;

    const zeroSheen = sheenAttenuation(0, 0.45, 0.8, 0.7, 0.55);
    const zeroClearcoat = clearcoatAttenuation(0, 0.8);
    expect(zeroSheen).toBe(1);
    expect(zeroClearcoat).toBe(1);
    expect((base * zeroSheen + 0) * zeroClearcoat + 0).toBe(base);

    const sheenScale = sheenAttenuation(0.85, 0.45, 0.8, 0.7, 0.55);
    const clearcoatScale = clearcoatAttenuation(0.9, 0.8);
    expect(sheenScale).toBeGreaterThanOrEqual(0);
    expect(sheenScale).toBeLessThan(1);
    expect(clearcoatScale).toBeGreaterThanOrEqual(0);
    expect(clearcoatScale).toBeLessThan(1);

    // Sheen attenuates only the base. The outer clearcoat attenuates the
    // base-plus-sheen result. Its own reflected lobe remains unattenuated.
    const layered =
      (base * sheenScale + sheenLobe) * clearcoatScale + clearcoatLobe;
    const accidentallyDoubleAttenuated =
      (base * sheenScale + sheenLobe * sheenScale) *
        clearcoatScale ** 2 +
      clearcoatLobe * clearcoatScale;
    expect(layered).toBeLessThan(base + sheenLobe + clearcoatLobe);
    expect(layered).not.toBeCloseTo(accidentallyDoubleAttenuated, 8);
  });

  it('uses the same ordered attenuation in opaque and transmissive same-side evaluation', () => {
    const opaque = sliceFunction(
      'evaluateBrdfFullWithClearcoatNormal',
      'evaluateFiniteSameSideBrdfFullWithClearcoatNormal',
    );
    const finite = sliceFunction(
      'evaluateFiniteBsdfFullWithClearcoatNormal',
      'bsdfDielectricFiniteEventProbabilities',
    );

    expect(opaque).toContain(
      'return (base * sheenAttenuation + sh) * clearcoatAttenuation + cc;',
    );
    expect(count(opaque, 'sheenBaseAttenuation(')).toBe(1);
    expect(count(opaque, 'clearcoatBaseAttenuation(')).toBe(1);

    const transmissiveSameSide = finite.slice(
      finite.indexOf('if (transmission > 0.0 && metallic == 0.0) {'),
      finite.indexOf(
        'return evaluateFiniteSameSideBrdfFullWithClearcoatNormal(',
      ),
    );
    expect(transmissiveSameSide).toContain(
      'return ((diffuse + specular) * sheenAttenuation + sh) *',
    );
    expect(transmissiveSameSide).toContain(
      'clearcoatAttenuation + cc;',
    );
    expect(count(transmissiveSameSide, 'sheenBaseAttenuation(')).toBe(1);
    expect(count(transmissiveSameSide, 'clearcoatBaseAttenuation(')).toBe(1);

    const roughTransmission = finite.slice(
      finite.indexOf(
        'dot(normal, wo) > 1e-5 && dot(normal, wi) < -1e-5',
      ),
      finite.indexOf(
        'if (transmission > 0.0 && metallic == 0.0) {',
        finite.indexOf(
          'dot(normal, wo) > 1e-5 && dot(normal, wi) < -1e-5',
        ),
      ),
    );
    expect(count(roughTransmission, 'sheenBaseAttenuation(')).toBe(1);
    expect(count(roughTransmission, 'clearcoatBaseAttenuation(')).toBe(1);
    expect(roughTransmission).toContain(
      'sheenAttenuation * clearcoatAttenuation;',
    );
    expect(roughTransmission).not.toContain('evalSheenLobe(');
    expect(roughTransmission).not.toContain('evalClearcoatLobe(');
  });

  it('uses the ratified view-normal Fresnel for coat reflection, attenuation, and emission', () => {
    const clearcoat = 0.8;
    const viewNormalCosine = 0.25;
    const unrelatedViewHalfCosine = 0.91;
    const expected = 1 - clearcoat * (
      0.04 + 0.96 * (1 - Math.abs(viewNormalCosine)) ** 5
    );
    expect(clearcoatAttenuation(clearcoat, viewNormalCosine)).toBeCloseTo(
      expected,
      12,
    );
    expect(clearcoatAttenuation(
      clearcoat,
      viewNormalCosine,
    )).not.toBeCloseTo(
      clearcoatAttenuation(clearcoat, unrelatedViewHalfCosine),
      4,
    );

    const layerWeight = sliceFunction(
      'clearcoatLayerWeight',
      'evalClearcoatLobe',
    );
    const lobe = sliceFunction(
      'evalClearcoatLobe',
      'clearcoatBaseAttenuation',
    );
    const attenuation = sliceFunction(
      'clearcoatBaseAttenuation',
      'clearcoatPdf',
    );
    expect(layerWeight).toContain(
      'abs(dot(clearcoatNormal, wo))',
    );
    expect(lobe).toContain(
      'clearcoatLayerWeight(clearcoat, normal, wo)',
    );
    expect(attenuation).toContain(
      'clearcoatLayerWeight(clearcoat, clearcoatNormal, wo)',
    );
    const sheenAttenuation = sliceFunction(
      'sheenBaseAttenuation',
      'charlieSheenPdf',
    );
    expect(sheenAttenuation).toContain(
      'clamp(abs(dot(normal, wi)), 0.0, 1.0)',
    );
    expect(sheenAttenuation).toContain(
      'clamp(abs(dot(normal, wo)), 0.0, 1.0)',
    );
    expect(lobe).not.toContain(
      'fresnelSchlick(vDotH, vec3f(0.04))',
    );
    expect(attenuation).not.toContain(
      'fresnelSchlick(vDotH, vec3f(0.04))',
    );

    const clearcoatNormalIndex = PT_WEBGPU_PATH_TRACE_KERNEL_WGSL.indexOf(
      'var clearcoatNormal = normal;',
    );
    const emissionAttenuationIndex = PT_WEBGPU_PATH_TRACE_KERNEL_WGSL.indexOf(
      '1.0 - clearcoatLayerWeight(mat.clearcoat, clearcoatNormal, -ray.direction)',
    );
    expect(clearcoatNormalIndex).toBeGreaterThanOrEqual(0);
    expect(emissionAttenuationIndex).toBeGreaterThan(clearcoatNormalIndex);
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'throughput * emitContribution * clearcoatEmissionAttenuation',
    );
  });

  it('replaces every sampled finite proposal with one full layered evaluation', () => {
    const finalizer = sliceFunction(
      'finalizeFiniteBounceSampleWithClearcoatNormal',
      'computeAnisotropicAxes',
    );
    expect(finalizer).toContain(
      'let finiteBsdf = evaluateFiniteBsdfFullWithClearcoatNormal(',
    );
    expect(finalizer).toContain(
      '(*result).throughputMul = finiteBsdf * cosine / marginalPdf;',
    );
    expect(count(finalizer, 'evaluateFiniteBsdfFullWithClearcoatNormal(')).toBe(
      1,
    );
    expect(finalizer).not.toContain('sheenBaseAttenuation(');
    expect(finalizer).not.toContain('clearcoatBaseAttenuation(');

    const sampler = sliceFunction(
      'sampleNextBounceDirectionWithClearcoatNormal',
      'sampleNextBounceDirection',
    );
    expect(count(
      sampler,
      'finalizeFiniteBounceSampleWithClearcoatNormal(',
    )).toBe(2);

    // Smooth dielectric reflection is the sole sampled path that bypasses the
    // finite finalizer, so it explicitly applies each lower-layer attenuation
    // exactly once.
    const smoothReflection = sampler.slice(
      sampler.indexOf('if (bsdfDielectricIsSmooth(roughness)) {'),
      sampler.indexOf('} else {', sampler.indexOf(
        'if (bsdfDielectricIsSmooth(roughness)) {',
      )),
    );
    expect(count(smoothReflection, 'sheenBaseAttenuation(')).toBe(1);
    expect(count(smoothReflection, 'clearcoatBaseAttenuation(')).toBe(1);
    expect(smoothReflection).toContain(
      'microfacetInterface.reflectance * sheenAttenuation *',
    );
    expect(smoothReflection).toContain(
      'clearcoatAttenuation * lobeWeightSum /',
    );

    const smoothTransmissionStart = sampler.indexOf(
      'if (bsdfDielectricIsSmooth(roughness)) {',
      sampler.indexOf(
        '// Refraction is asymmetric between radiance and importance transport',
      ),
    );
    const smoothTransmission = sampler.slice(
      smoothTransmissionStart,
      sampler.indexOf('} else {', smoothTransmissionStart),
    );
    expect(count(smoothTransmission, 'sheenBaseAttenuation(')).toBe(1);
    expect(count(smoothTransmission, 'clearcoatBaseAttenuation(')).toBe(1);
    expect(smoothTransmission).toContain(
      'sheenAttenuation * clearcoatAttenuation *',
    );
  });
});
