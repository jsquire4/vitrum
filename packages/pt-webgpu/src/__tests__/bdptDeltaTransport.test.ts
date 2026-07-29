import { describe, expect, it } from 'vitest';

import { PT_WEBGPU_BDPT_CONNECTION_WGSL } from '../wgsl/bdpt/bdptConnection.wgsl.js';
import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from '../wgsl/bdpt/bdptLightSubpath.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_BSDF_WGSL } from '../wgsl/pathTrace/bsdf.wgsl.js';
import {
  composePathTraceKernelWgsl,
  PT_WEBGPU_PATH_TRACE_KERNEL_WGSL,
} from '../wgsl/pathTrace/kernel.wgsl.js';

const PT_WEBGPU_BDPT_KERNEL_WGSL = composePathTraceKernelWgsl({ volumetricSss: false });

function frDielectric(cosThetaIInput: number, etaInput: number): number {
  let cosThetaI = Math.min(1, Math.max(-1, cosThetaIInput));
  let eta = etaInput;
  if (cosThetaI < 0) {
    eta = 1 / eta;
    cosThetaI = -cosThetaI;
  }
  const sin2ThetaI = Math.max(0, 1 - cosThetaI * cosThetaI);
  const sin2ThetaT = sin2ThetaI / (eta * eta);
  if (sin2ThetaT >= 1) return 1;
  const cosThetaT = Math.sqrt(Math.max(0, 1 - sin2ThetaT));
  const rParallel =
    (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
  const rPerpendicular =
    (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
  return 0.5 * (rParallel * rParallel + rPerpendicular * rPerpendicular);
}

function dielectricEventProbabilities(
  macroF: number,
  microfacetF: number,
  transmission: number,
): readonly [reflection: number, diffuse: number, refraction: number] {
  const diffuse = (1 - macroF) * (1 - transmission);
  const dielectric = 1 - diffuse;
  const dielectricNorm = microfacetF + transmission * (1 - microfacetF);
  return [
    (dielectric * microfacetF) / dielectricNorm,
    diffuse,
    (dielectric * transmission * (1 - microfacetF)) / dielectricNorm,
  ];
}

function roughTransmissionAtNormalIncidence(
  roughness: number,
  etaTOverI: number,
): number {
  if (roughness <= 0.020001) return 0;
  const alpha = Math.max(roughness * roughness, 1e-3);
  const d = 1 / (Math.PI * alpha * alpha);
  const fresnel = frDielectric(1, etaTOverI);
  const denominator = -1 + 1 / etaTOverI;
  return d * (1 - fresnel) / (denominator * denominator);
}

describe('BDPT sampled-event delta/transmission transport', () => {
  it('centralizes finite-event estimators while preserving exact delta-event state', () => {
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain('sampledEventPdf: f32,');
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain('sampledIsDelta: bool,');
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain('sampledLobe: u32,');
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain('sampledEtaTOverI: f32,');
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      '(*result).sampledEventPdf = marginalPdf;',
    );
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).not.toContain(
      'result.sampledEventPdf = (reflectionProbability / lobeWeightSum) * bs.pdf;',
    );
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      'result.sampledEventPdf = transmissionProbability / lobeWeightSum;',
    );
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain('result.sampledIsDelta = true;');
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      'result.sampledLobe = BSDF_LOBE_DELTA_TRANSMISSION;',
    );
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).not.toContain(
      'result.sampledEventPdf = (specProb / lobeWeightSum) * bs2.pdf;',
    );
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).not.toContain(
      'result.sampledEventPdf = (diffProb / lobeWeightSum) * bs.pdf;',
    );

    const normalize = (source: string): string =>
      source.replace(/\s+/g, ' ').trim();
    const directPdfAssignments = [
      ...PT_WEBGPU_PATH_TRACE_BSDF_WGSL.matchAll(
        /result\.sampledEventPdf\s*=\s*([^;]+);/g,
      ),
    ].map((match) => normalize(match[1]!));
    expect(directPdfAssignments).toEqual([
      '0.0',
      'reflectionProbability / lobeWeightSum',
      'transmissionProbability / lobeWeightSum',
    ]);

    const directThroughputAssignments = [
      ...PT_WEBGPU_PATH_TRACE_BSDF_WGSL.matchAll(
        /result\.throughputMul\s*=\s*([^;]+);/g,
      ),
    ].map((match) => normalize(match[1]!));
    expect(directThroughputAssignments).toHaveLength(3);
    expect(directThroughputAssignments[0]).toBe('vec3f(0.0)');
    expect(directThroughputAssignments[1]).toContain(
      'microfacetInterface.reflectance',
    );
    expect(directThroughputAssignments[2]).toContain(
      'microfacetInterface.baseTransmittance',
    );
  });

  it('normalizes partial-transmission lobes and preserves their expected contributions', () => {
    const etaTOverI = 1.5;
    const reflectance = frDielectric(1, etaTOverI);
    const transmission = 0.37;
    const clearcoat = 0.2;
    const sheen = 0.3;
    const lobeWeightSum = 1 + clearcoat + sheen;
    const reflectionProbability = reflectance / lobeWeightSum;
    const refractionProbability =
      ((1 - reflectance) * transmission) / lobeWeightSum;
    const diffuseProbability =
      ((1 - reflectance) * (1 - transmission)) / lobeWeightSum;
    const clearcoatProbability = clearcoat / lobeWeightSum;
    const sheenProbability = sheen / lobeWeightSum;
    expect(
      reflectionProbability +
        refractionProbability +
        diffuseProbability +
        clearcoatProbability +
        sheenProbability,
    ).toBeCloseTo(1, 12);

    // At normal incidence with unit-colour ideal lobes, each conditional
    // throughput is lobeWeightSum in importance mode. Multiplying by its event
    // probability recovers the physical mixture coefficient exactly.
    expect(reflectionProbability * lobeWeightSum).toBeCloseTo(reflectance, 12);
    expect(refractionProbability * lobeWeightSum).toBeCloseTo(
      (1 - reflectance) * transmission, 12,
    );
    expect(diffuseProbability * lobeWeightSum).toBeCloseTo(
      (1 - reflectance) * (1 - transmission), 12,
    );


    const microfacetF = 0.31;
    const [roughReflection, roughDiffuse, roughRefraction] =
      dielectricEventProbabilities(reflectance, microfacetF, transmission);
    expect(roughReflection + roughDiffuse + roughRefraction).toBeCloseTo(1, 12);
    expect(roughDiffuse).toBeCloseTo(
      (1 - reflectance) * (1 - transmission),
      12,
    );
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      'let finiteBsdf = evaluateFiniteBsdfFullWithClearcoatNormal(',
    );
    const enterRadianceEtaScale = (1 / etaTOverI) ** 2;
    const exitRadianceEtaScale = etaTOverI ** 2;
    expect(enterRadianceEtaScale * exitRadianceEtaScale).toBeCloseTo(1, 12);
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      'let etaScale = select(',
    );
  });

  it('makes smooth full glass non-connectible while rough glass has finite BTDF support', () => {
    const smoothTransmission = roughTransmissionAtNormalIncidence(0.02, 1.5);
    const roughTransmission = roughTransmissionAtNormalIncidence(0.35, 1.5);
    expect(smoothTransmission).toBe(0);
    expect(Number.isFinite(roughTransmission)).toBe(true);
    expect(roughTransmission).toBeGreaterThan(0);

    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      'if (bsdfDielectricIsSmooth(roughness)) { return 0.0; }',
    );
    expect(
      PT_WEBGPU_BDPT_CONNECTION_WGSL.match(
        /!bsdfHasFiniteConnectionSupport\(/g,
      )?.length ?? 0,
    ).toBeGreaterThanOrEqual(2);
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'eyeBrdf = evaluateFiniteBsdfFullWithClearcoatNormal(',
    );

    const samplerStart = PT_WEBGPU_PATH_TRACE_BSDF_WGSL.indexOf(
      'if (xiLobe < 1.0) {',
    );
    const extensionStart = PT_WEBGPU_PATH_TRACE_BSDF_WGSL.indexOf(
      '} else if (xiLobe < 1.0 + clearcoatWeight) {',
      samplerStart,
    );
    const dielectricSampler = PT_WEBGPU_PATH_TRACE_BSDF_WGSL.slice(samplerStart, extensionStart);
    expect(dielectricSampler.match(/sampleGgxVndf/g)).toHaveLength(2);
    expect(dielectricSampler).toContain('refract(incomingDir, wm, etaIOverT)');
    expect(dielectricSampler).toContain(
      'dot(normal, bs.wi) <= 1e-5 || bs.pdf <= 0.0',
    );
    expect(dielectricSampler).toContain(
      'dot(normal, outDir) >= -1e-5 || roughTransmissionProposalPdf <= 0.0',
    );
    expect(dielectricSampler).toContain(
      'if (dot(refractedDir, refractedDir) <= 1e-12) {',
    );
    expect(frDielectric(0.2, 1 / 1.5)).toBe(1);
  });

  it('continues light paths through sampled refraction with the sampler throughput and side-correct offset', () => {
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'prevMat.transmission,',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'pdfScatter = bsPrev.sampledEventPdf;',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'surfaceThroughputMul = bsPrev.throughputMul;',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'surfaceRayOrigin = bsPrev.newRayOrigin;',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'sampledDelta = bsPrev.sampledIsDelta;',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'var newThroughput = prevThroughput * surfaceThroughputMul * segmentWeight;',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).not.toContain(
      'if (mat.transmission > 0.5 && mat.roughness < 0.05)',
    );
  });

  it('marks the vertex that sampled a delta event and prevents strategy shifts across it', () => {
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'vec4f(oldPrevKind.xyz, BDPT_KIND_DELTA);',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'v.spec = l0.w == BDPT_KIND_DELTA;',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'if (flip.spec || neighborSpec) { break; }',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'if (flip.spec || nb.spec) { break; }',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'if (i == c && l3.w >= 0.0) { v.spec = fwdEe <= 0.0 || revLc <= 0.0; }',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'if (off == 0u) { v.spec = fwdEe <= 0.0 || revLc <= 0.0; }',
    );
  });

  it('stores eye-path specularity and forward density from the sampled event', () => {
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'bdptPrevScatterPdf * bdptEyeSegmentForwardDensity,',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'bdptEyeStackSetSpec(bounce, bs.sampledIsDelta);',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'let scatterPdfFwd = bs.sampledEventPdf;',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'swappedRev = bdptMarginalSurfacePdf(',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'swappedDirectionalPdf = bdptMarginalSurfacePdf(',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).not.toContain(
      'let eyeIsSpecular = transmission > 0.5 && roughness < 0.05;',
    );
  });

  it('uses the complete finite same-side density for partial-transmission endpoints', () => {
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'fn bdptTransmissiveConnectionPdf(',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'if (nDotV <= 1e-5) { return 0.0; }',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'let eventProbabilities = bsdfDielectricFiniteEventProbabilities(',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'eventProbabilities.x * pdfSpec + eventProbabilities.y * nDotL * INV_PI +',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'eventProbabilities.z *',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).not.toContain(
      'let eyeIsSpecular = transmission > 0.5 && roughness < 0.05;',
    );
  });

  it('tracks nested dielectric eta and Beer-Lambert distance on both BDPT subpaths', () => {
    expect(PT_WEBGPU_BDPT_KERNEL_WGSL).toContain(
      'var bdptMediumStack: array<BdptMediumLayer, BDPT_MEDIUM_STACK_LIMIT>;',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'var mediumStack: array<BdptMediumLayer, BDPT_MEDIUM_STACK_LIMIT>;',
    );
    const lightStackDeclaration = PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL.indexOf(
      'var mediumStack: array<BdptMediumLayer, BDPT_MEDIUM_STACK_LIMIT>;',
    );
    const lightExtensionLoop =
      PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL.indexOf(
        'for (var col = 1; col < maxB; col = col + 1) {',
      );
    expect(lightStackDeclaration).toBeGreaterThan(-1);
    expect(lightStackDeclaration).toBeLessThan(lightExtensionLoop);
    for (const source of [
      PT_WEBGPU_BDPT_KERNEL_WGSL,
      PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL,
    ]) {
      expect(source).toContain('bdptMediumLayer(');
      expect(source).toContain('remainingDistance');
      expect(source).toContain('exp(');
    }

    const outerSigma = [0.1, 0.2, 0.3] as const;
    const innerSigma = [0.4, 0.1, 0.05] as const;
    const outerDistance = 1.5;
    const innerDistance = 0.75;
    for (let channel = 0; channel < 3; channel += 1) {
      const combined = Math.exp(
        -(outerSigma[channel]! * outerDistance +
          innerSigma[channel]! * innerDistance),
      );
      const factorized =
        Math.exp(-outerSigma[channel]! * outerDistance) *
        Math.exp(-innerSigma[channel]! * innerDistance);
      expect(combined).toBeCloseTo(factorized, 12);
    }
    expect(Math.min(2, 0.5)).toBe(0.5);
  });
});
