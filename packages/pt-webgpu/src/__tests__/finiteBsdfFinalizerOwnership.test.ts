import { describe, expect, it } from 'vitest';

import { composePtWebgpuTraceWgsl } from '../wgsl/pathTraceBruteforce.wgsl.js';
import { PT_WEBGPU_TRACE_LITE_WGSL } from '../wgsl/pathTraceBruteforceLite.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_BSDF_WGSL } from '../wgsl/pathTrace/bsdf.wgsl.js';

function sliceBetween(source: string, start: string, end: string): string {
  const startIndex = source.indexOf(start);
  const endIndex = source.indexOf(end, startIndex + start.length);
  expect(startIndex, start).toBeGreaterThanOrEqual(0);
  expect(endIndex, end).toBeGreaterThan(startIndex);
  return source.slice(startIndex, endIndex);
}

function count(source: string, needle: string): number {
  return source.split(needle).length - 1;
}

describe('finite BSDF finalizer ownership', () => {
  it('does not calculate proposal-local continuous-specular estimators that are overwritten', () => {
    const sampler = sliceBetween(
      PT_WEBGPU_PATH_TRACE_BSDF_WGSL,
      'fn sampleNextBounceDirectionWithClearcoatNormal(',
      'fn sampleNextBounceDirection(',
    );

    expect(sampler).not.toContain(
      'result.sampledEventPdf = (reflectionProbability / lobeWeightSum) * bs.pdf;',
    );
    expect(sampler).not.toContain(
      'result.sampledEventPdf = (specProb / lobeWeightSum) * bs2.pdf;',
    );
    expect(sampler).toContain('let roughTransmissionProposalPdf =');
    expect(sampler).not.toContain('let ft = evaluateRoughDielectricTransmission(');
    expect(sampler).not.toContain('max(result.sampledEventPdf, 1e-10)');
    expect(sampler).not.toContain('ggxMultiscatterBoost(');
    expect(sampler).not.toContain('ggxMultiscatterBoostRoughness(');
    expect(sampler).not.toContain('result.sampleAllowsAreaMis = true;');
    expect(sampler).not.toContain(
      'result.sampleAllowsAreaMis = roughTransmissionProposalPdf > 0.0;',
    );
  });

  it('preserves the delta estimators that bypass finite finalization', () => {
    const sampler = sliceBetween(
      PT_WEBGPU_PATH_TRACE_BSDF_WGSL,
      'fn sampleNextBounceDirectionWithClearcoatNormal(',
      'fn sampleNextBounceDirection(',
    );

    expect(sampler).toContain(
      'result.sampledEventPdf = reflectionProbability / lobeWeightSum;',
    );
    expect(sampler).toContain(
      'microfacetInterface.reflectance * sheenAttenuation *',
    );
    expect(sampler).toContain(
      'result.sampledEventPdf = transmissionProbability / lobeWeightSum;',
    );
    expect(sampler).toContain('baseColor * transmissionWeight *');
    expect(count(sampler, 'result.sampledIsDelta = true;')).toBe(2);
  });

  it('keeps Kulla-Conty in the full finite evaluator reached by the finalizer', () => {
    const evaluator = sliceBetween(
      PT_WEBGPU_PATH_TRACE_BSDF_WGSL,
      'fn evaluateBrdfFullWithClearcoatNormal(',
      'fn evaluateFiniteSameSideBrdfFullWithClearcoatNormal(',
    );
    const finalizer = sliceBetween(
      PT_WEBGPU_PATH_TRACE_BSDF_WGSL,
      'fn finalizeFiniteBounceSampleWithClearcoatNormal(',
      '// D9.1',
    );

    expect(evaluator).toContain('ggxMultiscatterLobeRoughness(');
    expect(evaluator).toContain('ggxMultiscatterLobe(');
    expect(evaluator).toContain('let base = diff + spec + ms;');
    expect(finalizer).toContain('evaluateFiniteBsdfFullWithClearcoatNormal(');
    expect(finalizer).toContain(
      '(*result).throughputMul = finiteBsdf * cosine / marginalPdf;',
    );
    expect(finalizer).toContain('(*result).sampledEventPdf = marginalPdf;');
    expect(finalizer).toContain('(*result).sampleAllowsAreaMis = true;');
  });

  it('composes exactly one production Kulla-Conty implementation in both tiers', () => {
    for (const source of [
      composePtWebgpuTraceWgsl(false),
      PT_WEBGPU_TRACE_LITE_WGSL,
    ]) {
      expect(count(source, 'fn ggxMultiscatterLobe(')).toBe(1);
      expect(count(source, 'fn ggxDirectionalAlbedo(')).toBe(1);
    }
  });
});
