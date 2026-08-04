import { describe, expect, it } from 'vitest';
import { representBernoulliProbabilityF32 } from '@vitrum/shared-samplers';

import { PT_WEBGPU_SOBOL_RNG_WGSL } from '../wgsl/common.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_BSDF_WGSL } from '../wgsl/pathTrace/bsdf.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_CORE_WGSL } from '../wgsl/pathTrace/kernelCore.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL } from '../wgsl/pathTrace/material.wgsl.js';
import { RESERVOIR_PT_HERO_WGSL } from '../wgsl/pathTrace/reservoirPtHero.wgsl.js';
import { RESTIR_PT_PRODUCER_WGSL } from '../wgsl/pathTrace/restirPtProducer.wgsl.js';

const f32 = Math.fround;

function representedSplit(aRaw: number, bRaw: number): readonly [number, number] {
  const a = f32(Math.max(aRaw, 0));
  const b = f32(Math.max(bRaw, 0));
  if (!(a > 0)) return [0, 1];
  if (!(b > 0)) return [1, 0];
  const total = f32(a + b);
  if (a <= b) {
    const pA = f32(representBernoulliProbabilityF32(f32(a / total)));
    return [pA, f32(1 - pA)];
  }
  const pB = f32(representBernoulliProbabilityF32(f32(b / total)));
  return [f32(1 - pB), pB];
}

function extensionMixture(
  clearcoatRaw: number,
  sheenRaw: number,
): readonly [number, number, number] {
  const clearcoat = f32(Math.max(clearcoatRaw, 0));
  const sheen = f32(Math.max(sheenRaw, 0));
  const extension = f32(clearcoat + sheen);
  if (!(extension > 0)) return [1, 0, 0];
  const [extensionProbability, baseProbability] = representedSplit(extension, 1);
  const [clearcoatConditional] = representedSplit(clearcoat, sheen);
  const clearcoatProbability = f32(extensionProbability * clearcoatConditional);
  return [
    baseProbability,
    clearcoatProbability,
    f32(extensionProbability - clearcoatProbability),
  ];
}

function dielectricMixture(
  diffuseRaw: number,
  reflectionRaw: number,
  transmissionRaw: number,
): readonly [number, number, number] {
  const diffuse = f32(Math.max(diffuseRaw, 0));
  const reflection = f32(Math.max(reflectionRaw, 0));
  const transmission = f32(Math.max(transmissionRaw, 0));
  const dielectric = f32(reflection + transmission);
  const [dielectricProbability, diffuseProbability] = representedSplit(
    dielectric,
    diffuse,
  );
  const [reflectionConditional] = representedSplit(reflection, transmission);
  const reflectionProbability = f32(dielectricProbability * reflectionConditional);
  return [
    reflectionProbability,
    diffuseProbability,
    f32(dielectricProbability - reflectionProbability),
  ];
}

describe('pt-webgpu represented transport probabilities', () => {
  it('preserves every positive extension and dielectric support below one ulp at unity', () => {
    const extension = extensionMixture(2 ** -30, 2 ** -31);
    expect(extension.every((probability) => probability > 0)).toBe(true);
    expect(f32(extension[0] + extension[1] + extension[2])).toBe(1);

    const dielectric = dielectricMixture(2 ** -30, 1, 2 ** -31);
    expect(dielectric.every((probability) => probability > 0)).toBe(true);
    expect(f32(dielectric[0] + dielectric[1] + dielectric[2])).toBe(1);
  });

  it('uses nested represented decisions for sampling and the matching joint PDFs', () => {
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      'fn bsdfRepresentedDielectricEventProbabilities(',
    );
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      'fn brdfRepresentedExtensionLobeProbabilities(',
    );
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      'let chooseDiffuse = rand_f32(rng) < eventProbabilities.y;',
    );
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      'chooseReflection = rand_f32(rng) < eventProbabilities.w;',
    );
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      'extensionProbabilities.x * transmissionProbability',
    );
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      'reverseEventProbabilities.z *',
    );
  });

  it('uses the represented probability for roulette, alpha, and ReSTIR source mixtures', () => {
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_CORE_WGSL).toContain(
      'let survival = represented_bernoulli_probability_f32(',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_CORE_WGSL).toContain(
      'result.throughputMul = 1.0 / survival;',
    );
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL).toContain(
      'let representedOpacity = represented_bernoulli_probability_f32(',
    );
    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      'let extensionProbabilities = brdfRepresentedExtensionLobeProbabilities(',
    );
    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      'let surv = represented_bernoulli_probability_f32(',
    );
  });

  it('publishes the same 24-bit Bernoulli helper in Sobol compositions', () => {
    expect(PT_WEBGPU_SOBOL_RNG_WGSL).toContain(
      'fn represented_bernoulli_probability_f32(probability: f32) -> f32',
    );
    expect(PT_WEBGPU_SOBOL_RNG_WGSL).toContain(
      'floor(probability * 16777216.0 + 0.5)',
    );
  });

  it('uses the actual integer-ticket occurrence probability for ReSTIR WRS', () => {
    const updateStart = RESERVOIR_PT_HERO_WGSL.indexOf('fn updateReservoirPTLog(');
    const updateEnd = RESERVOIR_PT_HERO_WGSL.indexOf(
      'fn copyReservoirPTVisibleDomain(',
      updateStart,
    );
    const updateBody = RESERVOIR_PT_HERO_WGSL.slice(updateStart, updateEnd);
    expect(RESERVOIR_PT_HERO_WGSL).toContain(
      'const RPT_REPRESENTED_BUCKET_COUNT: u32 = 16777216u;',
    );
    expect(updateBody).toContain(
      'u32(ceil(f32(RPT_REPRESENTED_BUCKET_COUNT) * replacementRatio))',
    );
    expect(updateBody).toContain('RPT_REPRESENTED_BUCKET_COUNT - 1u');
    expect(updateBody).toContain('let ticket = pcgNext(rng) >> 8u;');
    expect(updateBody).toContain('accepted = ticket < replacementBuckets;');
    expect(updateBody).toContain('let keepBuckets = RPT_REPRESENTED_BUCKET_COUNT - replacementBuckets;');
    expect(updateBody).toContain('(*r).logSelectionProbabilityLow = nextLogProbability.y;');
    expect(updateBody).toContain('return accepted;');
    expect(updateBody).not.toContain('rand_f32(rng) < replacementRatio');

    const finaliseStart = RESERVOIR_PT_HERO_WGSL.indexOf(
      'fn finaliseReservoirPTWGris(',
    );
    const finaliseEnd = RESERVOIR_PT_HERO_WGSL.indexOf(
      'fn refreshReconnectionStatePT(',
      finaliseStart,
    );
    const finaliseBody = RESERVOIR_PT_HERO_WGSL.slice(finaliseStart, finaliseEnd);
    expect(finaliseBody).toContain('(*r).selectedLogWeight,');
    expect(finaliseBody).toContain('-(*r).logSelectionProbability,');
    expect(finaliseBody).toContain('-(*r).logSelectionProbabilityLow,');
    expect(finaliseBody).toContain(
      'let logW = selectedCorrection.x + selectedCorrection.y - log(pHatF);',
    );
    expect(finaliseBody).not.toContain('(*r).logWeightSum - log(pHatF)');
  });

  it('keeps represented WRS occurrence state invocation-local', () => {
    const storeStart = RESERVOIR_PT_HERO_WGSL.indexOf('fn storeReservoirPTHero_rw(');
    const storeEnd = RESERVOIR_PT_HERO_WGSL.indexOf(
      '// Streaming RIS reservoir update',
      storeStart,
    );
    const storeBody = RESERVOIR_PT_HERO_WGSL.slice(storeStart, storeEnd);
    expect(storeBody).toContain('buf[b + 15u]');
    expect(storeBody).not.toContain('selectedLogWeight');
    expect(storeBody).not.toContain('logSelectionProbability');
  });
});
