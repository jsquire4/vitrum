import { describe, expect, it } from 'vitest';
import { representBernoulliProbabilityF32 } from '@vitrum/shared-samplers';
import { NRC_INDEPENDENT_SUFFIX_WGSL } from '../nrcIndependentSuffix.wgsl.js';
import { NATIVE_GLASS_GI_WGSL } from '../risGiGlassWalk.wgsl.js';
import { RIS_GI_WGSL } from '../risGi.wgsl.js';
import { RIS_GI_NRC_BODY } from '../risGiNrc.wgsl.js';
import { SCENE_TRAVERSAL_WGSL } from '../sceneTraversal.wgsl.js';

describe('represented Bernoulli optical transport', () => {
  it('retains support for every tiny positive physical branch', () => {
    const represented = representBernoulliProbabilityF32(Number.MIN_VALUE);
    expect(represented).toBe(1 / 2 ** 24);
    expect(represented).toBeGreaterThan(0);
    expect(1 - represented).toBeGreaterThan(0);
  });

  it('uses the represented binary probability for both selection and correction', () => {
    const physicalTransmission = 0.37;
    const ideal = physicalTransmission / (1 + physicalTransmission);
    const transmittedPmf = representBernoulliProbabilityF32(ideal);
    const reflectedPmf = 1 - transmittedPmf;
    const transmittedContribution = 0.61 * physicalTransmission;
    const reflectedContribution = 0.29;
    const expectation =
      transmittedPmf * transmittedContribution / transmittedPmf +
      reflectedPmf * reflectedContribution / reflectedPmf;
    expect(expectation).toBeCloseTo(
      transmittedContribution + reflectedContribution,
      14,
    );

    for (const source of [RIS_GI_WGSL, NRC_INDEPENDENT_SUFFIX_WGSL]) {
      expect(source).toContain(
        'let transmissionBranchPdf = represented_bernoulli_probability_f32(',
      );
      expect(source).toContain(
        'rand_f32(rng) < transmissionBranchPdf',
      );
      expect(source).toContain(
        'transmissionPhysicalWeight / transmissionBranchPdf',
      );
      expect(source).toContain(
        'let reflectionBranchPdf = 1.0 - transmissionBranchPdf;',
      );
    }
  });

  it('publishes the three realized native closure PMFs from quantized thresholds', () => {
    const localThreshold = representBernoulliProbabilityF32(1 / 3);
    const reflectionThreshold = representBernoulliProbabilityF32(2 / 3);
    const localPmf = localThreshold;
    const reflectionPmf = reflectionThreshold - localThreshold;
    const transmissionPmf = 1 - reflectionThreshold;
    expect(localPmf).toBeGreaterThan(0);
    expect(reflectionPmf).toBeGreaterThan(0);
    expect(transmissionPmf).toBeGreaterThan(0);
    expect(localPmf + reflectionPmf + transmissionPmf).toBe(1);

    const local = 0.17;
    const reflection = 0.31;
    const transmission = 0.53;
    const expectation =
      localPmf * local / localPmf +
      reflectionPmf * reflection / reflectionPmf +
      transmissionPmf * transmission / transmissionPmf;
    expect(expectation).toBeCloseTo(local + reflection + transmission, 14);

    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'let localBranchPdf = localBranchThreshold;',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'reflectionBranchThreshold - localBranchThreshold;',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'let transmissionBranchPdf = 1.0 - reflectionBranchThreshold;',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'vec3f(selectedOpticalBranchPdf)',
    );
  });

  it('uses represented probabilities in PPG, defensive mixtures, slabs, and roulette', () => {
    for (const source of [RIS_GI_WGSL, RIS_GI_NRC_BODY]) {
      expect(source).toContain(
        'let alpha = represented_bernoulli_probability_f32(',
      );
      expect(source).toContain('if (bern < alpha)');
      expect(source).toContain('reservoirGiLogProposalMixture(alpha,');
    }
    expect(NRC_INDEPENDENT_SUFFIX_WGSL).toContain(
      'let specularMixProbability = represented_bernoulli_probability_f32(',
    );
    expect(NRC_INDEPENDENT_SUFFIX_WGSL).toContain(
      'specularMixProbability * pdfSpec +',
    );
    expect(NRC_INDEPENDENT_SUFFIX_WGSL).toContain(
      'let survive = represented_bernoulli_probability_f32(',
    );
    for (const source of [
      RIS_GI_WGSL,
      NATIVE_GLASS_GI_WGSL,
      NRC_INDEPENDENT_SUFFIX_WGSL,
    ]) {
      expect(source).toContain(
        'represented_bernoulli_probability_f32(0.5)',
      );
    }
  });
});

describe('fixed-origin accepted optical continuation', () => {
  it('suppresses only the exact source feature at an open zero-distance bound', () => {
    expect(SCENE_TRAVERSAL_WGSL).toContain(
      'fn traceSceneFirstHitWithOpticalSourceExclusion(',
    );
    expect(SCENE_TRAVERSAL_WGSL).toContain(
      'opticalSourceFeatureSuppressesTriangle(',
    );
    expect(SCENE_TRAVERSAL_WGSL).toContain(
      '!candidate.didHit || !(candidate.dist > exclusiveMinT)',
    );
    expect(SCENE_TRAVERSAL_WGSL).toContain(
      'ray, boundsMin, boundsMax, 0.0, result.hit.dist,',
    );
  });

  it('carries the exact feature through every native dielectric continuation', () => {
    for (const source of [
      RIS_GI_WGSL,
      NATIVE_GLASS_GI_WGSL,
      NRC_INDEPENDENT_SUFFIX_WGSL,
    ]) {
      expect(source).toContain(
        'traceSceneFirstHitAlphaMaskTexturedWithOpticalSource(',
      );
      expect(source).toContain('sceneOpticalSourceFeatureForExactHit(');
      expect(source).toContain('traceSceneRetraceOpticalHit(');
    }
    expect(RIS_GI_WGSL).toContain('let nextRay = Ray(nextOrigin, rayDirection);');
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'let walkRay = Ray(walkOrigin, refractDir);',
    );
    expect(NRC_INDEPENDENT_SUFFIX_WGSL).toContain(
      'nextRay = Ray(currentPos, nextDir);',
    );
  });
});
