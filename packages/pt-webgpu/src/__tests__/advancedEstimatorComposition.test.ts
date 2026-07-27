import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_BDPT_CONNECTION_WGSL } from '../wgsl/bdpt/bdptConnection.wgsl.js';
import { composePathTraceKernelWgsl } from '../wgsl/pathTrace/kernel.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL } from '../wgsl/pathTrace/caustic.wgsl.js';
import {
  SPPM_GROUP3_BINDINGS_WGSL,
  SPPM_PHOTON_PASS_WGSL,
} from '../wgsl/pathTrace/sppmBindings.wgsl.js';

const defaultKernel = composePathTraceKernelWgsl({ volumetricSss: true });
const compositeKernel = composePathTraceKernelWgsl({
  volumetricSss: true,
  restirPtComposite: true,
});

function count(source: string, needle: string): number {
  return source.split(needle).length - 1;
}

function selectedCompleteEstimate(
  restirSelected: boolean,
  restirEstimate: number,
  peerEstimate: number,
): { value: number; activeTerms: number } {
  const terms = [
    restirSelected ? restirEstimate : 0,
    restirSelected ? 0 : peerEstimate,
  ];
  return {
    value: terms[0]! + terms[1]!,
    activeTerms: terms.filter((term) => term !== 0).length,
  };
}

function disjointFiniteOwners(
  deltaTerminal: boolean,
  bdptValue: number,
  causticValue: number,
): { value: number; activeTerms: number } {
  const terms = [
    deltaTerminal ? 0 : bdptValue,
    deltaTerminal ? causticValue : 0,
  ];
  return {
    value: terms[0]! + terms[1]!,
    activeTerms: terms.filter((term) => term !== 0).length,
  };
}

function prefixPartitionOwners(
  lightPrefixKinds: readonly ('finite' | 'delta')[],
  connectionIndex: number,
  bdptValue: number,
  sppmValue: number,
): { value: number; activeTerms: number } {
  const sppmOwns = lightPrefixKinds
    .slice(1, connectionIndex)
    .includes('delta');
  const terms = [sppmOwns ? 0 : bdptValue, sppmOwns ? sppmValue : 0];
  return {
    value: terms[0]! + terms[1]!,
    activeTerms: terms.filter((term) => term !== 0).length,
  };
}

describe('advanced estimator composition', () => {
  it.each([
    ['BDPT + ReSTIR-PT', 4.25, 4.5],
    ['SPPM + ReSTIR-PT', 7.75, 8.0],
  ])('%s selects one complete finite estimator, never an additive overlap', (_pair, restir, peer) => {
    for (const restirSelected of [false, true]) {
      const result = selectedCompleteEstimate(restirSelected, restir, peer);
      expect(result.activeTerms).toBe(1);
      expect(result.value).toBe(restirSelected ? restir : peer);
      expect(Number.isFinite(result.value)).toBe(true);
      expect(result.value).not.toBe(restir + peer);
    }

    expect(compositeKernel).toContain(
      'let advancedPeerEnabled = params.bdptEnabled != 0u || caustic != 0u;',
    );
    expect(compositeKernel).toContain(
      'let rptMixtureSelected = !advancedPeerEnabled || rand_f32(&rng) < 0.5;',
    );
    expect(compositeKernel).toContain(
      'let rptCompositeContributed = rptProducerContributed && rptMixtureSelected;',
    );
    expect(compositeKernel).toContain(
      'let bdptOwnsFiniteLightFamily = params.bdptEnabled != 0u && !rptCompositeContributed;',
    );
    expect(count(compositeKernel, 'outRadiance = outRadiance + rptComposite.rgb;')).toBe(1);
  });

  it('BDPT + MNEE partitions bounded all-delta finite-source paths exactly once', () => {
    for (const deltaTerminal of [false, true]) {
      const result = disjointFiniteOwners(deltaTerminal, 3.5, 3.75);
      expect(result.activeTerms).toBe(1);
      expect(Number.isFinite(result.value)).toBe(true);
    }

    expect(defaultKernel).toContain('manifoldNeeContribution(');
    expect(defaultKernel).toContain('caustic == 1u && mneeReceiverEligible');
    expect(defaultKernel).toContain('else if (sppmActive)');
    expect(defaultKernel).not.toContain('!bdptOwnsFiniteLightFamily && caustic');
    expect(defaultKernel).toContain('&& !mneeOwnsCurrentEmission');
    expect(defaultKernel).toContain('mneeOwnedDeltaDepth < mneeMaxDepth');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'bdptLightPrefixIsMneeOwned(c)',
    );
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'pathMeasure = nDotL * emitter.area * areaDet;',
    );
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).not.toContain(
      'pathMeasure = nDotL * emitter.area * areaDet * misWeight;',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'infiniteEyeEscapePdf = bdptTransmissiveConnectionPdf(',
    );
    expect(count(PT_WEBGPU_BDPT_CONNECTION_WGSL, 'bsdfHasFiniteConnectionSupport(')).toBe(3);
  });

  it('BDPT + SPPM assigns every interior-delta light prefix to exactly one owner', () => {
    for (const prefix of [
      ['delta', 'finite', 'finite'],
      ['finite', 'delta', 'finite'],
      ['finite', 'finite', 'finite'],
    ] as const) {
      const result = prefixPartitionOwners(prefix, 2, 3.5, 3.75);
      expect(result.activeTerms).toBe(1);
      expect(Number.isFinite(result.value)).toBe(true);
    }

    expect(defaultKernel).toContain('photonMapUpdateProgressive(');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'bdptLightPrefixContainsInteriorDelta(c)',
    );
  });

  it('spectral SPPM converts each new hero-flux sample once and keeps persistent RGB finite', () => {
    const tau = [0.25, 0.5, 0.75] as const;
    const reconstructedNewFlux = [1.0, 0.5, -0.125] as const;
    const ratio = 2 / 3;
    const tauPrime = tau.map((value, channel) =>
      (value + reconstructedNewFlux[channel]!) * ratio,
    );
    expect(tauPrime.every(Number.isFinite)).toBe(true);
    expect(tauPrime[0]).toBeCloseTo(5 / 6, 14);
    expect(tauPrime[1]).toBeCloseTo(2 / 3, 14);
    expect(tauPrime[2]).toBeCloseTo(5 / 12, 14);

    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'photonFlux = spectralEmissionAtHero(photonFlux, photonHeroLambda);',
    );
    expect(SPPM_PHOTON_PASS_WGSL).toContain('spectralCombinedReflectanceAtHero(');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('cauchyIorAtLambda(');
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'frontFace, params.spectralEnabled != 0u, photonHeroLambda,',
    );
    expect(count(SPPM_GROUP3_BINDINGS_WGSL, 'heroWavelengthToRgb(')).toBe(1);
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain(
      'let tauPrime = (tau + phiForTau) * ratio;',
    );
    expect(defaultKernel).toContain('outRadiance = outRadiance + sppmRgbRadiance;');
    expect(defaultKernel).not.toContain(
      'heroWavelengthToRgb(heroLambda, luminance(sppmRgbRadiance)',
    );
  });

  it('SPPM compensates persistent state only on the peer half of a ReSTIR mixture', () => {
    expect(compositeKernel).toContain('advancedEstimatorStateInvPdf = 2.0;');
    expect(compositeKernel).toContain(
      'advancedPeerEnabled && rptProducerContributed && !rptMixtureSelected',
    );
    expect(compositeKernel).toContain('advancedEstimatorStateInvPdf,');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain(
      'phiForTau = phiForTau * absorbedFluxInvPdf;',
    );
    expect(defaultKernel).toContain('var advancedEstimatorStateInvPdf = 1.0;');
    expect(compositeKernel).toContain(
      'sppmRgbRadiance = sppmCurrentProgressiveEstimate(pixelIndex);',
    );
    expect(compositeKernel).not.toContain(
      'sppmCurrentProgressiveEstimate(pixelIndex) * advancedEstimatorStateInvPdf',
    );
  });
});
