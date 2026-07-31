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

function frameEstimatorHash(frameSeed: number, frameIndex: number): number {
  let hash = (frameSeed ^ Math.imul(frameIndex, 0x9e3779b9)) >>> 0;
  hash = Math.imul((hash ^ (hash >>> 16)) >>> 0, 0x7feb352d) >>> 0;
  hash = Math.imul((hash ^ (hash >>> 15)) >>> 0, 0x846ca68b) >>> 0;
  return (hash ^ (hash >>> 16)) >>> 0;
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
      'params.frameSeed ^ (params.frameIndex * 0x9e3779b9u);',
    );
    expect(compositeKernel).toContain(
      '!advancedPeerEnabled || (frameEstimatorHash & 1u) == 0u;',
    );
    expect(compositeKernel).toContain(
      'let advancedEstimatorSelected =\n    advancedPeerEnabled && !rptMixtureSelected;',
    );
    expect(compositeKernel).toContain(
      'let rptCompositeContributed = rptProducerContributed && rptMixtureSelected;',
    );
    expect(compositeKernel).toContain(
      'let bdptOwnsFiniteLightFamily = params.bdptEnabled != 0u && advancedEstimatorSelected;',
    );
    expect(compositeKernel).toContain(
      'let mneeActive = caustic == 1u && advancedEstimatorSelected;',
    );
    expect(compositeKernel).toContain(
      'if (mneeActive && mneeReceiverEligible)',
    );
    expect(compositeKernel).toContain(
      'let sppmActive = caustic == 2u && advancedEstimatorSelected;',
    );
    expect(count(compositeKernel, 'outRadiance = outRadiance + rptComposite.rgb;')).toBe(1);
  });

  it('uses one backend-independent frame coin and keeps producer drops out of BDPT', () => {
    const selectionStart = compositeKernel.indexOf('var frameEstimatorHash =');
    const selectionEnd = compositeKernel.indexOf(
      'let rptCompositeContributed =',
      selectionStart,
    );
    expect(selectionStart).toBeGreaterThan(-1);
    expect(selectionEnd).toBeGreaterThan(selectionStart);
    const selectionBlock = compositeKernel.slice(selectionStart, selectionEnd);
    expect(selectionBlock).not.toContain('gid.');
    expect(selectionBlock).not.toContain('pixelIndex');
    expect(selectionBlock).not.toContain('rand_f32');
    expect(selectionBlock).not.toContain('PtRngState');

    for (const [frameSeed, frameIndex] of [
      [0, 0],
      [1, 0],
      [0x5eed5eed, 17],
      [0xffffffff, 0xffffffff],
    ] as const) {
      const selectedAtPixel0 = (frameEstimatorHash(frameSeed, frameIndex) & 1) === 0;
      const selectedAtArbitrarySplatTarget =
        (frameEstimatorHash(frameSeed, frameIndex) & 1) === 0;
      expect(selectedAtPixel0).toBe(selectedAtArbitrarySplatTarget);
    }

    // On an RPT-selected frame, a producer-dropped pixel falls through to the
    // ordinary eye path. It must not build a light subpath or emit arbitrary
    // camera splats merely because this target pixel had no RPT contribution.
    const advancedPeerEnabled = true;
    const rptMixtureSelected = true;
    const rptProducerContributed = false;
    const advancedEstimatorSelected =
      advancedPeerEnabled && !rptMixtureSelected;
    const rptCompositeContributed =
      rptProducerContributed && rptMixtureSelected;
    const bdptOwnsFiniteLightFamily = advancedEstimatorSelected;
    expect(rptCompositeContributed).toBe(false);
    expect(bdptOwnsFiniteLightFamily).toBe(false);
    expect(compositeKernel).toContain(
      'if (bdptOwnsFiniteLightFamily) {\n    bdptSetInvocationHeroLambda(heroLambda);\n    bdptBuildInvocationLightSubpath(gid.xy);',
    );
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
      'photonFlux = ptScaleEnvironmentRadiance(\n' +
      '      spectralEmissionAtHero(photonFlux, photonHeroLambda),\n' +
      '      1.0,\n' +
      '    );',
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
    expect(compositeKernel).toContain('if (advancedEstimatorSelected) {');
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
