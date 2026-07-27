import { describe, expect, it } from 'vitest';
import {
  boundedSmsMultiplicityExpectation,
  boundedSmsMultiplicityWeight,
  frozenSmsTopologyProposalDensity,
  provesPlanarUniqueSmsRoot,
  sameSmsRoot,
  smsMultiplicitySeed,
} from '../manifoldSmsMultiplicity.js';

describe('bounded SMS multiplicity recurrence', () => {
  it('pins K=1 and the downward, monotone convergence to 1/p', () => {
    for (const p of [0.01, 0.2, 0.5, 1]) {
      expect(boundedSmsMultiplicityExpectation(p, 1)).toBeCloseTo(1, 13);
      let previous = 0;
      for (const cap of [1, 2, 4, 8, 16, 32]) {
        const value = boundedSmsMultiplicityExpectation(p, cap);
        expect(value).toBeGreaterThanOrEqual(previous);
        expect(value).toBeLessThanOrEqual(1 / p);
        previous = value;
      }
    }
    expect(boundedSmsMultiplicityExpectation(0.5, 32)).toBeCloseTo(2, 8);
    expect(boundedSmsMultiplicityWeight(null, 8)).toBe(8);
    expect(boundedSmsMultiplicityWeight(3, 8)).toBe(3);
  });

  it('uses a deterministic, trial-separated seed stream', () => {
    const args = [123, 456, 2, 0] as const;
    expect(smsMultiplicitySeed(...args, 1)).toBe(smsMultiplicitySeed(...args, 1));
    expect(new Set(Array.from({ length: 8 }, (_, i) => smsMultiplicitySeed(...args, i + 1))).size).toBe(8);
  });

  it('keeps uniform-area root seeds out of the frozen-topology denominator', () => {
    const proposal = {
      chainLengthPmf: 1 / 3,
      endpointSelectionPmf: 1 / 4,
      endpointPdf: 2,
      facets: [
        { pairPmf: 0.25, seedArea: 1, offsetNormalPdf: 0.5, eventPmf: 0.5 },
        { pairPmf: 0.75, seedArea: 100, offsetNormalPdf: 0.25, eventPmf: 1 },
      ],
    };
    const density = frozenSmsTopologyProposalDensity(proposal);
    const uniformlyScaled = frozenSmsTopologyProposalDensity({
      ...proposal,
      facets: proposal.facets.map((facet) => ({
        ...facet,
        seedArea: facet.seedArea * 1_000_000,
      })),
    });
    expect(uniformlyScaled).toBe(density);

    // Discrete topology is frozen during recurrence and therefore remains in q.
    const changedDiscretePmf = frozenSmsTopologyProposalDensity({
      ...proposal,
      facets: [
        { ...proposal.facets[0]!, pairPmf: 0.5 },
        proposal.facets[1]!,
      ],
    });
    expect(changedDiscretePmf / density).toBeCloseTo(2, 13);
    expect(() => frozenSmsTopologyProposalDensity({
      ...proposal,
      facets: [{ ...proposal.facets[0]!, seedArea: 0 }],
    })).toThrow(/seed area/);
  });

  it('does not merge near-coincident roots outside the positional tolerance', () => {
    const root = [{ x: 1, y: 2, z: 3 }];
    expect(sameSmsRoot(root, [{ x: 1.000_09, y: 2, z: 3 }], 0.000_1)).toBe(true);
    expect(sameSmsRoot(root, [{ x: 1.000_11, y: 2, z: 3 }], 0.000_1)).toBe(false);
  });

  it('only proves uniqueness for the restricted planar delta transmission family', () => {
    const base = {
      chainLength: 1,
      event: 'transmission' as const,
      roughness: 0,
      constantShadingFrame: true,
      hasNormalMap: false,
      hasBumpMap: false,
      hasLayerNormalMap: false,
      etaIncident: 1,
      etaTransmitted: 1.5,
      incidentCosine: 0.8,
      endpointPlaneDistance: -1,
      receiverPlaneDistance: 1,
    };
    expect(provesPlanarUniqueSmsRoot(base)).toBe(true);
    expect(provesPlanarUniqueSmsRoot({ ...base, event: 'reflection' })).toBe(false);
    expect(provesPlanarUniqueSmsRoot({ ...base, hasNormalMap: true })).toBe(false);
    expect(provesPlanarUniqueSmsRoot({ ...base, chainLength: 2 })).toBe(false);
    expect(provesPlanarUniqueSmsRoot({
      ...base,
      etaIncident: 1.5,
      etaTransmitted: 1,
      incidentCosine: 0.5,
    })).toBe(false);
  });
});
