import { describe, expect, it } from 'vitest';
import { CASCADE_MERGE_WGSL } from '../src/wgsl/cascadeMerge.wgsl.js';
import { PROBE_RAY_CAST_WGSL } from '../src/wgsl/probeRayCast.wgsl.js';

type Rgb = readonly [number, number, number];
type Scatter = readonly [number, number, number, number];
// Column-major, matching WGSL mat3x3f construction and multiplication.
type Mat3 = readonly [Rgb, Rgb, Rgb];

interface MediumState {
  readonly material: number;
  readonly instance: number;
  readonly color: Rgb;
  readonly attenuationDistance: number;
  readonly mapScale: number;
  readonly scattering: Rgb;
  readonly transmissionPaid: boolean;
}

interface BoundaryEvent {
  readonly kind: 'enter' | 'exit' | 'terminal';
  readonly material?: number;
  readonly instance?: number;
  readonly segment: number;
  readonly transmission?: number;
  readonly medium?: Omit<MediumState, 'transmissionPaid'>;
}

interface ContainmentCrossing {
  readonly material: number;
  readonly instance: number;
  readonly side: 1 | -1;
  readonly coverage: 'hole' | 'solid' | 'fractional';
}

function functionBody(source: string, name: string): string {
  const marker = `fn ${name}(`;
  const start = source.indexOf(marker);
  if (start < 0) throw new Error(`Missing ${name}`);
  const brace = source.indexOf('{', start);
  if (brace < 0) throw new Error(`Missing ${name} body`);
  let depth = 0;
  for (let i = brace; i < source.length; i += 1) {
    if (source[i] === '{') depth += 1;
    if (source[i] === '}') {
      depth -= 1;
      if (depth === 0) return source.slice(brace + 1, i);
    }
  }
  throw new Error(`Unterminated ${name}`);
}

function multiply(a: Rgb, b: Rgb | number): Rgb {
  return typeof b === 'number'
    ? [a[0] * b, a[1] * b, a[2] * b]
    : [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
}

function mediumTransfer(
  color: Rgb,
  attenuationDistance: number,
  mapScale: number,
  scattering: Rgb,
  segment: number,
): Rgb {
  const distance = Math.max(segment, 0);
  const g = Math.min(1, Math.max(0, mapScale));
  const channelTransfer = (channel: number, scatter: number): number => {
    const absorption = g === 0 || distance === 0
      ? 1
      : channel ** (distance * g / attenuationDistance);
    return absorption * Math.exp(-Math.max(scatter, 0) * distance);
  };
  return [
    channelTransfer(color[0], scattering[0]),
    channelTransfer(color[1], scattering[1]),
    channelTransfer(color[2], scattering[2]),
  ];
}

function resolvedOpticalDistance(
  segmentDistance: number,
  thicknessMapScale: number,
  authoredThickness: number | null,
): number {
  const segment = Math.max(segmentDistance, 0);
  if (authoredThickness == null) return segment;
  return Math.min(
    segment,
    Math.max(authoredThickness, 0) *
      Math.min(1, Math.max(thicknessMapScale, 0)),
  );
}

function shadowExtinction(
  color: Rgb,
  attenuationDistance: number,
  mapScale: number,
  scattering: Scatter,
  segment: number,
): Rgb {
  return mediumTransfer(
    color,
    attenuationDistance,
    mapScale,
    [scattering[0], scattering[1], scattering[2]],
    segment,
  );
}

function diagonalTransfer(value: Rgb): Mat3 {
  return [
    [value[0], 0, 0],
    [0, value[1], 0],
    [0, 0, value[2]],
  ];
}

function applyTransfer(transfer: Mat3, value: Rgb): Rgb {
  return [
    transfer[0][0] * value[0] + transfer[1][0] * value[1] + transfer[2][0] * value[2],
    transfer[0][1] * value[0] + transfer[1][1] * value[1] + transfer[2][1] * value[2],
    transfer[0][2] * value[0] + transfer[1][2] * value[1] + transfer[2][2] * value[2],
  ];
}

function composeTransfer(left: Mat3, right: Mat3): Mat3 {
  return [
    applyTransfer(left, right[0]),
    applyTransfer(left, right[1]),
    applyTransfer(left, right[2]),
  ];
}

function mediumRadianceTransfer(
  color: Rgb,
  attenuationDistance: number,
  mapScale: number,
  scattering: Scatter,
  albedo: Rgb,
  segment: number,
): Mat3 {
  const distance = Math.max(segment, 0);
  const absorption = mediumTransfer(
    color, attenuationDistance, mapScale, [0, 0, 0], distance,
  );
  if (!(distance > 0)) return diagonalTransfer(absorption);

  const sigmaS: Rgb = [
    Math.max(scattering[0], 0),
    Math.max(scattering[1], 0),
    Math.max(scattering[2], 0),
  ];
  const sigmaA: Rgb = [
    absorption[0] > 0 ? -Math.log(absorption[0]) / distance : Number.POSITIVE_INFINITY,
    absorption[1] > 0 ? -Math.log(absorption[1]) / distance : Number.POSITIVE_INFINITY,
    absorption[2] > 0 ? -Math.log(absorption[2]) / distance : Number.POSITIVE_INFINITY,
  ];
  const sigmaT: Rgb = [
    sigmaA[0] + sigmaS[0],
    sigmaA[1] + sigmaS[1],
    sigmaA[2] + sigmaS[2],
  ];
  const transmittance: Rgb = [
    absorption[0] * Math.exp(-sigmaS[0] * distance),
    absorption[1] * Math.exp(-sigmaS[1] * distance),
    absorption[2] * Math.exp(-sigmaS[2] * distance),
  ];
  const scatterAlbedo: Rgb = [
    sigmaT[0] > 0 ? sigmaS[0] / sigmaT[0] : 0,
    sigmaT[1] > 0 ? sigmaS[1] / sigmaT[1] : 0,
    sigmaT[2] > 0 ? sigmaS[2] / sigmaT[2] : 0,
  ];
  const g = Math.min(0.99, Math.max(-0.99, scattering[3]));
  const denominator = 1 + g * g;
  const phase = (1 - g * g) /
    (4 * Math.PI * denominator * Math.sqrt(denominator));
  const sourceScale: Rgb = [
    Math.max(albedo[0], 0) * scatterAlbedo[0] * (1 - transmittance[0]) * phase,
    Math.max(albedo[1], 0) * scatterAlbedo[1] * (1 - transmittance[1]) * phase,
    Math.max(albedo[2], 0) * scatterAlbedo[2] * (1 - transmittance[2]) * phase,
  ];
  return [
    [transmittance[0] + sourceScale[0] * 0.2126, sourceScale[1] * 0.2126, sourceScale[2] * 0.2126],
    [sourceScale[0] * 0.7152, transmittance[1] + sourceScale[1] * 0.7152, sourceScale[2] * 0.7152],
    [sourceScale[0] * 0.0722, sourceScale[1] * 0.0722, transmittance[2] + sourceScale[2] * 0.0722],
  ];
}

function retainsEmitterEmission(options: {
  readonly inspection: number;
  readonly event: 'reflection' | 'transmission';
  readonly roughness: number;
  readonly emitterCount: number;
  readonly skipEmitter?: boolean;
  readonly unlit?: boolean;
}): boolean {
  return options.unlit === true ||
    options.inspection === 0 ||
    options.event === 'transmission' ||
    options.roughness <= 0 ||
    options.emitterCount === 0 ||
    options.skipEmitter === true;
}

function runBulkStateMachine(
  seed: readonly MediumState[],
  events: readonly BoundaryEvent[],
): { transfer: Rgb; scalarPayments: number[]; terminalReached: boolean } {
  const stack = seed.map(medium => ({ ...medium }));
  let transfer: Rgb = [1, 1, 1];
  const scalarPayments: number[] = [];

  for (const event of events) {
    const top = stack.at(-1);
    if (top) {
      transfer = multiply(transfer, mediumTransfer(
        top.color,
        top.attenuationDistance,
        top.mapScale,
        top.scattering,
        event.segment,
      ));
    }
    if (event.kind === 'terminal') {
      return { transfer, scalarPayments, terminalReached: stack.length === 0 };
    }
    if (event.kind === 'enter') {
      if (!event.medium || event.transmission == null) throw new Error('bad entry');
      scalarPayments.push(event.transmission);
      stack.push({ ...event.medium, transmissionPaid: true });
      continue;
    }
    if (!top || top.material !== event.material || top.instance !== event.instance) {
      throw new Error('out-of-order bulk exit');
    }
    if (!top.transmissionPaid) {
      if (event.transmission == null) throw new Error('missing seeded-exit transmission');
      scalarPayments.push(event.transmission);
    }
    stack.pop();
  }
  return { transfer, scalarPayments, terminalReached: false };
}

function replayContainment(
  crossings: readonly ContainmentCrossing[],
  options: {
    readonly capacity?: number;
    readonly reachedTarget?: boolean;
  } = {},
): readonly string[] {
  const capacity = options.capacity ?? 8;
  const stack: string[] = [];
  for (const crossing of crossings) {
    if (crossing.coverage === 'fractional') {
      throw new Error('fractional bulk boundary');
    }
    if (crossing.coverage === 'hole') continue;
    const key = `${crossing.material}:${crossing.instance}`;
    if (crossing.side === 1) {
      if (stack.length >= capacity) throw new Error('live-medium capacity');
      stack.push(key);
    } else if (stack.pop() !== key) {
      throw new Error('non-LIFO bulk boundary');
    }
  }
  if (options.reachedTarget === false && stack.length !== 0) {
    throw new Error('miss with live medium');
  }
  return stack;
}

function dielectricFresnel(cosTheta: number, etaI: number, etaT: number): number {
  const c = Math.min(1, Math.max(0, Math.abs(cosTheta)));
  const sin2T = (etaI / etaT) ** 2 * (1 - c * c);
  if (sin2T >= 1) return 1;
  const cosT = Math.sqrt(Math.max(0, 1 - sin2T));
  const rs = (etaI * c - etaT * cosT) / (etaI * c + etaT * cosT);
  const rp = (etaT * c - etaI * cosT) / (etaT * c + etaI * cosT);
  return 0.5 * (rs * rs + rp * rp);
}

function mixtureExpectation(
  transmission: number,
  fresnel: number,
  localSurfaceSource: number,
  reflectedLo: number,
  transmittedLo: number,
  paidExit = false,
): number {
  const tWeight = paidExit ? 1 : transmission;
  const pTransmission = tWeight / (1 + tWeight);
  const pReflection = 1 - pTransmission;
  const reflectionSample = fresnel * reflectedLo / pReflection;
  const transmissionSample = pTransmission > 0
    ? tWeight * (1 - fresnel) * transmittedLo / pTransmission
    : 0;
  return localSurfaceSource +
    pReflection * reflectionSample + pTransmission * transmissionSample;
}

function layeredDirectResponse(
  baseResponse: number,
  reflectionResponse: number,
  faceLayerTransmission: number,
  filmTransmission: number,
): number {
  return baseResponse * faceLayerTransmission * filmTransmission +
    reflectionResponse * faceLayerTransmission;
}

function acceptsTerminal(
  interfaceEvents: number,
  budget: number,
  terminalInspections: number,
): boolean {
  return interfaceEvents <= budget && terminalInspections === 1;
}

describe('RC dielectric transport CPU oracle', () => {
  it('uses authored topology with entry-owned G and scattering over actual distance', () => {
    const result = runBulkStateMachine([], [
      {
        kind: 'enter', segment: 0, transmission: 0.8,
        medium: {
          material: 10, instance: 3, color: [0.81, 0.64, 0.49],
          attenuationDistance: 2, mapScale: 0.25,
          scattering: [0.1, 0.2, 0.3],
        },
      },
      {
        kind: 'enter', segment: 4, transmission: 0.6,
        medium: {
          material: 11, instance: 4, color: [0.5, 0.6, 0.7],
          attenuationDistance: 1, mapScale: 0.5,
          scattering: [0.03, 0.02, 0.01],
        },
      },
      { kind: 'exit', material: 11, instance: 4, segment: 2, transmission: 0.1 },
      { kind: 'exit', material: 10, instance: 3, segment: 3, transmission: 0.1 },
      { kind: 'terminal', segment: 0 },
    ]);

    const outerFirst = mediumTransfer([0.81, 0.64, 0.49], 2, 0.25, [0.1, 0.2, 0.3], 4);
    const inner = mediumTransfer([0.5, 0.6, 0.7], 1, 0.5, [0.03, 0.02, 0.01], 2);
    const outerLast = mediumTransfer([0.81, 0.64, 0.49], 2, 0.25, [0.1, 0.2, 0.3], 3);
    expect(result.transfer).toEqual(multiply(multiply(outerFirst, inner), outerLast));
    expect(result.scalarPayments).toEqual([0.8, 0.6]);
    expect(result.terminalReached).toBe(true);
  });

  it('attenuates a starts-inside first segment and pays scalar t once at its exit', () => {
    const seeded: MediumState = {
      material: 7,
      instance: 12,
      color: [0.64, 0.81, 1],
      attenuationDistance: 2,
      mapScale: 0.4,
      scattering: [0.05, 0, 0.1],
      transmissionPaid: false,
    };
    const result = runBulkStateMachine([seeded], [
      { kind: 'exit', material: 7, instance: 12, segment: 5, transmission: 0.7 },
      { kind: 'terminal', segment: 3 },
    ]);
    expect(result.transfer).toEqual(mediumTransfer(
      seeded.color,
      seeded.attenuationDistance,
      seeded.mapScale,
      seeded.scattering,
      5,
    ));
    expect(result.scalarPayments).toEqual([0.7]);
    expect(result.terminalReached).toBe(true);
  });

  it('treats authored G=0 as absorption identity without suppressing actual-distance scattering', () => {
    const transfer = mediumTransfer([0.01, 0.2, 0.9], 0.5, 0, [0.1, 0.2, 0.3], 4);
    expect(transfer[0]).toBeCloseTo(Math.exp(-0.4), 14);
    expect(transfer[1]).toBeCloseTo(Math.exp(-0.8), 14);
    expect(transfer[2]).toBeCloseTo(Math.exp(-1.2), 14);
    expect(mediumTransfer([0.2, 0.3, 0.4], 1, 0, [0, 0, 0], 100)).toEqual([1, 1, 1]);
  });

  it('caps authored absorption distance but gives synthetic bulk the full segment', () => {
    expect(resolvedOpticalDistance(9, 0, null)).toBe(9);
    expect(resolvedOpticalDistance(9, 0.25, null)).toBe(9);
    expect(resolvedOpticalDistance(9, 0.25, 2)).toBe(0.5);
    expect(resolvedOpticalDistance(9, 0.25, 20)).toBe(5);
    expect(resolvedOpticalDistance(9, 1, 20)).toBe(9);
  });

  it('reduces the radiance matrix exactly to diagonal absorption when scattering is zero', () => {
    const absorption = mediumTransfer(
      [0.81, 0.64, 0.49], 2, 0.25, [0, 0, 0], 4,
    );
    expect(mediumRadianceTransfer(
      [0.81, 0.64, 0.49], 2, 0.25, [0, 0, 0, 0.8], [0.2, 0.7, 1], 4,
    )).toEqual(diagonalTransfer(absorption));
  });

  it('preserves canonical cross-channel in-scatter, albedo, anisotropy, and actual distance', () => {
    const greenInput: Rgb = [0, 1, 0];
    const isotropic = mediumRadianceTransfer(
      [1, 1, 1], 1, 1, [1, 0, 0, 0], [1, 0, 0], 1,
    );
    const redFromGreen = applyTransfer(isotropic, greenInput);
    expect(redFromGreen[0]).toBeGreaterThan(0);
    expect(redFromGreen[1]).toBeCloseTo(1, 14);
    expect(redFromGreen[2]).toBeCloseTo(0, 14);

    const anisotropic = mediumRadianceTransfer(
      [1, 1, 1], 1, 1, [1, 0, 0, 0.8], [1, 0, 0], 1,
    );
    expect(applyTransfer(anisotropic, greenInput)[0]).not.toBeCloseTo(
      redFromGreen[0], 14,
    );
    const isotropicExtinction = shadowExtinction(
      [1, 1, 1], 1, 1, [1, 0, 0, 0], 1,
    );
    const anisotropicExtinction = shadowExtinction(
      [1, 1, 1], 1, 1, [1, 0, 0, 0.8], 1,
    );
    expect(anisotropicExtinction).toEqual(isotropicExtinction);
    expect(applyTransfer(mediumRadianceTransfer(
      [1, 1, 1], 1, 1, [1, 0, 0, 0], [0, 0, 0], 1,
    ), greenInput)[0]).toBe(0);

    const twiceAsFar = mediumRadianceTransfer(
      [1, 1, 1], 1, 1, [1, 0, 0, 0], [1, 0, 0], 2,
    );
    expect(applyTransfer(twiceAsFar, greenInput)[0]).toBeGreaterThan(
      redFromGreen[0],
    );
  });

  it('keeps a zero attenuation-colour lane exactly absorbing with nonzero scattering', () => {
    const transfer = mediumRadianceTransfer(
      [0, 0.8, 0.9], 1, 1, [4, 4, 4, 0], [1, 1, 1], 2,
    );
    const result = applyTransfer(transfer, [9, 7, 5]);
    expect(result[0]).toBe(0);
    expect(result.every(Number.isFinite)).toBe(true);
  });

  it('composes RGB interface weights after the cross-channel medium operator', () => {
    const medium = mediumRadianceTransfer(
      [0.8, 0.7, 0.6], 2, 0.5, [0.2, 0.1, 0.3, -0.2], [0.9, 0.4, 0.7], 3,
    );
    const interfaceWeights: Rgb = [0.2, 0.5, 0.9];
    const terminal: Rgb = [3, 5, 7];
    const composed = composeTransfer(diagonalTransfer(interfaceWeights), medium);
    const actual = applyTransfer(composed, terminal);
    const expected = multiply(applyTransfer(medium, terminal), interfaceWeights);
    expect(actual[0]).toBeCloseTo(expected[0], 14);
    expect(actual[1]).toBeCloseTo(expected[1], 14);
    expect(actual[2]).toBeCloseTo(expected[2], 14);
    const spectralAbsorption = applyTransfer(mediumRadianceTransfer(
      [0.3, 0.8, 0.95], 1, 1, [0, 0, 0, 0], [1, 1, 1], 2,
    ), [1, 1, 1]);
    expect(spectralAbsorption[0]).toBeCloseTo(0.3 ** 2, 14);
    expect(spectralAbsorption[1]).toBeCloseTo(0.8 ** 2, 14);
    expect(spectralAbsorption[2]).toBeCloseTo(0.95 ** 2, 14);
  });

  it('replays exact outside-to-origin LIFO topology without a disjoint-shell cap', () => {
    const disjoint = Array.from({ length: 24 }, (_, instance) => [
      { material: instance, instance, side: 1 as const, coverage: 'solid' as const },
      { material: instance, instance, side: -1 as const, coverage: 'solid' as const },
    ]).flat();
    expect(replayContainment([
      ...disjoint,
      { material: 99, instance: 0, side: 1, coverage: 'hole' },
      { material: 7, instance: 4, side: 1, coverage: 'solid' },
      // Repeated material+instance identities remain valid when they are
      // strictly nested: duplicate pushes/pop order carry the topology.
      { material: 7, instance: 4, side: 1, coverage: 'solid' },
    ])).toEqual(['7:4', '7:4']);
    expect(replayContainment([
      { material: 3, instance: 1, side: 1, coverage: 'solid' },
      { material: 3, instance: 1, side: -1, coverage: 'solid' },
    ])).toEqual([]);
    expect(() => replayContainment([
      { material: 1, instance: 1, side: 1, coverage: 'solid' },
      { material: 2, instance: 2, side: 1, coverage: 'solid' },
      { material: 1, instance: 1, side: -1, coverage: 'solid' },
    ])).toThrow('non-LIFO bulk boundary');
    expect(() => replayContainment([
      { material: 1, instance: 1, side: 1, coverage: 'solid' },
    ], { reachedTarget: false })).toThrow('miss with live medium');
    expect(() => replayContainment([
      { material: 1, instance: 1, side: 1, coverage: 'fractional' },
    ])).toThrow('fractional bulk boundary');
  });

  it('fails an out-of-order or wrong-instance nested exit closed', () => {
    const outer: MediumState = {
      material: 1, instance: 3, color: [1, 1, 1],
      attenuationDistance: 1, mapScale: 1, scattering: [0, 0, 0],
      transmissionPaid: true,
    };
    const inner = { ...outer, material: 2, instance: 4 };
    expect(() => runBulkStateMachine([outer, inner], [
      { kind: 'exit', material: 1, instance: 3, segment: 1 },
    ])).toThrow('out-of-order bulk exit');
    expect(() => runBulkStateMachine([outer], [
      { kind: 'exit', material: 1, instance: 4, segment: 1 },
    ])).toThrow('out-of-order bulk exit');
  });

  it('makes the unit-envelope estimator unbiased and retains exact TIR at t=1', () => {
    for (const transmission of [0, 0.2, 0.7, 1]) {
      const fresnel = 0.23;
      const localSurface = (1 - transmission) * 2 + 4;
      const expected = localSurface + fresnel * 5 +
        transmission * (1 - fresnel) * 11;
      expect(mixtureExpectation(
        transmission, fresnel, localSurface, 5, 11,
      )).toBeCloseTo(expected, 14);
    }
    expect(mixtureExpectation(1, 1, 4, 9, 100)).toBeCloseTo(13, 14);
    expect(dielectricFresnel(0.5, 1.52, 1)).toBe(1);
    expect(mixtureExpectation(0.4, 0.2, 4, 5, 7, true)).toBeCloseTo(
      4 + 0.2 * 5 + 0.8 * 7,
      14,
    );
  });

  it('keeps absolute film reflection live at the T=0 and t=1 endpoints', () => {
    expect(layeredDirectResponse(7, 3, 0.8, 0)).toBeCloseTo(2.4, 14);
    expect(layeredDirectResponse(7, 0, 0.8, 0)).toBe(0);
    const fullyTransmissiveLocal = (1 - 1) * 7 + 3;
    expect(fullyTransmissiveLocal).toBe(3);
  });

  it('reserves one terminal inspection after the eighth interface event', () => {
    expect(acceptsTerminal(8, 8, 1)).toBe(true);
    expect(acceptsTerminal(9, 8, 1)).toBe(false);
    expect(acceptsTerminal(8, 8, 0)).toBe(false);
  });

  it('retains Le only for paths without a represented rough-reflection NEE owner', () => {
    const common = { inspection: 1, roughness: 0.4, emitterCount: 3 } as const;
    expect(retainsEmitterEmission({ ...common, event: 'reflection' })).toBe(false);
    expect(retainsEmitterEmission({ ...common, event: 'transmission' })).toBe(true);
    expect(retainsEmitterEmission({ ...common, event: 'reflection', roughness: 0 })).toBe(true);
    expect(retainsEmitterEmission({ ...common, event: 'reflection', emitterCount: 0 })).toBe(true);
    expect(retainsEmitterEmission({ ...common, event: 'reflection', skipEmitter: true })).toBe(true);
    expect(retainsEmitterEmission({ ...common, event: 'reflection', unlit: true })).toBe(true);
    expect(retainsEmitterEmission({ ...common, event: 'reflection', inspection: 0 })).toBe(true);
    // The event that finally escapes a reciprocal thin sheet is transmission,
    // even if one or more internal reflection events preceded it.
    expect(retainsEmitterEmission({ ...common, event: 'transmission' })).toBe(true);
  });
});

describe('RC dielectric transport WGSL composition', () => {
  it('uses the realised 2^24 RNG probability in branch tests and estimators', () => {
    const lattice = 2 ** 24;
    const represented = (probability: number): number => {
      const p = Math.fround(probability);
      if (!(p > 0)) return 0;
      if (p >= 1) return 1;
      return Math.min(
        lattice - 1,
        Math.max(1, Math.floor(Math.fround(Math.fround(p * lattice) + 0.5))),
      ) / lattice;
    };
    expect(represented(0)).toBe(0);
    expect(represented(1)).toBe(1);
    expect(represented(2 ** -149)).toBe(1 / lattice);

    const alpha = functionBody(
      PROBE_RAY_CAST_WGSL,
      'rcMaterialAlphaDiscardedForProbeHit',
    );
    const suffix = functionBody(
      PROBE_RAY_CAST_WGSL,
      'rcTraceDielectricSuffixChannel',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'fn represented_bernoulli_probability_f32(probability: f32) -> f32',
    );
    expect(alpha).toContain(
      'let representedCoverage = represented_bernoulli_probability_f32(alpha.coverage);',
    );
    expect(alpha).toContain(
      'rcAlphaBlendCoverageHash(hit, ray, layer) >= representedCoverage',
    );
    expect(suffix).toContain(
      'let transmissionBranchPdf = represented_bernoulli_probability_f32(',
    );
    expect(suffix).toContain('transmissionBranchPdf > 0.0 &&');
    expect(suffix).toContain('/ transmissionBranchPdf,');
    expect(suffix).toContain('/ reflectionBranchPdf,');
  });

  it('composes one exact anisotropic reflection/transmission lobe implementation', () => {
    const lobe = functionBody(PROBE_RAY_CAST_WGSL, 'rcSampleDielectricLobe');
    expect(PROBE_RAY_CAST_WGSL.match(/fn rcSampleDielectricLobe\(/g)).toHaveLength(1);
    expect(lobe).toContain('rcSampleVndfTangentAnisotropic(');
    expect(lobe).toContain('rcDistributionGGXAnisotropic(');
    expect(lobe).toContain('let reflectedDirection = safe_normalize(reflect(-wo, wm));');
    expect(lobe).toContain('let refractedRaw = refract(-wo, wm, etaIncident / etaTarget);');
    expect(lobe).toContain('let tirRgb = sin2TargetRgb >= vec3f(1.0);');
    expect(lobe).toContain('reflectanceRgb = select(reflectanceRgb, vec3f(1.0), tirRgb);');
    expect(lobe).toContain('transmittanceRgb = select(transmittanceRgb, vec3f(0.0), tirRgb);');
    expect(lobe).toContain('out.weightRgb = reflectanceRgb * directionalBaseWeight;');
    expect(PROBE_RAY_CAST_WGSL).not.toContain('fn rcSampleGgxDielectricTransmission(');
  });

  it('keeps packed topology separate from an entry-owned finite distance cap', () => {
    const material = functionBody(PROBE_RAY_CAST_WGSL, 'rcSampleProbeHitMaterial');
    const absorption = functionBody(PROBE_RAY_CAST_WGSL, 'rcMediumSegmentAbsorption');
    const active = functionBody(PROBE_RAY_CAST_WGSL, 'rcMediumActiveDistance');
    const segment = functionBody(PROBE_RAY_CAST_WGSL, 'rcMediumRadianceSegmentTransfer');
    expect(material).toContain('out.bulkThickness = materialOpticalThickness(hit.indices.w);');
    expect(material).toContain('out.thicknessMapScale = materialOpticalThicknessMapScale(');
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'fn materialOpticalHasAuthoredThickness(triIndex: u32) -> bool',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'materialOpticalHasAuthoredThickness(triIndex)',
    );
    expect(material).not.toContain('out.bulkThickness = out.bulkThickness *');
    expect(absorption).toContain('let distance = max(activeDistance, 0.0);');
    expect(absorption).not.toContain('materialOpticalHasAuthoredThickness(');
    expect(absorption).not.toContain('thicknessMapScale');
    expect(absorption).toContain('triIndex, distance, rgbBeer,');
    expect(absorption).toContain('var absorption = vec3f(1.0);');
    expect(active).toContain('min(segment, max(remainingDistance, 0.0))');
    expect(segment).toContain('let sigmaS = max(scattering.rgb, vec3f(0.0));');
    expect(segment).toContain('let positiveInfinity = bitcast<f32>(0x7f800000u);');
    expect(segment).toContain('absorption.x > 0.0');
    expect(segment).toContain('rcHenyeyGreensteinPhase(0.0, clamp(scattering.a, -0.99, 0.99))');
    expect(segment).toContain('return mat3x3f(');
  });

  it('owns local substrate, emission, exact optics, nesting, and terminal receiver in one suffix', () => {
    const suffix = functionBody(PROBE_RAY_CAST_WGSL, 'rcTraceDielectricSuffixChannel');
    const kernel = functionBody(PROBE_RAY_CAST_WGSL, 'probeRayCastKernel');
    expect(suffix).toContain('var accumulatedRadiance = 0.0;');
    expect(suffix).toContain('rcSuffixTransferredChannel(throughput, emissionSource, channel);');
    expect(suffix).toContain('if (interfaceUnlit != 0u) {');
    expect(suffix).toContain('rcSuffixTransferredChannel(throughput, localSurfaceSource, channel);');
    expect(suffix.indexOf('if (interfaceUnlit != 0u)')).toBeLessThan(
      suffix.indexOf('rcSuffixTransferredChannel(throughput, localSurfaceSource, channel);'),
    );
    expect(suffix).not.toContain('opaquePhysicalWeight');
    expect(suffix).toContain('let interfaceLobe = rcSampleDielectricLobe(');
    expect(suffix).toContain('mediumBoundary[top] == currentBoundary &&');
    expect(suffix).toContain('mediumRepresented[top] == currentRepresented;');
    expect(suffix).toContain('inspection <= RC_GLASS_STATIC_MAX_INTERFACES');
    expect(suffix).toContain('if (mediumDepth != 0u) { return accumulatedRadiance; }');
    expect(suffix).toContain('rcSuffixTransferredChannel(throughput, env, channel);');
    expect(suffix).toContain('let exitLobe = rcSampleDielectricLobe(');
    expect(suffix).toContain('exitLobe.kind != RC_DIELECTRIC_EVENT_TRANSMISSION');
    expect(suffix).not.toContain('var slabExited');
    expect(suffix).not.toContain('loop {');
    expect(suffix).not.toContain('continuationStep');
    expect(suffix).toContain('ray.origin = currentHitPoint;');
    expect(suffix).toContain('currentSourceFeature, false, true');
    expect(kernel.match(/rcTraceDielectricSuffixChannel\(/g)).toHaveLength(3);
    expect(kernel).not.toContain('localSurfaceRadiance + transContrib');
  });

  it('separates absolute film reflection from base-layer transmittance', () => {
    const response = functionBody(PROBE_RAY_CAST_WGSL, 'rcEvaluateProbeDirectResponse');
    const sources = functionBody(
      PROBE_RAY_CAST_WGSL, 'rcShadeTransmissionInterfaceSources',
    );
    expect(response).toContain('let baseResponse = diffuse * nDotL;');
    expect(response).toContain('let reflectionResponse = spec * nDotL');
    expect(response).toContain('baseResponse * mat.layerTransmission +');
    expect(response).toContain(
      'reflectionResponse * mat.reflectionLayerTransmission;',
    );
    expect(sources).toContain('directMat.transmission = 1.0;');
    expect(sources).not.toContain(
      'directSun + emitterNEE + pointSpotLights + bakedOutgoing\n  ) * probeMat.layerTransmission',
    );
  });

  it('uses correlated RGB event variates while dispersion alone changes geometry', () => {
    const suffix = functionBody(PROBE_RAY_CAST_WGSL, 'rcTraceDielectricSuffixChannel');
    const random = functionBody(PROBE_RAY_CAST_WGSL, 'rcSuffixRandom');
    expect(random).not.toContain('channel');
    const lobe = functionBody(PROBE_RAY_CAST_WGSL, 'rcSampleDielectricLobe');
    expect(lobe).toContain('let etaIncident = rcRgbChannel(etaIncidentRgb, channel);');
    expect(lobe).toContain('let etaTarget = rcRgbChannel(etaTargetRgb, channel);');
    expect(suffix).toContain('var targetIor = probeMat.opticalIor;');
    expect(suffix).not.toMatch(/rcSuffixRandom\([^\n]*channel/);
  });

  it('uses matrix-valued radiance transfer while shadow rays remain extinction-only', () => {
    const radianceTransfer = functionBody(
      PROBE_RAY_CAST_WGSL, 'rcMediumRadianceSegmentTransfer',
    );
    const shadowExtinction = functionBody(
      PROBE_RAY_CAST_WGSL, 'rcMediumShadowExtinction',
    );
    const suffix = functionBody(PROBE_RAY_CAST_WGSL, 'rcTraceDielectricSuffixChannel');
    const shadow = functionBody(PROBE_RAY_CAST_WGSL, 'rcTraceShadowTransmittance');
    expect(radianceTransfer).toContain('return mat3x3f(');
    expect(radianceTransfer).toContain('sourceScale * 0.7152');
    expect(suffix).toContain('var throughput = rcSuffixDiagonalTransfer(vec3f(1.0));');
    expect(suffix).toContain('var mediumScattering: array<vec4f, 8>;');
    expect(suffix).toContain('var mediumAlbedo: array<vec3f, 8>;');
    expect(suffix).toContain('throughput = throughput * segmentTransfer;');
    expect(suffix).toContain('interfaceLobe.weightRgb');
    expect(suffix).not.toContain('rcRgbChannel(segmentTransfer');
    expect(shadow).toContain('rcMediumShadowExtinction(');
    expect(shadow).not.toContain('rcMediumRadianceSegmentTransfer(');
    expect(shadowExtinction).not.toContain('albedo');
    expect(shadowExtinction).not.toContain('sourceScale');
    expect(shadowExtinction).not.toContain('mat3x3f');
  });

  it('makes explicit bulk transport own volume once and tracks emitter-hit ownership', () => {
    const sources = functionBody(
      PROBE_RAY_CAST_WGSL, 'rcShadeTransmissionInterfaceSources',
    );
    const suffix = functionBody(PROBE_RAY_CAST_WGSL, 'rcTraceDielectricSuffixChannel');
    expect(sources).not.toContain('max(probeMat.bulkThickness, 0.0) > 0.0;');
    expect(sources).toContain('if (!suppressOpaqueSubstrate && !explicitBulkSegment)');
    expect(PROBE_RAY_CAST_WGSL).toContain('explicitBulkSegment: bool,');
    expect(suffix).toContain('pairedPaidExit, hasBulkTopology,');
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'false, initialEvent.encodedBoundaryId != 0u,',
    );
    expect(suffix).toContain('var arrivedWithoutNeeOwner = true;');
    expect(suffix).toContain('let includeEmitterEmission =');
    expect(suffix).toContain('(mat.flags & MATERIAL_FLAG_SKIP_EMITTER) != 0u;');
    expect(suffix).toContain('interfaceLobe.kind == RC_DIELECTRIC_EVENT_TRANSMISSION ||');
    expect(suffix).toContain('probeMat.roughness <= 0.0 ||');
    expect(suffix).toContain('u.emitterCount == 0u;');
    expect(suffix).toContain('nextArrivesWithoutNeeOwner = true;');
    expect(suffix).toContain('exitLobe.kind != RC_DIELECTRIC_EVENT_TRANSMISSION');
  });

  it('classifies starts-inside media from a fixed origin with exact identity LIFOs', () => {
    const suffix = functionBody(PROBE_RAY_CAST_WGSL, 'rcTraceDielectricSuffixChannel');
    const shadow = functionBody(PROBE_RAY_CAST_WGSL, 'rcTraceShadowTransmittance');
    for (const body of [suffix, shadow]) {
      expect(body).toContain('let containmentRay = Ray(');
      expect(body).toContain('surface < surfaceBudget;');
      expect(body).toContain('containmentRay, containmentMinT, noSourceFeature, true, false');
      expect(body).toContain('temporaryBoundary[top] != containmentEvent.encodedBoundaryId');
      expect(body).toContain('temporaryRepresented[top] !=');
      expect(body).toContain('mediumTransmissionPaid[mediumDepth] = 0u;');
      expect(body).toContain('mediumBoundary[lower] = mediumBoundary[upper];');
      expect(body).toContain('mediumRepresented[lower] = mediumRepresented[upper];');
      expect(body).not.toContain('containmentStep');
      expect(body).not.toContain('nextContainmentOrigin');
    }
    expect(PROBE_RAY_CAST_WGSL).not.toContain('fn rcBeginContainmentReplay(');
    expect(PROBE_RAY_CAST_WGSL).not.toContain('fn rcSceneRootBounds(');
    expect(suffix).toContain('currentBoundary = nextEvent.encodedBoundaryId;');
    expect(suffix).toContain('currentRepresented = nextEvent.representedPrimitiveInstanceId;');
    expect(shadow).toContain('return vec3f(0.0);');
  });

  it('keeps resolved glass/opaque intervals terminal while empty intervals merge once', () => {
    expect(CASCADE_MERGE_WGSL).toContain('if (local.a > 0.5) { return; }');
    expect(CASCADE_MERGE_WGSL).toContain(
      'rc_lowerCascade[lowerOutIdx] = vec4f(local.rgb + merged, 1.0);',
    );
  });
});
