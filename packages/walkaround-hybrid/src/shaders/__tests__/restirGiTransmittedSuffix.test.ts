import { describe, expect, it } from 'vitest';

import { WGSL_MODULES } from '../../pipeline/wgslModules.js';
import { composeWgsl } from '../../pipeline/wgslComposer.js';
import { RIS_GI_MODULE, RIS_GI_WGSL } from '../risGi.wgsl.js';

type Rgb = readonly [number, number, number];
type OpticalOutcome = 'reflection' | 'transmission';

function add(a: Rgb, b: Rgb): Rgb {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function scale(value: Rgb, factor: number): Rgb {
  return [value[0] * factor, value[1] * factor, value[2] * factor];
}

function expectedMaterialClosure(
  emission: Rgb,
  opaque: Rgb,
  reflected: Rgb,
  transmitted: Rgb,
  transmission: number,
  interfaceReflectance: number,
  interfaceTransmittance: number,
): Rgb {
  return add(
    emission,
    add(
      scale(opaque, 1 - transmission),
      add(
        scale(reflected, interfaceReflectance),
        scale(
          transmitted,
          transmission * interfaceTransmittance,
        ),
      ),
    ),
  );
}

function envelopeProbability(
  outcome: OpticalOutcome,
  transmissionPhysicalWeight: number,
): number {
  const pTransmission = transmissionPhysicalWeight
    / (1 + transmissionPhysicalWeight);
  return outcome === 'transmission'
    ? pTransmission
    : 1 - pTransmission;
}

function envelopeEstimator(
  outcome: OpticalOutcome,
  emission: Rgb,
  opaque: Rgb,
  reflected: Rgb,
  transmitted: Rgb,
  authoredTransmission: number,
  interfaceReflectance: number,
  interfaceTransmittance: number,
  scalarAlreadyPaid = false,
): Rgb {
  const transmissionPhysicalWeight = scalarAlreadyPaid
    ? 1
    : authoredTransmission;
  const pdf = envelopeProbability(outcome, transmissionPhysicalWeight);
  if (outcome === 'reflection') {
    const opaqueWeight = scalarAlreadyPaid ? 0 : 1 - authoredTransmission;
    return add(
      emission,
      scale(
        add(
          scale(opaque, opaqueWeight),
          scale(reflected, interfaceReflectance),
        ),
        1 / pdf,
      ),
    );
  }
  return add(
    emission,
    scale(
      transmitted,
      transmissionPhysicalWeight * interfaceTransmittance / pdf,
    ),
  );
}

interface MediumBoundary {
  readonly materialId: number;
  readonly instanceId: number;
  readonly frontFacing: boolean;
  readonly authoredTransmission: number;
  readonly baseThickness: number;
  readonly mappedThicknessScale: number;
  readonly alphaAccepted?: boolean;
}

interface MediumState {
  readonly materialId: number;
  readonly instanceId: number;
  readonly transmissionPaid: boolean;
}

function sameMedium(a: MediumState, b: MediumBoundary): boolean {
  return a.materialId === b.materialId && a.instanceId === b.instanceId;
}

function isBulkBoundary(boundary: MediumBoundary): boolean {
  return boundary.authoredTransmission > 0 && boundary.baseThickness > 0;
}

function boundaryEnvelope(
  stack: readonly MediumState[],
  boundary: MediumBoundary,
): { transmissionWeight: number; opaqueWeight: number } {
  const top = stack.at(-1);
  const pairedPaidExit = isBulkBoundary(boundary)
    && !boundary.frontFacing
    && top !== undefined
    && sameMedium(top, boundary)
    && top.transmissionPaid;
  return {
    transmissionWeight: pairedPaidExit ? 1 : boundary.authoredTransmission,
    opaqueWeight: pairedPaidExit ? 0 : 1 - boundary.authoredTransmission,
  };
}

function applyBoundaryEvent(
  stack: readonly MediumState[],
  boundary: MediumBoundary,
  event: OpticalOutcome,
): readonly MediumState[] {
  // Topology is independent of the local mapped-thickness texel. Reflection
  // never mutates or marks the stack.
  if (!isBulkBoundary(boundary) || event === 'reflection') return stack;
  if (boundary.frontFacing) {
    return [...stack, {
      materialId: boundary.materialId,
      instanceId: boundary.instanceId,
      transmissionPaid: true,
    }];
  }
  const top = stack.at(-1);
  if (!top || !sameMedium(top, boundary)) {
    throw new Error('non-LIFO bulk exit');
  }
  return stack.slice(0, -1);
}

function seedContainmentStack(
  crossings: readonly MediumBoundary[],
): readonly MediumState[] {
  const state = new Map<string, { balance: number; minimum: number }>();
  const innerToOuter: MediumState[] = [];
  for (const crossing of crossings) {
    // This is the semantic policy implemented by the textured first-hit scan:
    // rejected alpha boundaries never reach the winding classifier.
    if (crossing.alphaAccepted === false || !isBulkBoundary(crossing)) continue;
    const key = `${crossing.materialId}:${crossing.instanceId}`;
    const winding = state.get(key) ?? { balance: 0, minimum: 0 };
    const nextBalance = winding.balance + (crossing.frontFacing ? 1 : -1);
    if (nextBalance < winding.minimum) {
      innerToOuter.push({
        materialId: crossing.materialId,
        instanceId: crossing.instanceId,
        transmissionPaid: false,
      });
      winding.minimum = nextBalance;
    }
    winding.balance = nextBalance;
    state.set(key, winding);
  }
  for (const winding of state.values()) {
    if (winding.balance !== winding.minimum) {
      throw new Error('open or malformed containment winding');
    }
  }
  return innerToOuter.reverse();
}

function mappedBeerTransfer(
  baseReference: number,
  baseThickness: number,
  mapScale: number,
  segmentDistance: number,
): { rgb: number; spectralDistance: number } {
  const mappedReference = mapScale <= 0
    ? 1
    : baseReference ** mapScale;
  return {
    rgb: mappedReference ** (segmentDistance / baseThickness),
    spectralDistance: segmentDistance * mapScale,
  };
}

function segmentSingleScatter(
  absorption: Rgb,
  scattering: Rgb,
  albedo: Rgb,
  distance: number,
  incident: Rgb,
): Rgb {
  const transmittance = absorption.map((value, channel) => (
    value * Math.exp(-scattering[channel]! * distance)
  )) as unknown as Rgb;
  const luminance = 0.2126 * incident[0]
    + 0.7152 * incident[1]
    + 0.0722 * incident[2];
  return transmittance.map((value, channel) => {
    const sigmaA = -Math.log(Math.max(absorption[channel]!, 1e-30))
      / distance;
    const sigmaT = sigmaA + scattering[channel]!;
    const scatterAlbedo = sigmaT > 0 ? scattering[channel]! / sigmaT : 0;
    const phase = 1 / (4 * Math.PI);
    const sourceScale = albedo[channel]! * scatterAlbedo
      * (1 - value) * phase;
    return value * incident[channel]! + sourceScale * luminance;
  }) as unknown as Rgb;
}

function pcgStep(state: number): { state: number; output: number } {
  const nextState = (
    Math.imul(state, 747796405) + 2891336453
  ) >>> 0;
  const shift = (nextState >>> 28) + 4;
  const word = Math.imul(
    ((nextState >>> shift) ^ nextState),
    277803737,
  );
  return {
    state: nextState,
    output: ((word >>> 22) ^ word) >>> 0,
  };
}

function correlatedSuffixSeeds(parentState: number): {
  parentAfter: number;
  channels: readonly [number, number, number];
} {
  const step = pcgStep(parentState);
  return {
    parentAfter: step.state,
    channels: [step.output, step.output, step.output],
  };
}

function boundedInterfaceWalk(
  events: readonly OpticalOutcome[],
  budget: number,
): { consumed: number; completed: boolean } {
  let consumed = 0;
  for (const event of events) {
    if (consumed >= budget) return { consumed, completed: false };
    consumed += 1;
    if (event === 'transmission') return { consumed, completed: true };
  }
  return { consumed, completed: false };
}

function inspectTerminalAfterInterfaces(
  boundaries: readonly ('dielectric' | 'opaque')[],
  budget: number,
): { consumed: number; opaqueReturned: boolean; overflowed: boolean } {
  let consumed = 0;
  for (const boundary of boundaries) {
    // Terminal material classification occurs before the interface-budget
    // guard, so an opaque receiver reached by the final allowed interface is
    // still returned.
    if (boundary === 'opaque') {
      return { consumed, opaqueReturned: true, overflowed: false };
    }
    if (consumed >= budget) {
      return { consumed, opaqueReturned: false, overflowed: true };
    }
    consumed += 1;
  }
  return { consumed, opaqueReturned: false, overflowed: false };
}

describe('ordinary ReSTIR-GI dielectric suffix closure', () => {
  it('unbiasedly reconstructs persistent reflection, opaque share, and t-weighted transmission', () => {
    const emission: Rgb = [0.2, 0.1, 0.05];
    const opaque: Rgb = [0.8, 0.4, 0.2];
    const reflected: Rgb = [0.3, 0.6, 0.9];
    const transmitted: Rgb = [0.1, 0.5, 0.7];
    for (const transmission of [0.01, 0.25, 0.7, 1]) {
      for (const [reflectance, transmittance] of [
        [0.04, 0.96],
        [0.35, 0.4],
        [1, 0],
      ] as const) {
        let expectation: Rgb = [0, 0, 0];
        for (const outcome of ['reflection', 'transmission'] as const) {
          const probability = envelopeProbability(outcome, transmission);
          expectation = add(
            expectation,
            scale(envelopeEstimator(
              outcome,
              emission,
              opaque,
              reflected,
              transmitted,
              transmission,
              reflectance,
              transmittance,
            ), probability),
          );
        }
        const expected = expectedMaterialClosure(
          emission,
          opaque,
          reflected,
          transmitted,
          transmission,
          reflectance,
          transmittance,
        );
        for (const channel of [0, 1, 2] as const) {
          expect(expectation[channel]).toBeCloseTo(expected[channel], 12);
        }
      }
    }
  });

  it('retains Fresnel/TIR reflection at fully authored transmission', () => {
    const emission: Rgb = [0.1, 0.1, 0.1];
    const opaque: Rgb = [0.7, 0.5, 0.3];
    const reflected: Rgb = [0.9, 0.8, 0.6];
    const transmitted: Rgb = [0.2, 0.4, 0.8];
    const expected = expectedMaterialClosure(
      emission, opaque, reflected, transmitted, 1, 1, 0,
    );
    let expectation: Rgb = [0, 0, 0];
    for (const outcome of ['reflection', 'transmission'] as const) {
      expectation = add(expectation, scale(
        envelopeEstimator(
          outcome,
          emission,
          opaque,
          reflected,
          transmitted,
          1,
          1,
          0,
        ),
        envelopeProbability(outcome, 1),
      ));
    }
    for (const channel of [0, 1, 2] as const) {
      expect(expectation[channel]).toBeCloseTo(expected[channel], 12);
    }
    expect(expected).toEqual(add(emission, reflected));
  });

  it('pays scalar transmission once on entry and never again on the paired exit', () => {
    const entry: MediumBoundary = {
      materialId: 7,
      instanceId: 11,
      frontFacing: true,
      authoredTransmission: 0.4,
      baseThickness: 0.4,
      mappedThicknessScale: 0,
    };
    const exit: MediumBoundary = {
      ...entry,
      frontFacing: false,
      authoredTransmission: 0.15,
      mappedThicknessScale: 1,
    };
    let stack: readonly MediumState[] = [];
    expect(boundaryEnvelope(stack, entry)).toEqual({
      transmissionWeight: 0.4,
      opaqueWeight: 0.6,
    });
    stack = applyBoundaryEvent(stack, entry, 'transmission');
    expect(stack).toEqual([{
      materialId: 7,
      instanceId: 11,
      transmissionPaid: true,
    }]);
    expect(boundaryEnvelope(stack, exit)).toEqual({
      transmissionWeight: 1,
      opaqueWeight: 0,
    });
    const reflected = applyBoundaryEvent(stack, exit, 'reflection');
    expect(reflected).toEqual(stack);
    stack = applyBoundaryEvent(stack, exit, 'transmission');
    expect(stack).toEqual([]);
  });

  it('seeds nested containment as unpaid, ignores cutout and opaque thick shells, and enforces LIFO', () => {
    const boundary = (
      materialId: number,
      instanceId: number,
      frontFacing: boolean,
      overrides: Partial<MediumBoundary> = {},
    ): MediumBoundary => ({
      materialId,
      instanceId,
      frontFacing,
      authoredTransmission: 0.6,
      baseThickness: 1,
      mappedThicknessScale: 1,
      ...overrides,
    });
    const crossings = [
      boundary(90, 1, false, { alphaAccepted: false }),
      boundary(80, 1, false, { authoredTransmission: 0 }),
      boundary(2, 9, false),
      boundary(1, 9, false),
      boundary(3, 5, true),
      boundary(3, 5, false),
    ] as const;
    const seeded = seedContainmentStack(crossings);
    expect(seedContainmentStack(crossings)).toEqual(seeded);
    expect(seeded).toEqual([
      { materialId: 1, instanceId: 9, transmissionPaid: false },
      { materialId: 2, instanceId: 9, transmissionPaid: false },
    ]);
    expect(boundaryEnvelope(seeded, boundary(2, 9, false))).toEqual({
      transmissionWeight: 0.6,
      opaqueWeight: 0.4,
    });
    const afterReflection = applyBoundaryEvent(
      seeded, boundary(2, 9, false), 'reflection',
    );
    expect(afterReflection).toEqual(seeded);
    const afterExit = applyBoundaryEvent(
      seeded, boundary(2, 9, false), 'transmission',
    );
    expect(afterExit).toEqual([
      { materialId: 1, instanceId: 9, transmissionPaid: false },
    ]);
    expect(() => applyBoundaryEvent(
      seeded, boundary(1, 9, false), 'transmission',
    )).toThrow('non-LIFO bulk exit');
  });

  it('attenuates the incoming segment of an inside-volume suffix before paying its first exit scalar', () => {
    const seeded: readonly MediumState[] = [{
      materialId: 4,
      instanceId: 8,
      transmissionPaid: false,
    }];
    const exit: MediumBoundary = {
      materialId: 4,
      instanceId: 8,
      frontFacing: false,
      authoredTransmission: 0.35,
      baseThickness: 0.5,
      mappedThicknessScale: 0.4,
    };
    const firstSegmentDistance = 1.25;
    const absorption = mappedBeerTransfer(
      0.25,
      exit.baseThickness,
      exit.mappedThicknessScale,
      firstSegmentDistance,
    ).rgb;
    const scatterExtinction = Math.exp(-0.2 * firstSegmentDistance);
    const preExitThroughput = absorption * scatterExtinction;
    const firstEmission = 3.5;
    const attenuatedFirstEmission = preExitThroughput * firstEmission;
    const envelope = boundaryEnvelope(seeded, exit);
    expect(preExitThroughput).toBeCloseTo(
      0.25 ** (0.4 * firstSegmentDistance / 0.5)
        * Math.exp(-0.2 * firstSegmentDistance),
      12,
    );
    expect(envelope).toEqual({
      transmissionWeight: 0.35,
      opaqueWeight: 0.65,
    });
    expect(preExitThroughput * envelope.transmissionWeight).toBeCloseTo(
      preExitThroughput * 0.35,
      12,
    );
    expect(attenuatedFirstEmission).toBeCloseTo(
      firstEmission
        * 0.25 ** (0.4 * firstSegmentDistance / 0.5)
        * Math.exp(-0.2 * firstSegmentDistance),
      12,
    );
    expect(attenuatedFirstEmission).not.toBe(firstEmission);
    expect(applyBoundaryEvent(seeded, exit, 'transmission')).toEqual([]);
  });

  it('keeps authored topology while applying thickness-map scale once to RGB and spectral absorption', () => {
    const baseReference = 0.25;
    const baseThickness = 0.5;
    const segmentDistance = 1.25;
    for (const mapScale of [0, 0.2, 1]) {
      const transfer = mappedBeerTransfer(
        baseReference, baseThickness, mapScale, segmentDistance,
      );
      expect(transfer.rgb).toBeCloseTo(
        baseReference ** (mapScale * segmentDistance / baseThickness),
        12,
      );
      expect(transfer.spectralDistance).toBeCloseTo(
        segmentDistance * mapScale,
        12,
      );
      const topologyBoundary: MediumBoundary = {
        materialId: 1,
        instanceId: 2,
        frontFacing: true,
        authoredTransmission: 1,
        baseThickness,
        mappedThicknessScale: mapScale,
      };
      expect(isBulkBoundary(topologyBoundary)).toBe(true);
    }
    expect(mappedBeerTransfer(
      baseReference, baseThickness, 0, segmentDistance,
    )).toEqual({ rgb: 1, spectralDistance: 0 });
  });

  it('carries cross-channel single in-scatter instead of extinction alone', () => {
    const incident: Rgb = [0, 4, 0];
    const result = segmentSingleScatter(
      [0.8, 0.7, 0.6],
      [0.4, 0, 0],
      [1, 0, 0],
      2,
      incident,
    );
    // A scalar per-channel throughput would leave red at exactly zero. The
    // 3x3 source operator converts incident luminance into red in-scatter.
    expect(result[0]).toBeGreaterThan(0);
    expect(result[1]).toBeCloseTo(2.8, 12);
    expect(result[2]).toBe(0);
  });

  it('clones one parent RNG draw into correlated RGB suffix streams', () => {
    const first = correlatedSuffixSeeds(0x12345678);
    expect(first.channels[0]).toBe(first.channels[1]);
    expect(first.channels[1]).toBe(first.channels[2]);
    expect(first.parentAfter).not.toBe(0x12345678);
    // Parent advancement is independent of how much work any channel performs.
    expect(correlatedSuffixSeeds(0x12345678)).toEqual(first);
  });

  it('bounds reciprocal reflection chains without mutating medium topology', () => {
    const maxInterfaces = 8;
    expect(boundedInterfaceWalk(
      Array.from({ length: 20 }, () => 'reflection' as const),
      maxInterfaces,
    )).toEqual({ consumed: 8, completed: false });
    expect(boundedInterfaceWalk([
      'reflection', 'reflection', 'reflection', 'reflection',
      'reflection', 'reflection', 'reflection', 'transmission',
    ], maxInterfaces)).toEqual({ consumed: 8, completed: true });
  });

  it('returns the opaque receiver after interface eight but rejects interface nine', () => {
    expect(inspectTerminalAfterInterfaces([
      'dielectric', 'dielectric', 'dielectric', 'dielectric',
      'dielectric', 'dielectric', 'dielectric', 'dielectric',
      'opaque',
    ], 8)).toEqual({
      consumed: 8,
      opaqueReturned: true,
      overflowed: false,
    });
    expect(inspectTerminalAfterInterfaces([
      'dielectric', 'dielectric', 'dielectric', 'dielectric',
      'dielectric', 'dielectric', 'dielectric', 'dielectric',
      'dielectric',
    ], 8)).toEqual({
      consumed: 8,
      opaqueReturned: false,
      overflowed: true,
    });
  });

  it('composes the conditional anisotropic lobes and local-only suffix path', () => {
    const composed = composeWgsl(RIS_GI_MODULE, WGSL_MODULES);
    expect(composed).toContain('fn restirGiSampleDielectricLobe(');
    expect(composed).toContain('fn ggxSampleVndfTangentAnisotropic(');
    expect(composed).toContain('var mediumTransmissionPaid: array<u32, 8>;');
    expect(composed).toContain('traceSceneFirstHitAlphaMaskTextured(');
    expect(composed).toContain(
      'mediumThicknessMapScale[top],\n      firstSegmentDistance,',
    );
    expect(composed).toContain(
      'segmentThicknessMapScale,\n        segmentDistance,',
    );
    expect(composed).toContain(
      'bounceHit.dist,\n          shadingNs,',
    );
    expect(composed).toContain('fn restirGiSuffixSegmentTransfer(');
    expect(composed).toContain('var prefixTransfer =');
    expect(composed).toContain('sourceScale * 0.2126');
    expect(composed).toContain(
      'prefixTransfer, firstEmissionLo, channel,',
    );
    expect(composed).toContain(
      'scatteredEmissionLo, rawEmissionLo, transmission > 0.0',
    );
    expect(composed).toContain(
      'scatteredOpaqueLo, rawOpaqueLo, transmission > 0.0',
    );
    expect(composed).toContain('xsPayload.emissionLo,');
    expect(composed).toContain('Lo = dielectricLo;');
    expect(composed).not.toContain(
      'Lo = xsPayload.emissionLo + dielectricLo;',
    );
    expect(composed).toContain('var channelRng = suffixRngSeed;');

    const branchStart = RIS_GI_WGSL.indexOf('let xsTransmission = clamp(');
    const insertStart = RIS_GI_WGSL.indexOf(
      'updateReservoirGIWithMetadata(', branchStart,
    );
    expect(branchStart).toBeGreaterThanOrEqual(0);
    expect(insertStart).toBeGreaterThan(branchStart);
    expect(RIS_GI_WGSL.slice(branchStart, insertStart)).toContain(
      'sampleFlags = GI_SAMPLE_FLAG_LOCAL_ESTIMATOR;',
    );
  });
});
